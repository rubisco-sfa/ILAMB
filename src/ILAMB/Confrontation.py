import glob
import logging
import os
import re

import cftime as cf
import numpy as np
import pylab as plt
from mpi4py import MPI
from netCDF4 import Dataset
from sympy import sympify

from ILAMB import Post as post
from ILAMB import ilamblib as il
from ILAMB.constants import space_opts, time_opts
from ILAMB.Regions import Regions
from ILAMB.Relationship import Relationship
from ILAMB.Variable import Variable

logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)


def getVariableList(dataset):
    """Extracts the list of variables in the dataset that aren't
    dimensions or the bounds of dimensions.

    """
    variables = [
        v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()
    ]
    for d in dataset.dimensions.keys():
        try:
            variables.pop(variables.index(dataset.variables[d].getncattr("bounds")))
        except:
            pass
    return variables


def replace_url(string):
    url = re.findall(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+~]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        string,
    )
    for u in url:
        u_text = u
        u_text = u_text.replace("https://doi.org/doi:", "doi:")
        u_text = u_text.replace("https://doi.org/", "doi:")
        u_text = u_text.replace("http://doi.org/doi:", "doi:")
        u_text = u_text.replace("http://doi.org/", "doi:")
        string = string.replace(u, "<a href='%s'>%s</a>" % (u, u_text))
    # if no https link was found, then it may be a doi link which we
    # want to hyperlink appropriately
    if len(url) == 0:
        url = re.findall(
            "doi:(?:[a-zA-Z]|[0-9]|[$-_@.&+~]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            string,
        )
        for u in url:
            # if doi is in a text reference, it ends with a period that we do not want
            u = u.strip(".")
            u_link = u.replace("doi:", "https://doi.org/")
            string = string.replace(u, "<a href='%s'>%s</a>" % (u_link, u))
    return string


def parse_bibtex(string):
    citation = ""
    entry_begins = []
    for m in re.finditer("@", string):
        entry_begins.append(m.start())
    entry_begins.append(len(string))
    for i in range(len(entry_begins) - 1):
        entry = string[entry_begins[i] : entry_begins[i + 1]]
        e = dict(re.findall(r"\s+(\w+)\s+=\s+\{(.*)\}", entry))
        if i > 0:
            citation += "<br>"
        citation += "<dd>"
        if "author" in e:
            citation += e["author"]
        if "year" in e:
            citation += " (%s)" % e["year"]
        if "title" in e:
            citation += ", %s" % e["title"]
        if "journal" in e:
            citation += ", <i>%s</i>" % e["journal"]
        if "number" in e:
            citation += ", <i>%s</i>" % e["number"]
        if "page" in e:
            citation += ", %s" % e["page"]
        if "doi" in e:
            citation += ", %s" % replace_url(e["doi"])
        citation += "</dd>"
    return citation


def create_data_header(attr, val):
    vals = val.split(";")
    html = "<p><dl><dt><b>&nbsp;&nbsp;%s:</dt></b>" % (attr.capitalize())
    for v in vals:
        v = v.strip()
        if v.startswith("@"):
            html += parse_bibtex(v)
        else:
            html += "<dd>%s</dd>" % (replace_url(v))
    html += "</dl></p>"
    return html


class Confrontation(object):
    """A generic class for confronting model results with observational data.

    This class is meant to provide the user with a simple way to
    specify observational datasets and compare them to model
    results. A generic analysis routine is called which checks mean
    states of the variables, afterwhich the results are tabulated and
    plotted automatically. A HTML page is built dynamically as plots
    are created based on available information and successful
    analysis.

    Parameters
    ----------
    name : str
        a name for the confrontation
    source : str
        full path to the observational dataset
    variable_name : str
        name of the variable to extract from the source dataset
    output_path : str, optional
        path into which all output from this confrontation will be generated
    alternate_vars : list of str, optional
        other accepted variable names when extracting from models
    derived : str, optional
        an algebraic expression which captures how the confrontation variable may be generated
    regions : list of str, optional
        a list of regions over which the spatial analysis will be performed (default is global)
    table_unit : str, optional
        the unit to use in the output HTML table
    plot_unit : str, optional
        the unit to use in the output images
    space_mean : bool, optional
        enable to take spatial means (as opposed to spatial integrals) in the analysis (enabled by default)
    relationships : list of ILAMB.Confrontation.Confrontation, optional
        a list of confrontations with whose data we use to study relationships
    cmap : str, optional
        the colormap to use in rendering plots (default is 'jet')
    land : str, bool
        enable to force the masking of areas with no land (default is False)
    limit_type : str
        change the types of plot limits, one of ['minmax', '99per' (default)]
    """

    def __init__(self, **keywords):
        # Initialize
        self.master = True
        self.name = keywords.get("name", None)
        self.source = keywords.get("source", None)
        self.variable = keywords.get("variable", None)
        self.output_path = keywords.get("output_path", "./")
        self.alternate_vars = keywords.get("alternate_vars", [])
        self.derived = keywords.get("derived", None)
        self.regions = list(keywords.get("regions", ["global"]))
        self.data = None
        self.cmap = keywords.get("cmap", "jet")
        self.land = keywords.get("land", False)
        self.limits = None
        self.longname = self.output_path
        self.longname = "/".join(
            self.longname.replace("//", "/").rstrip("/").split("/")[-2:]
        )
        self.table_unit = keywords.get("table_unit", None)
        self.plot_unit = keywords.get("plot_unit", None)
        self.space_mean = keywords.get("space_mean", True)
        self.relationships = keywords.get("relationships", None)
        self.df_errs = keywords.get("df_errs", None)
        self.keywords = keywords
        self.extents = np.asarray([[-90.0, +90.0], [-180.0, +180.0]])
        self.study_limits = []
        self.cweight = 1
        self.scale_factor = float(keywords.get("scale_factor", 1.0))

        # Make sure the source data exists

        if not os.path.isfile(self.source):
            msg = (
                "\n\nI am looking for data for the %s confrontation here\n\n"
                % self.name
            )
            msg += "%s\n\nbut I cannot find it. " % self.source
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?\n"
            logger.debug(f"[{self.longname}] {msg}")
            raise il.MisplacedData(msg)

        # Setup a html layout for generating web views of the results
        pages = []

        # Mean State page
        pages.append(post.HtmlPage("MeanState", "Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(
            ["Temporally integrated period mean", "Spatially integrated regional mean"]
        )

        # Datasites page
        self.hasSites = False
        self.lbls = None
        y0 = None
        yf = None
        with Dataset(self.source) as dataset:
            if "data" in dataset.dimensions:
                # self.hasSites = True
                if "site_name" in dataset.ncattrs():
                    self.lbls = dataset.site_name.split(",")
                else:
                    self.lbls = [
                        "site%d" % s for s in range(len(dataset.dimensions["data"]))
                    ]
            if "time" in dataset.dimensions:
                t = dataset.variables["time"]
                tdata = t[[0, -1]]
                if "bounds" in t.ncattrs():
                    tdata = dataset.variables[t.bounds]
                    tdata = [tdata[0, 0], tdata[-1, 1]]
                tdata = cf.num2date(tdata, units=t.units, calendar=t.calendar)
                y0 = tdata[0].year
                yf = tdata[1].year

        if self.hasSites:
            pages.append(post.HtmlSitePlotsPage("SitePlots", "Site Plots"))
            pages[-1].setHeader("CNAME / RNAME / MNAME")
            pages[-1].setSections([])
            var = Variable(
                filename=self.source,
                variable_name=self.variable,
                alternate_vars=self.alternate_vars,
            ).integrateInTime(mean=True)
            if self.plot_unit is not None:
                var.convert(self.plot_unit)
            pages[-1].lat = var.lat
            pages[-1].lon = var.lon
            pages[-1].vname = self.variable
            pages[-1].unit = var.unit
            pages[-1].vals = var.data
            pages[-1].sites = self.lbls

        # Relationships page
        if self.relationships is not None:
            pages.append(post.HtmlPage("Relationships", "Relationships"))
            pages[-1].setHeader("CNAME / RNAME / MNAME")
            pages[-1].setSections(list(self.relationships))
        pages.append(post.HtmlAllModelsPage("AllModels", "All Models"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation", "Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        with Dataset(self.source) as dset:

            def _attribute_sort(attr):
                # If the attribute begins with one of the ones we
                # specifically order, return the index into order. If
                # it does not, return the number of entries in the
                # list and the file's order will be preserved.
                order = [
                    "title",
                    "version",
                    "institution",
                    "source",
                    "history",
                    "references",
                    "comments",
                    "convention",
                ]
                for i, a in enumerate(order):
                    if attr.lower().startswith(a):
                        return i
                return len(order)

            attrs = dset.ncattrs()
            attrs = sorted(attrs, key=_attribute_sort)

            for attr in attrs:
                try:
                    val = dset.getncattr(attr)
                    if type(val) != str:
                        val = str(val)
                    pages[-1].text += create_data_header(attr, val)
                except:
                    pass
        self.layout = post.HtmlLayout(pages, self.longname, years=(y0, yf))

        # Define relative weights of each score in the overall score
        # (FIX: need some way for the user to modify this)
        self.weight = {
            "Bias Score": 1.0,
            "RMSE Score": 2.0,
            "Seasonal Cycle Score": 1.0,
            "Interannual Variability Score": 1.0,
            "Spatial Distribution Score": 1.0,
        }

    def requires(self):
        if self.derived is not None:
            ands = [arg.name for arg in sympify(self.derived).free_symbols]
            ors = []
        else:
            ands = []
            ors = [self.variable] + self.alternate_vars
        return ands, ors

    def stageData(self, m):
        r"""Extracts model data which matches the observational dataset.

        The datafile associated with this confrontation defines what
        is to be extracted from the model results. If the
        observational data represents sites, as opposed to spatially
        defined over a latitude/longitude grid, then the model results
        will be sampled at the site locations to match. The spatial
        grids need not align, the analysis will handle the
        interpolations when necesary.

        If both datasets are defined on the same temporal scale, then
        the maximum overlap time is computed and the datasets are
        clipped to match. If there is some disparity in the temporal
        scale (e.g. annual mean observational data and monthly mean
        model results) then we coarsen the model results automatically
        to match the observational data.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model result context

        Returns
        -------
        obs : ILAMB.Variable.Variable
            the variable context associated with the observational dataset
        mod : ILAMB.Variable.Variable
            the variable context associated with the model result
        """
        obs = Variable(
            filename=self.source,
            variable_name=self.variable,
            alternate_vars=self.alternate_vars,
            t0=None if len(self.study_limits) != 2 else self.study_limits[0],
            tf=None if len(self.study_limits) != 2 else self.study_limits[1],
        )
        with np.errstate(all="ignore"):
            obs.data *= self.scale_factor
        if obs.time is None:
            raise il.NotTemporalVariable()
        self.pruneRegions(obs)

        # The reference might be layered and we want to extract a
        # slice to compare against models
        if "depth" in self.keywords and obs.layered:
            obs.trim(d=[self.keywords["depth"] - 0.01, self.keywords["depth"] + 0.01])
            if obs.depth.size > 1:
                obs = obs.integrateInDepth(mean=True)
                shp = list(obs.data.shape)
                shp.insert(1, 1)
                obs.data.shape = shp
                obs.depth = np.asarray([self.keywords["depth"]])
                obs.depth_bnds = np.asarray(
                    [[self.keywords["depth"] - 0.01, self.keywords["depth"] + 0.01]]
                )
                obs.layered = True
                obs.name = self.variable

        # Try to extract a commensurate quantity from the model
        mod = m.extractTimeSeries(
            self.variable,
            alt_vars=self.alternate_vars,
            expression=self.derived,
            initial_time=obs.time_bnds[0, 0],
            final_time=obs.time_bnds[-1, 1],
            lats=None if obs.spatial else obs.lat,
            lons=None if obs.spatial else obs.lon,
        )
        obs, mod = il.MakeComparable(
            obs,
            mod,
            mask_ref=True,
            clip_ref=True,
            extents=self.extents,
            logstring="[%s][%s]" % (self.longname, m.name),
        )

        # Check the order of magnitude of the data and convert to help avoid roundoff errors
        def _reduceRoundoffErrors(var):
            if "s-1" in var.unit:
                return var.convert(var.unit.replace("s-1", "d-1"))
            if "kg" in var.unit:
                return var.convert(var.unit.replace("kg", "g"))
            return var

        def _getOrder(var):
            return np.log10(np.abs(var.data).clip(1e-16)).mean()

        order = _getOrder(obs)
        count = 0
        while order < -2 and count < 2:
            obs = _reduceRoundoffErrors(obs)
            order = _getOrder(obs)
            count += 1

        # convert the model data to the same unit
        mod = mod.convert(obs.unit)

        return obs, mod

    def pruneRegions(self, var):
        # remove regions if there is no data from the input variable
        r = Regions()
        self.regions = [region for region in self.regions if r.hasData(region, var)]

    def confront(self, m):
        r"""Confronts the input model with the observational data.

        This routine performs a mean-state analysis the details of
        which may be found in the documentation of
        ILAMB.ilamblib.AnalysisMeanState. If relationship information
        was provided, it will also perform the analysis documented in
        ILAMB.ilamblib.AnalysisRelationship. Output from the analysis
        is stored in a netCDF4 file in the output path.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results
        """
        # Grab the data
        obs, mod = self.stageData(m)

        mod_file = os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name))
        obs_file = os.path.join(self.output_path, "%s_Benchmark.nc" % (self.name,))
        with il.FileContextManager(self.master, mod_file, obs_file) as fcm:
            # Encode some names and colors
            fcm.mod_dset.setncatts(
                {
                    "name": m.name,
                    "color": m.color,
                    "weight": self.cweight,
                    "complete": 0,
                }
            )
            if self.master:
                fcm.obs_dset.setncatts(
                    {
                        "name": "Benchmark",
                        "color": np.asarray([0.5, 0.5, 0.5]),
                        "weight": self.cweight,
                        "complete": 0,
                    }
                )

            # Read in some options and run the mean state analysis
            mass_weighting = self.keywords.get("mass_weighting", False)
            skip_rmse = self.keywords.get("skip_rmse", False)
            skip_iav = self.keywords.get("skip_iav", True)
            skip_cycle = self.keywords.get("skip_cycle", False)
            rmse_score_basis = self.keywords.get("rmse_score_basis", "cycle")
            if obs.spatial:
                il.AnalysisMeanStateSpace(
                    obs,
                    mod,
                    dataset=fcm.mod_dset,
                    regions=self.regions,
                    benchmark_dataset=fcm.obs_dset,
                    table_unit=self.table_unit,
                    plot_unit=self.plot_unit,
                    space_mean=self.space_mean,
                    skip_rmse=skip_rmse,
                    skip_iav=skip_iav,
                    skip_cycle=skip_cycle,
                    mass_weighting=mass_weighting,
                    rmse_score_basis=rmse_score_basis,
                    df_errs=self.df_errs,
                )
            else:
                il.AnalysisMeanStateSites(
                    obs,
                    mod,
                    dataset=fcm.mod_dset,
                    regions=self.regions,
                    benchmark_dataset=fcm.obs_dset,
                    table_unit=self.table_unit,
                    plot_unit=self.plot_unit,
                    space_mean=self.space_mean,
                    skip_rmse=skip_rmse,
                    skip_iav=skip_iav,
                    skip_cycle=skip_cycle,
                    mass_weighting=mass_weighting,
                    df_errs=self.df_errs,
                )
            fcm.mod_dset.setncattr("complete", 1)
            if self.master:
                fcm.obs_dset.setncattr("complete", 1)
        logger.info("[%s][%s] Success" % (self.longname, m.name))

    def determinePlotLimits(self):
        """Determine the limits of all plots which are inclusive of all ranges.

        The routine will open all netCDF files in the output path and
        add the maximum and minimum of all variables which are
        designated to be plotted. If legends are desired for a given
        plot, these are rendered here as well. This routine should be
        called before calling any plotting routine.

        """

        filelist = glob.glob(os.path.join(self.output_path, "*.nc"))
        benchmark_file = [f for f in filelist if "Benchmark" in f]

        # There may be regions in which there is no benchmark data and
        # these should be weeded out. If the plotting phase occurs in
        # the same run as the analysis phase, this is not needed.
        if benchmark_file:
            with Dataset(benchmark_file[0]) as dset:
                if "MeanState" in dset.groups:
                    Vs = getVariableList(dset.groups["MeanState"])
                else:
                    Vs = []
            Vs = [v for v in Vs if "timeint" in v]
            if Vs:
                self.pruneRegions(
                    Variable(
                        filename=benchmark_file[0],
                        variable_name=Vs[0],
                        groupname="MeanState",
                    )
                )

        # Determine the min/max of variables over all models
        limits = {}
        for fname in filelist:
            with Dataset(fname) as dataset:
                if "MeanState" not in dataset.groups:
                    continue
                variables = getVariableList(dataset.groups["MeanState"])
                for vname in variables:
                    var = dataset.groups["MeanState"].variables[vname]
                    if var[...].size <= 1:
                        continue
                    pname = vname.split("_")[0]

                    """If the plot is a time series, it has been averaged over regions
                    already and we need a separate dictionary for the
                    region as well. These can be based on the
                    percentiles from the attributes of the netCDF
                    variables."""
                    if pname in time_opts:
                        region = vname.split("_")[-1]
                        if pname not in limits:
                            limits[pname] = {}
                        if region not in limits[pname]:
                            limits[pname][region] = {}
                            limits[pname][region]["min"] = +1e20
                            limits[pname][region]["max"] = -1e20
                            limits[pname][region]["unit"] = post.UnitStringToMatplotlib(
                                var.getncattr("units")
                            )
                        limits[pname][region]["min"] = min(
                            limits[pname][region]["min"], var.getncattr("min")
                        )
                        limits[pname][region]["max"] = max(
                            limits[pname][region]["max"], var.getncattr("max")
                        )

                    else:
                        """If the plot is spatial, we want to set the limits as a percentile
                        of all data across models and the
                        benchmark. So here we load the data up and in
                        another pass will compute the percentiles."""
                        if pname not in limits:
                            limits[pname] = {}
                            limits[pname]["min"] = +1e20
                            limits[pname]["max"] = -1e20
                            limits[pname]["unit"] = post.UnitStringToMatplotlib(
                                var.getncattr("units")
                            )
                            limits[pname]["data"] = var[...].compressed()
                        else:
                            limits[pname]["data"] = np.hstack(
                                [limits[pname]["data"], var[...].compressed()]
                            )

        # For those limits which we built up data across all models, compute the percentiles
        for pname in limits.keys():
            if "data" in limits[pname] and len(limits[pname]["data"]) > 0:
                limits[pname]["min"], limits[pname]["max"] = np.percentile(
                    limits[pname]["data"], [1, 99]
                )

        # Second pass to plot legends (FIX: only for master?)
        for pname in limits.keys():
            try:
                opts = space_opts[pname]
            except:
                continue

            # Determine plot limits and colormap
            if opts["sym"]:
                vabs = max(abs(limits[pname]["min"]), abs(limits[pname]["max"]))
                limits[pname]["min"] = -vabs
                limits[pname]["max"] = vabs
            if "shift" in pname:
                limits[pname]["min"] = -6
                limits[pname]["max"] = +6
            if "phase" in pname:
                limits[pname]["min"] = 0
                limits[pname]["max"] = 365

            # if a score, force to be [0,1]
            if "score" in pname:
                limits[pname]["min"] = 0
                limits[pname]["max"] = 1

            limits[pname]["cmap"] = opts["cmap"]
            if limits[pname]["cmap"] == "choose":
                limits[pname]["cmap"] = self.cmap
            if "score" in pname:
                limits[pname]["cmap"] = plt.get_cmap(limits[pname]["cmap"])

            # Plot a legend for each key
            if opts["haslegend"]:
                fig, ax = plt.subplots(figsize=(6.8, 1.0), tight_layout=True)
                label = opts["label"]
                if label == "unit":
                    label = limits[pname]["unit"]
                post.ColorBar(
                    ax,
                    vmin=limits[pname]["min"],
                    vmax=limits[pname]["max"],
                    cmap=limits[pname]["cmap"],
                    ticks=opts["ticks"],
                    ticklabels=opts["ticklabels"],
                    label=label,
                )
                fig.savefig(os.path.join(self.output_path, "legend_%s.png" % (pname)))
                plt.close()

        # Determine min/max of relationship variables
        for fname in glob.glob(os.path.join(self.output_path, "*.nc")):
            with Dataset(fname) as dataset:
                for g in dataset.groups.keys():
                    if "relationship" not in g:
                        continue
                    grp = dataset.groups[g]
                    if g not in limits:
                        limits[g] = {}
                        limits[g]["xmin"] = +1e20
                        limits[g]["xmax"] = -1e20
                        limits[g]["ymin"] = +1e20
                        limits[g]["ymax"] = -1e20
                    limits[g]["xmin"] = min(
                        limits[g]["xmin"], grp.variables["ind_bnd"][0, 0]
                    )
                    limits[g]["xmax"] = max(
                        limits[g]["xmax"], grp.variables["ind_bnd"][-1, -1]
                    )
                    limits[g]["ymin"] = min(
                        limits[g]["ymin"], grp.variables["dep_bnd"][0, 0]
                    )
                    limits[g]["ymax"] = max(
                        limits[g]["ymax"], grp.variables["dep_bnd"][-1, -1]
                    )

        self.limits = limits

    def computeOverallScore(self, m):
        """Computes the overall composite score for a given model.

        This routine opens the netCDF results file associated with
        this confrontation-model pair, and then looks for a "scalars"
        group in the dataset as well as any subgroups that may be
        present. For each grouping of scalars, it will blend any value
        with the word "Score" in the name to render an overall score,
        overwriting the existing value if present.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        """

        def _computeOverallScore(scalars):
            """Given a netCDF4 group of scalars, blend them into an overall score"""
            scores = {}
            variables = [
                v
                for v in scalars.variables.keys()
                if "Score" in v and "Overall" not in v
            ]
            for region in self.regions:
                overall_score = 0.0
                sum_of_weights = 0.0
                for v in variables:
                    if region not in v:
                        continue
                    score = v.replace(region, "").strip()
                    weight = 1.0
                    if score in self.weight:
                        weight = self.weight[score]
                    overall_score += weight * scalars.variables[v][...]
                    sum_of_weights += weight
                if np.abs(overall_score) < 1e-12:
                    overall_score = np.nan
                overall_score /= max(sum_of_weights, 1e-12)
                scores["Overall Score %s" % region] = overall_score
            return scores

        fname = os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name))
        if not os.path.isfile(fname):
            return
        with Dataset(fname, mode="r+") as dataset:
            datasets = [
                dataset.groups[grp] for grp in dataset.groups if "scalars" not in grp
            ]
            groups = [grp for grp in dataset.groups if "scalars" not in grp]
            datasets.append(dataset)
            groups.append(None)
            for dset, grp in zip(datasets, groups):
                if "scalars" in dset.groups:
                    scalars = dset.groups["scalars"]
                    score = _computeOverallScore(scalars)
                    for key in score.keys():
                        if key in scalars.variables:
                            scalars.variables[key][0] = score[key]
                        else:
                            Variable(data=score[key], name=key, unit="1").toNetCDF4(
                                dataset, group=grp
                            )

    def compositePlots(self):
        """Renders plots which display information of all models.

        This routine renders plots which contain information from all
        models. Thus only the master process will run this routine,
        and only after all analysis has finished.

        """
        if not self.master:
            return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        models = []
        colors = []
        corr = {}
        std = {}
        cycle = {}
        has_cycle = False
        has_std = False
        for fname in glob.glob(os.path.join(self.output_path, "*.nc")):
            dataset = Dataset(fname)
            if "MeanState" not in dataset.groups:
                continue
            dset = dataset.groups["MeanState"]
            models.append(dataset.getncattr("name"))
            colors.append(dataset.getncattr("color"))
            for region in self.regions:
                if region not in cycle:
                    cycle[region] = []
                key = [
                    v for v in dset.variables.keys() if ("cycle_" in v and region in v)
                ]
                if len(key) > 0:
                    has_cycle = True
                    cycle[region].append(
                        Variable(
                            filename=fname, groupname="MeanState", variable_name=key[0]
                        )
                    )

                if region not in std:
                    std[region] = []
                if region not in corr:
                    corr[region] = []

                key = []
                if "scalars" in dset.groups:
                    key = [
                        v
                        for v in dset.groups["scalars"].variables.keys()
                        if ("Spatial Distribution Score" in v and region in v)
                    ]
                if len(key) > 0:
                    has_std = True
                    sds = dset.groups["scalars"].variables[key[0]]
                    corr[region].append(sds.getncattr("R"))
                    std[region].append(sds.getncattr("std"))

        # composite annual cycle plot
        if has_cycle and len(models) > 2:
            page.addFigure(
                "Spatially integrated regional mean",
                "compcycle",
                "RNAME_compcycle.png",
                side="ANNUAL CYCLE",
                legend=False,
            )

        for region in self.regions:
            if region not in cycle:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 2.8), tight_layout=True)
            for name, color, var in zip(models, colors, cycle[region]):
                dy = 0.05 * (
                    self.limits["cycle"][region]["max"]
                    - self.limits["cycle"][region]["min"]
                )
                var.plot(
                    ax,
                    lw=2,
                    color=color,
                    label=name,
                    ticks=time_opts["cycle"]["ticks"],
                    ticklabels=time_opts["cycle"]["ticklabels"],
                    vmin=self.limits["cycle"][region]["min"] - dy,
                    vmax=self.limits["cycle"][region]["max"] + dy,
                )
                ylbl = time_opts["cycle"]["ylabel"]
                if ylbl == "unit":
                    ylbl = post.UnitStringToMatplotlib(var.unit)
                ax.set_ylabel(ylbl)
            fig.savefig(os.path.join(self.output_path, "%s_compcycle.png" % (region)))
            plt.close()

        # plot legends with model colors (sorted with Benchmark data on top)
        page.addFigure(
            "Spatially integrated regional mean",
            "legend_compcycle",
            "legend_compcycle.png",
            side="MODEL COLORS",
            legend=False,
        )

        def _alphabeticalBenchmarkFirst(key):
            key = key[0].lower()
            if key == "BENCHMARK":
                return "A"
            return key

        tmp = sorted(zip(models, colors), key=_alphabeticalBenchmarkFirst)
        fig, ax = plt.subplots()
        for model, color in tmp:
            ax.plot(0, 0, "o", mew=0, ms=8, color=color, label=model)
        handles, labels = ax.get_legend_handles_labels()
        plt.close()

        ncol = np.ceil(float(len(models)) / 11.0).astype(int)
        if ncol > 0:
            fig, ax = plt.subplots(figsize=(3.0 * ncol, 2.8), tight_layout=True)
            ax.legend(
                handles, labels, loc="upper right", ncol=ncol, fontsize=10, numpoints=1
            )
            ax.axis(False)
            fig.savefig(os.path.join(self.output_path, "legend_compcycle.png"))
            fig.savefig(os.path.join(self.output_path, "legend_spatial_variance.png"))
            fig.savefig(os.path.join(self.output_path, "legend_temporal_variance.png"))
            plt.close()

        # spatial distribution Taylor plot
        if has_std:
            page.addFigure(
                "Temporally integrated period mean",
                "spatial_variance",
                "RNAME_spatial_variance.png",
                side="SPATIAL TAYLOR DIAGRAM",
                legend=False,
            )
            page.addFigure(
                "Temporally integrated period mean",
                "legend_spatial_variance",
                "legend_spatial_variance.png",
                side="MODEL COLORS",
                legend=False,
            )
        if "Benchmark" in models:
            colors.pop(models.index("Benchmark"))
        for region in self.regions:
            if not (region in std and region in corr):
                continue
            if len(std[region]) != len(corr[region]):
                continue
            if len(std[region]) == 0:
                continue
            fig = plt.figure(figsize=(6.0, 6.0))
            post.TaylorDiagram(
                np.asarray(std[region]), np.asarray(corr[region]), 1.0, fig, colors
            )
            fig.savefig(
                os.path.join(self.output_path, "%s_spatial_variance.png" % region)
            )
            plt.close()

    def modelPlots(self, m):
        """For a given model, create the plots of the analysis results.

        This routine will extract plotting information out of the
        netCDF file which results from the analysis and create
        plots. Note that determinePlotLimits should be called before
        this routine.

        """
        self._relationship(m)
        bname = os.path.join(self.output_path, "%s_Benchmark.nc" % (self.name))
        fname = os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name))
        if not os.path.isfile(bname):
            return
        if not os.path.isfile(fname):
            return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        with Dataset(fname) as dataset:
            group = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color = dataset.getncattr("color")
            for vname in variables:
                # is this a variable we need to plot?
                pname = vname.split("_")[0]
                if group.variables[vname][...].size <= 1:
                    continue
                var = Variable(
                    filename=fname, groupname="MeanState", variable_name=vname
                )

                if (var.spatial or (var.ndata is not None)) and not var.temporal:
                    # grab plotting options
                    if pname not in self.limits.keys():
                        continue
                    if pname not in space_opts:
                        continue
                    opts = space_opts[pname]

                    # add to html layout
                    page.addFigure(
                        opts["section"],
                        pname,
                        opts["pattern"],
                        side=opts["sidelbl"],
                        legend=opts["haslegend"],
                    )

                    # plot variable
                    for region in self.regions:
                        ax = var.plot(
                            None,
                            region=region,
                            vmin=self.limits[pname]["min"],
                            vmax=self.limits[pname]["max"],
                            cmap=self.limits[pname]["cmap"],
                        )
                        fig = ax.get_figure()
                        fig.savefig(
                            os.path.join(
                                self.output_path,
                                "%s_%s_%s.png" % (m.name, region, pname),
                            )
                        )
                        plt.close()

                    # Jumping through hoops to get the benchmark plotted and in the html output
                    if self.master and (
                        pname == "timeint" or pname == "phase" or pname == "iav"
                    ):
                        opts = space_opts[pname]

                        # add to html layout
                        page.addFigure(
                            opts["section"],
                            "benchmark_%s" % pname,
                            opts["pattern"].replace("MNAME", "Benchmark"),
                            side=opts["sidelbl"].replace("MODEL", "BENCHMARK"),
                            legend=True,
                        )

                        # plot variable
                        obs = Variable(
                            filename=bname, groupname="MeanState", variable_name=vname
                        )
                        for region in self.regions:
                            ax = obs.plot(
                                None,
                                region=region,
                                vmin=self.limits[pname]["min"],
                                vmax=self.limits[pname]["max"],
                                cmap=self.limits[pname]["cmap"],
                            )
                            fig = ax.get_figure()
                            fig.savefig(
                                os.path.join(
                                    self.output_path,
                                    "Benchmark_%s_%s.png" % (region, pname),
                                )
                            )
                            plt.close()

                if not (var.spatial or (var.ndata is not None)) and var.temporal:
                    # grab the benchmark dataset to plot along with
                    try:
                        obs = Variable(
                            filename=bname, groupname="MeanState", variable_name=vname
                        ).convert(var.unit)
                    except:
                        continue

                    # grab plotting options
                    if pname not in time_opts:
                        continue
                    opts = time_opts[pname]

                    # add to html layout
                    page.addFigure(
                        opts["section"],
                        pname,
                        opts["pattern"],
                        side=opts["sidelbl"],
                        legend=opts["haslegend"],
                    )

                    # plot variable
                    for region in self.regions:
                        if region not in vname:
                            continue
                        fig, ax = plt.subplots(figsize=(6.8, 2.8), tight_layout=True)
                        obs.plot(ax, lw=2, color="k", alpha=0.5)
                        var.plot(
                            ax,
                            lw=2,
                            color=color,
                            label=m.name,
                            ticks=opts["ticks"],
                            ticklabels=opts["ticklabels"],
                        )

                        dy = 0.05 * (
                            self.limits[pname][region]["max"]
                            - self.limits[pname][region]["min"]
                        )
                        ax.set_ylim(
                            self.limits[pname][region]["min"] - dy,
                            self.limits[pname][region]["max"] + dy,
                        )
                        ylbl = opts["ylabel"]
                        if ylbl == "unit":
                            ylbl = post.UnitStringToMatplotlib(var.unit)
                        ax.set_ylabel(ylbl)
                        fig.savefig(
                            os.path.join(
                                self.output_path,
                                "%s_%s_%s.png" % (m.name, region, pname),
                            )
                        )
                        plt.close()

        logger.info("[%s][%s] Success" % (self.longname, m.name))

    def sitePlots(self, m):
        """ """
        if not self.hasSites:
            return

        obs, mod = self.stageData(m)
        for i in range(obs.ndata):
            fig, ax = plt.subplots(figsize=(6.8, 2.8), tight_layout=True)
            tmask = np.where(mod.data.mask[:, i] == False)[0]
            if tmask.size > 0:
                tmin, tmax = tmask[[0, -1]]
            else:
                tmin = 0
                tmax = mod.time.size - 1

            t = mod.time[tmin : (tmax + 1)]
            x = mod.data[tmin : (tmax + 1), i]
            y = obs.data[tmin : (tmax + 1), i]
            ax.plot(t, y, "-k", lw=2, alpha=0.5)
            ax.plot(t, x, "-", color=m.color)

            ind = np.where(t % 365 < 30.0)[0]
            ticks = t[ind] - (t[ind] % 365)
            ticklabels = (ticks / 365.0 + 1850.0).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_ylabel(post.UnitStringToMatplotlib(mod.unit))
            fig.savefig(
                os.path.join(
                    self.output_path, "%s_%s_%s.png" % (m.name, self.lbls[i], "time")
                )
            )
            plt.close()

    def generateHtml(self):
        """Generate the HTML for the results of this confrontation.

        This routine opens all netCDF files and builds a table of
        metrics. Then it passes the results to the HTML generator and
        saves the result in the output directory. This only occurs on
        the confrontation flagged as master.

        """
        # only the master processor needs to do this
        if not self.master:
            return

        for page in self.layout.pages:
            # build the metric dictionary
            metrics = {}
            page.models = []
            for fname in glob.glob(os.path.join(self.output_path, "*.nc")):
                with Dataset(fname) as dataset:
                    mname = dataset.getncattr("name")
                    if mname != "Benchmark":
                        page.models.append(mname)
                    if page.name not in dataset.groups:
                        continue
                    group = dataset.groups[page.name]

                    # if the dataset opens, we need to add the model (table row)
                    metrics[mname] = {}

                    # each model will need to have all regions
                    for region in self.regions:
                        metrics[mname][region] = {}

                    # columns in the table will be in the scalars group
                    if "scalars" not in group.groups:
                        continue

                    # we add scalars to the model/region based on the region
                    # name being in the variable name. If no region is found,
                    # we assume it is the global region.
                    grp = group.groups["scalars"]
                    for vname in grp.variables.keys():
                        found = False
                        for region in self.regions:
                            if vname.endswith(" %s" % region):
                                found = True
                                var = grp.variables[vname]
                                name = "".join(vname.rsplit(" %s" % region, 1))
                                metrics[mname][region][name] = Variable(
                                    name=name, unit=var.units, data=var[...]
                                )
                        if not found:
                            var = grp.variables[vname]
                            if "global" not in metrics[mname]:
                                logger.debug(
                                    "[%s][%s] 'global' not in region list = [%s]"
                                    % (self.longname, mname, ",".join(self.regions))
                                )
                                raise ValueError()
                            metrics[mname]["global"][vname] = Variable(
                                name=vname, unit=var.units, data=var[...]
                            )
            page.setMetrics(metrics)

        # write the HTML page
        with open(
            os.path.join(self.output_path, "%s.html" % (self.name)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(self.layout))

    def _relationship(self, m, nbin=25):
        """ """

        def _retrieveData(filename):
            key = None
            with Dataset(filename, mode="r") as dset:
                key = [
                    v
                    for v in dset.groups["MeanState"].variables.keys()
                    if "timeint_" in v
                ]
            return Variable(
                filename=filename, groupname="MeanState", variable_name=key[0]
            )

        # If there are no relationships to analyze, get out of here
        if self.relationships is None:
            return

        # Get the HTML page
        page = [page for page in self.layout.pages if "Relationships" in page.name]
        if len(page) == 0:
            return
        page = page[0]

        # Try to get the dependent data from the model and obs
        try:
            ref_dep = _retrieveData(
                os.path.join(self.output_path, "%s_%s.nc" % (self.name, "Benchmark"))
            )
            com_dep = _retrieveData(
                os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name))
            )
            dep_name = self.longname.split("/")[0]
            ref_dep.name = "%s/%s" % (dep_name, self.name)
            com_dep.name = "%s/%s" % (dep_name, m.name)
        except:
            return

        with Dataset(
            os.path.join(self.output_path, "%s_%s.nc" % (self.name, m.name)), mode="r+"
        ) as results:
            # Grab/create a relationship and scalars group
            group = None
            if "Relationships" not in results.groups:
                group = results.createGroup("Relationships")
            else:
                group = results.groups["Relationships"]
            if "scalars" not in group.groups:
                scalars = group.createGroup("scalars")
            else:
                scalars = group.groups["scalars"]

            # for each relationship...
            for c in self.relationships:
                # try to get the independent data from the model and obs
                try:
                    ref_ind = _retrieveData(
                        os.path.join(c.output_path, "%s_%s.nc" % (c.name, "Benchmark"))
                    )
                    com_ind = _retrieveData(
                        os.path.join(c.output_path, "%s_%s.nc" % (c.name, m.name))
                    )
                    ind_name = c.longname.split("/")[0]
                    ref_ind.name = "%s/%s" % (ind_name, c.name)
                    com_ind.name = "%s/%s" % (ind_name, m.name)
                except:
                    continue

                # if any one of the data sources are sites, they all
                # should be, also check that the lat/lons are the same
                src = [ref_ind, ref_dep, com_ind, com_dep]
                for i, a in enumerate(src):
                    for j, b in enumerate(src):
                        if i == j:
                            continue
                        if a.ndata and b.ndata:
                            assert np.allclose(a.lat, b.lat) * np.allclose(a.lon, b.lon)
                        if a.ndata and not b.ndata:
                            src[j] = b.extractDatasites(a.lat, a.lon)
                ref_ind, ref_dep, com_ind, com_dep = src

                # create relationships
                ref = Relationship(ref_ind, ref_dep, order=1)
                com = Relationship(com_ind, com_dep, order=1)

                # set limits to global across models
                ref.limits = (
                    [self.limits["timeint"]["min"], self.limits["timeint"]["max"]],
                    [c.limits["timeint"]["min"], c.limits["timeint"]["max"]],
                )
                com.limits = (
                    [self.limits["timeint"]["min"], self.limits["timeint"]["max"]],
                    [c.limits["timeint"]["min"], c.limits["timeint"]["max"]],
                )

                # Add figures to the html page
                page.addFigure(
                    c.longname,
                    "benchmark_rel_%s" % ind_name,
                    "Benchmark_RNAME_rel_%s.png" % ind_name,
                    legend=False,
                    benchmark=False,
                )
                page.addFigure(
                    c.longname,
                    "rel_%s" % ind_name,
                    "MNAME_RNAME_rel_%s.png" % ind_name,
                    legend=False,
                    benchmark=False,
                )
                page.addFigure(
                    c.longname,
                    "rel_func_%s" % ind_name,
                    "MNAME_RNAME_rel_func_%s.png" % ind_name,
                    legend=False,
                    benchmark=False,
                )

                # Analysis over regions
                longname = c.longname.replace(
                    "/", "|"
                )  # we want the source too, but netCDF doesn't like the '/'
                for region in self.regions:
                    ref.makeComparable(com, region=region)

                    # Make the plots
                    fig, ax = plt.subplots(figsize=(6, 5.25), tight_layout=True)
                    ref.plotDistribution(ax, region=region)
                    fig.savefig(
                        os.path.join(
                            self.output_path,
                            "%s_%s_rel_%s.png" % ("Benchmark", region, ind_name),
                        )
                    )
                    plt.close()

                    fig, ax = plt.subplots(figsize=(6, 5.25), tight_layout=True)
                    com.plotDistribution(ax, region=region)
                    fig.savefig(
                        os.path.join(
                            self.output_path,
                            "%s_%s_rel_%s.png" % (m.name, region, ind_name),
                        )
                    )
                    plt.close()

                    fig, ax = plt.subplots(figsize=(6, 5.25), tight_layout=True)
                    com.plotFunction(ax, region=region, shift=-0.1, color=m.color)
                    ref.plotFunction(ax, region=region, shift=+0.1)
                    fig.savefig(
                        os.path.join(
                            self.output_path,
                            "%s_%s_rel_func_%s.png" % (m.name, region, ind_name),
                        )
                    )
                    plt.close()

                    # Score the distribution
                    score = ref.scoreHellinger(com, region=region)
                    sname = "%s Hellinger Distance %s" % (longname, region)
                    if sname in scalars.variables:
                        scalars.variables[sname][0] = score
                    else:
                        Variable(name=sname, unit="1", data=score).toNetCDF4(
                            results, group="Relationships"
                        )

                    # Score the functional response
                    score = ref.scoreRMSE(com, region=region)
                    sname = "%s Score %s" % (longname, region)
                    if sname in scalars.variables:
                        scalars.variables[sname][0] = score
                    else:
                        Variable(name=sname, unit="1", data=score).toNetCDF4(
                            results, group="Relationships"
                        )
