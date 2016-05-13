import ilamblib as il
from Variable import *
from constants import four_code_regions,space_opts,time_opts,mid_months,bnd_months
import os,glob,re
from netCDF4 import Dataset
import Post as post
import pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    """
    def __init__(self,**keywords):
        
        # Initialize
        self.master         = True
        self.name           = keywords.get("name",None)
        self.source         = keywords.get("source",None)
        self.variable       = keywords.get("variable",None)
        self.output_path    = keywords.get("output_path","./" % self.name)
        self.alternate_vars = keywords.get("alternate_vars",[])
        self.derived        = keywords.get("derived",None)
        self.regions        = keywords.get("regions",["global"])
        self.data           = None
        self.cmap           = keywords.get("cmap","jet")
        self.land           = keywords.get("land",False)
        self.limits         = None
        self.longname       = self.output_path
        self.longname       = self.longname.replace("//","/").replace("./","").replace("_build/","")
        if self.longname[-1] == "/": self.longname = self.longname[:-1]
        self.longname       = "/".join(self.longname.split("/")[1:])
        self.table_unit     = keywords.get("table_unit",None)
        self.plot_unit      = keywords.get("plot_unit",None)
        self.space_mean     = keywords.get("space_mean",True)        
        self.relationships  = keywords.get("relationships",None)
        self.keywords       = keywords
        
        # Make sure the source data exists
        try:
            os.stat(self.source)
        except:
            msg  = "\n\nI am looking for data for the %s confrontation here\n\n" % self.name
            msg += "%s\n\nbut I cannot find it. " % self.source
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?\n"
            raise il.MisplacedData(msg)

        # Setup a html layout for generating web views of the results
        self.layout = post.HtmlLayout(self,regions=self.regions)
        self.layout.setHeader("CNAME / RNAME / MNAME")
        self.layout.setSections(["Temporally integrated period mean",
                                 "Spatially integrated regional mean",
                                 "Period Mean Relationships"])

        # Define relative weights of each score in the overall score
        # (FIX: need some way for the user to modify this)
        self.weight = {"Bias Score"                    :1.,
                       "RMSE Score"                    :2.,
                       "Seasonal Cycle Score"          :1.,
                       "Interannual Variability Score" :1.,
                       "Spatial Distribution Score"    :1.}
        
    def stageData(self,m):
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
        if self.data is None:
            obs = Variable(filename       = self.source,
                           variable_name  = self.variable,
                           alternate_vars = self.alternate_vars)
            self.data = obs
        else:
            obs = self.data
        if obs.time is None: raise il.NotTemporalVariable()
        t0 = obs.time_bnds[0, 0]
        tf = obs.time_bnds[1,-1]
        
        if obs.spatial:
            try:
                mod = m.extractTimeSeries(self.variable,
                                          alt_vars     = self.alternate_vars,
                                          initial_time = t0,
                                          final_time   = tf)
            except:
                mod = m.derivedVariable  (self.variable,self.derived,
                                          initial_time = t0,
                                          final_time   = tf)
        else:
            try:
                mod = m.extractTimeSeries(self.variable,
                                          alt_vars     = self.alternate_vars,
                                          lats         = obs.lat,
                                          lons         = obs.lon,
                                          initial_time = t0,
                                          final_time   = tf)
            except:
                mod = m.derivedVariable  (self.variable,self.derived,
                                          lats         = obs.lat,
                                          lons         = obs.lon,
                                          initial_time = t0,
                                          final_time   = tf)

        obs,mod = il.MakeComparable(obs,mod,mask_ref=True,clip_ref=True)
        
        # Check the order of magnitude of the data and convert to help avoid roundoff errors
        def _reduceRoundoffErrors(var):
            if "s-1" in var.unit: return var.convert(var.unit.replace("s-1","d-1"))
            if "kg"  in var.unit: return var.convert(var.unit.replace("kg" ,"g"  ))
            return var
        def _getOrder(var):
            return np.log10(np.abs(var.data).clip(1e-16)).mean()
        order = _getOrder(obs)
        count = 0
        while order < -2 and count < 2:
            obs    = _reduceRoundoffErrors(obs)
            order  = _getOrder(obs)
            count += 1
            
        # convert the model data to the same unit
        mod = mod.convert(obs.unit)

        return obs,mod

    def confront(self,m):
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
        obs,mod = self.stageData(m)

        # Open a dataset for recording the results of this
        # confrontation, record for the benchmark if we are the master
        # process.
        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        benchmark_results = None
        fname = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        if self.master:
            benchmark_results = Dataset(fname,mode="w")
            benchmark_results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})

        # Perform the standard fluxrate analysis
        mass_weighting = self.keywords.get("mass_weighting",False)
        skip_rmse      = self.keywords.get("skip_rmse"     ,False)
        skip_iav       = self.keywords.get("skip_iav"      ,False)
        try:
            il.AnalysisMeanState(obs,mod,dataset   = results,
                                 regions           = self.regions,
                                 benchmark_dataset = benchmark_results,
                                 table_unit        = self.table_unit,
                                 plot_unit         = self.plot_unit,
                                 space_mean        = self.space_mean,
                                 skip_rmse         = skip_rmse,
                                 skip_iav          = skip_iav,
                                 mass_weighting    = mass_weighting)
        except:
            results.close()
            os.system("rm -f %s/%s_%s.nc" % (self.output_path,self.name,m.name))
            if self.master:
                benchmark_results.close()
                os.system("rm -f %s" % fname)
            raise il.AnalysisError()
        
        # Perform relationship analysis
        obs_dep,mod_dep = obs,mod
        dep_name        = self.longname.split("/")[0]
        dep_plot_unit   = self.plot_unit
        if (dep_plot_unit is None): dep_plot_unit = obs_dep.unit

        if self.relationships is not None:
            for c in self.relationships:
                obs_ind,mod_ind = c.stageData(m) # independent variable
                ind_name = c.longname.split("/")[0]            
                ind_plot_unit = c.plot_unit
                if (ind_plot_unit is None): ind_plot_unit = obs_ind.unit
                if self.master:
                    il.AnalysisRelationship(obs_dep,obs_ind,benchmark_results,ind_name,
                                            dep_plot_unit=dep_plot_unit,ind_plot_unit=ind_plot_unit,
                                            regions=self.regions)
                il.AnalysisRelationship(mod_dep,mod_ind,results,ind_name,
                                        dep_plot_unit=dep_plot_unit,ind_plot_unit=ind_plot_unit,
                                        regions=self.regions)

        # close files
        results.close()
        if self.master: benchmark_results.close()
                
    def determinePlotLimits(self):
        """Determine the limits of all plots which are inclusive of all ranges.

        The routine will open all netCDF files in the output path and
        add the maximum and minimum of all variables which are
        designated to be plotted. If legends are desired for a given
        plot, these are rendered here as well. This routine should be
        called before calling any plotting routine.

        """
        # Determine the min/max of variables over all models
        limits = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            try:
                dataset = Dataset(fname)
            except:
                continue
            variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
            for vname in variables:
                var   = dataset.variables[vname]
                pname = vname.split("_")[0]
                if var[...].size <= 1: continue
                if not (space_opts.has_key(pname) or time_opts.has_key(pname)): continue
                if not limits.has_key(pname):
                    limits[pname] = {}
                    limits[pname]["min"]  = +1e20
                    limits[pname]["max"]  = -1e20
                    limits[pname]["unit"] = post.UnitStringToMatplotlib(var.getncattr("units"))
                limits[pname]["min"] = min(limits[pname]["min"],var.getncattr("min"))
                limits[pname]["max"] = max(limits[pname]["max"],var.getncattr("max"))
            dataset.close()
        
        # Second pass to plot legends (FIX: only for master?)
        for pname in limits.keys():

            try:
                opts = space_opts[pname]
            except:
                continue
            
            # Determine plot limits and colormap
            if opts["sym"]:
                vabs =  max(abs(limits[pname]["min"]),abs(limits[pname]["min"]))
                limits[pname]["min"] = -vabs
                limits[pname]["max"] =  vabs
            limits[pname]["cmap"] = opts["cmap"]
            if limits[pname]["cmap"] == "choose": limits[pname]["cmap"] = self.cmap

            # Plot a legend for each key
            if opts["haslegend"]:
                fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
                label  = opts["label"]
                if label == "unit": label = limits[pname]["unit"]
                post.ColorBar(ax,
                              vmin = limits[pname]["min"],
                              vmax = limits[pname]["max"],
                              cmap = limits[pname]["cmap"],
                              ticks = opts["ticks"],
                              ticklabels = opts["ticklabels"],
                              label = label)
                fig.savefig("%s/legend_%s.png" % (self.output_path,pname))
                plt.close()

        # Determine min/max of relationship variables
        for fname in glob.glob("%s/*.nc" % self.output_path):
            try:
                dataset = Dataset(fname)
            except:
                continue
            for g in dataset.groups.keys():
                if "relationship" not in g: continue
                grp = dataset.groups[g]
                if not limits.has_key(g):
                    limits[g] = {}
                    limits[g]["xmin"] = +1e20
                    limits[g]["xmax"] = -1e20
                    limits[g]["ymin"] = +1e20
                    limits[g]["ymax"] = -1e20
                limits[g]["xmin"] = min(limits[g]["xmin"],grp.variables["ind_bnd"][ 0, 0])
                limits[g]["xmax"] = max(limits[g]["xmax"],grp.variables["ind_bnd"][-1,-1])
                limits[g]["ymin"] = min(limits[g]["ymin"],grp.variables["dep_bnd"][ 0, 0])
                limits[g]["ymax"] = max(limits[g]["ymax"],grp.variables["dep_bnd"][-1,-1])
            dataset.close()
            
        self.limits = limits

    def computeOverallScore(self,m):
        """Computes the overall composite score for a given model.

        This routine will try to open the model's netCDF file which
        contains the analysis results, and then loop over variables
        which contribute to the overall score. This number is added to
        the dataset as a new variable of scalar type.

        """
        fname = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        try:
            dataset = Dataset(fname,mode="r+")
        except:
            return
        if "scalars" not in dataset.groups.keys(): return
        grp       = dataset.groups["scalars"]
        variables = [v for v in grp.variables.keys() if "Score" in v and "Overall" not in v]
        for region in self.regions:
            overall_score  = 0.
            sum_of_weights = 0.
            for v in variables:
                if region not in v: continue
                score = v.replace(region,"").strip()
                if not self.weight.has_key(score): continue
                overall_score  += self.weight[score]*grp.variables[v][...]
                sum_of_weights += self.weight[score]
            overall_score /= max(sum_of_weights,1e-12)
            name = "Overall Score %s" % region
            if name in grp.variables.keys():
                grp.variables[name][0] = overall_score
            else:
                Variable(data=overall_score,name=name,unit="1").toNetCDF4(dataset)
        dataset.close()

    def compositePlots(self):
        """Renders plots which display information of all models.

        This routine renders plots which contain information from all
        models. Thus only the master process will run this routine,
        and only after all analysis has finished.

        """
        if not self.master: return
        models = []
        colors = []
        corr   = {}
        std    = {}
        cycle  = {}
        has_cycle = False
        has_std   = False
        for fname in glob.glob("%s/*.nc" % self.output_path):
            dataset = Dataset(fname)
            models.append(dataset.getncattr("name"))
            colors.append(dataset.getncattr("color"))
            for region in self.regions:
                if not std.  has_key(region): std  [region] = []
                if not corr. has_key(region): corr [region] = []
                if not cycle.has_key(region): cycle[region] = []
                key = [v for v in dataset.variables.keys() if ("corr_" in v and region in v)]
                if len(key)>0: corr [region].append(Variable(filename=fname,variable_name=key[0]).data.data)
                key = [v for v in dataset.variables.keys() if ("std_"  in v and region in v)]
                if len(key)>0:
                    has_std = True
                    std  [region].append(Variable(filename=fname,variable_name=key[0]).data.data)
                key = [v for v in dataset.variables.keys() if ("cycle_"  in v and region in v)]
                if len(key)>0:
                    has_cycle = True
                    cycle[region].append(Variable(filename=fname,variable_name=key[0]))
                
        # composite annual cycle plot
        if has_cycle and len(models) > 2:
            self.layout.addFigure("Spatially integrated regional mean",
                                  "compcycle",
                                  "RNAME_compcycle.png",
                                  side   = "CYCLES",
                                  legend = True)

        for region in self.regions:
            if not cycle.has_key(region): continue
            fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
            for name,color,var in zip(models,colors,cycle[region]):
                var.plot(ax,lw=2,color=color,label=name,
                         ticks      = time_opts["cycle"]["ticks"],
                         ticklabels = time_opts["cycle"]["ticklabels"],
                         vmin       = self.limits["cycle"]["min"],
                         vmax       = self.limits["cycle"]["max"])
                ylbl = time_opts["cycle"]["ylabel"]
                if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                ax.set_ylabel(ylbl)
            fig.savefig("%s/%s_compcycle.png" % (self.output_path,region))
            plt.close()

        # plot legends with model colors (sorted with Benchmark data on top)
        def _alphabeticalBenchmarkFirst(key):
            key = key[0].upper()
            if key == "BENCHMARK": return 0
            return key
        tmp = sorted(zip(models,colors),key=_alphabeticalBenchmarkFirst)
        fig,ax = plt.subplots()
        for model,color in tmp:
            ax.plot(0,0,'o',mew=0,ms=8,color=color,label=model)
        handles,labels = ax.get_legend_handles_labels()
        plt.close()
        fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
        ax.legend(handles,labels,loc="upper left",ncol=3,fontsize=10,numpoints=1)
        ax.axis('off')
        fig.savefig("%s/legend_compcycle.png" % self.output_path)
        fig.savefig("%s/legend_spatial_variance.png" % self.output_path)
        plt.close()
        
        # spatial distribution Taylor plot
        if has_std:
            self.layout.addFigure("Temporally integrated period mean",
                                  "spatial_variance",
                                  "RNAME_spatial_variance.png",
                                  side   = "SPATIAL DISTRIBUTION",
                                  legend = True)       
        if "Benchmark" in models: colors.pop(models.index("Benchmark"))
        for region in self.regions:
            if not (std.has_key(region) and corr.has_key(region)): continue
            if len(std[region]) != len(corr[region]): continue
            if len(std[region]) == 0: continue
            fig = plt.figure(figsize=(6.0,6.0))
            post.TaylorDiagram(np.asarray(std[region]),np.asarray(corr[region]),1.0,fig,colors)
            fig.savefig("%s/%s_spatial_variance.png" % (self.output_path,region))
            plt.close()

        
    def modelPlots(self,m):
        """For a given model, create the plots of the analysis results.

        This routine will extract plotting information out of the
        netCDF file which results from the analysis and create
        plots. Note that determinePlotLimits should be called before
        this routine.

        """
        bname     = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname     = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        try:
            dataset   = Dataset(fname)
        except:
            return
        variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
        color     = dataset.getncattr("color")
        for vname in variables:

            # is this a variable we need to plot?
            pname = vname.split("_")[0]
            if dataset.variables[vname][...].size <= 1: continue
            var = Variable(filename=fname,variable_name=vname)
            
            if (var.spatial or (var.ndata is not None)) and not var.temporal:

                # grab plotting options
                if pname not in self.limits.keys(): continue
                opts = space_opts[pname]

                # add to html layout
                self.layout.addFigure(opts["section"],
                                      pname,
                                      opts["pattern"],
                                      side   = opts["sidelbl"],
                                      legend = opts["haslegend"])

                # plot variable
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,
                             region = region,
                             vmin   = self.limits[pname]["min"],
                             vmax   = self.limits[pname]["max"],
                             cmap   = self.limits[pname]["cmap"])
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

                # Jumping through hoops to get the benchmark plotted and in the html output
                if self.master and (pname == "timeint" or pname == "phase"):

                    opts = space_opts[pname]

                    # add to html layout
                    self.layout.addFigure(opts["section"],
                                          "benchmark_%s" % pname,
                                          opts["pattern"].replace("MNAME","Benchmark"),
                                          side   = opts["sidelbl"].replace("MODEL","BENCHMARK"),
                                          legend = False)

                    # plot variable
                    obs = Variable(filename=bname,variable_name=vname)
                    for region in self.regions:
                        fig = plt.figure(figsize=(6.8,2.8))
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        obs.plot(ax,
                                 region = region,
                                 vmin   = self.limits[pname]["min"],
                                 vmax   = self.limits[pname]["max"],
                                 cmap   = self.limits[pname]["cmap"])
                        fig.savefig("%s/Benchmark_%s_%s.png" % (self.output_path,region,pname))
                        plt.close()
                    
            if not (var.spatial or (var.ndata is not None)) and var.temporal:
                
                # grab the benchmark dataset to plot along with
                obs = Variable(filename=bname,variable_name=vname).convert(var.unit)
                
                # grab plotting options
                opts = time_opts[pname]

                # add to html layout
                self.layout.addFigure(opts["section"],
                                      pname,
                                      opts["pattern"],
                                      side   = opts["sidelbl"],
                                      legend = opts["haslegend"])

                # plot variable
                for region in self.regions:
                    if region not in vname: continue
                    fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                    obs.plot(ax,lw=2,color='k',alpha=0.5)
                    var.plot(ax,lw=2,color=color,label=m.name,
                             ticks     =opts["ticks"],
                             ticklabels=opts["ticklabels"])
                    ax.set_ylim(self.limits[pname]["min"],
                                self.limits[pname]["max"])
                    ylbl = opts["ylabel"]
                    if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                    ax.set_ylabel(ylbl)
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

        datasets = [dataset]
        names    = [m.name]
        if self.master:
            datasets.append(Dataset(bname))
            names.append("Benchmark")
            
        for data,name in zip(datasets,names):
            groups = [g for g in data.groups.keys() if "relationship" in g]
            if name == "Benchmark":
                dep_name = self.longname
            else:
                dep_name = self.longname.split("/")[0] + "/" + name

            for g in groups:
                if name == "Benchmark":
                    ind_name = g.replace("relationship_","").split("_")[-1]
                else:
                    ind_name = g.replace("relationship_","").split("_")[-1] + "/" + name

                grp       = data.groups[g]
                ind       = grp.variables["ind"][...]
                dep       = grp.variables["dep"][...]
                ind_bnd   = grp.variables["ind_bnd"][...]
                dep_bnd   = grp.variables["dep_bnd"][...]
                histogram = grp.variables["histogram"][...].T
                ind_edges = np.zeros(ind_bnd.shape[0]+1); ind_edges[:-1] = ind_bnd[:,0]; ind_edges[-1] = ind_bnd[-1,1]
                dep_edges = np.zeros(dep_bnd.shape[0]+1); dep_edges[:-1] = dep_bnd[:,0]; dep_edges[-1] = dep_bnd[-1,1]
                fig,ax    = plt.subplots(figsize=(6,5.25),tight_layout=True)
                cmap      = 'plasma'
                if not plt.cm.cmap_d.has_key(cmap): cmap = 'summer'
                pc        = ax.pcolormesh(ind_edges,dep_edges,histogram,
                                          norm=LogNorm(),
                                          cmap=cmap)
                x,y = grp.variables["ind_mean"],grp.variables["dep_mean"]
                ax.plot(x,y,'-w',lw=3,alpha=0.75)
                #ax.fill_between(grp.variables["ind_mean"][...],
                #                grp.variables["dep_mean"][...]-grp.variables["dep_std"][...],
                #                grp.variables["dep_mean"][...]+grp.variables["dep_std"][...],
                #                color='k',alpha=0.25,lw=0)
                
                div       = make_axes_locatable(ax)
                fig.colorbar(pc,cax=div.append_axes("right",size="5%",pad=0.05),
                             orientation="vertical",
                             label="Fraction of total datasites")
                ax.set_xlabel("%s,  %s" % (ind_name,post.UnitStringToMatplotlib(x.getncattr("unit"))))
                ax.set_ylabel("%s,  %s" % (dep_name,post.UnitStringToMatplotlib(y.getncattr("unit"))))
                ax.set_xlim(self.limits[g]["xmin"],self.limits[g]["xmax"])
                ax.set_ylim(self.limits[g]["ymin"],self.limits[g]["ymax"])
                short_name = g.replace("relationship_","rel_")
                fig.savefig("%s/%s_%s.png" % (self.output_path,name,short_name))
                plt.close()
                if "global_" in short_name:
                    short_name = short_name.replace("global_","")
                    self.layout.addFigure("Period Mean Relationships",
                                          short_name,
                                          "MNAME_RNAME_%s.png" % (short_name),
                                          legend = False,
                                          benchmark = True)

        # Code to add a Whittaker diagram (FIX: this is messy, need to rethink data access, redundant computation)
        Ts = []; T_plot_units = []; T_labels = []
        Ps = []; P_plot_units = []; P_labels = []
        if self.relationships is not None:
            for c in self.relationships:
                
                if "Temperature" in c.longname:
                    obs,mod = c.stageData(m)
                    Ts.append(mod)
                    T_plot_units.append(c.plot_unit)
                    T_labels.append(c.longname.split("/")[0] + "/" + m.name)
                    if self.master:
                        Ts.append(obs)
                        T_plot_units.append(c.plot_unit)
                        T_labels.append(c.longname)

                if "Precipitation" in c.longname:
                    obs,mod = c.stageData(m)
                    Ps.append(mod)
                    P_plot_units.append(c.plot_unit)
                    P_labels.append(c.longname.split("/")[0] + "/" + m.name)
                    if self.master:
                        Ps.append(obs)
                        P_plot_units.append(c.plot_unit)
                        P_labels.append(c.longname)

        if len(Ts) == 0 or len(Ps) == 0: return
        
        filenames = [fname]
        Z_labels  = [self.longname.split("/")[0] + "/" + m.name]
        if self.master:
            filenames.append(bname)
            Z_labels.append(self.longname)

        for region in self.regions:
            
            T_key = [key for key in self.limits.keys() if ("Temperature" in key and region in key)][0]
            T_min = self.limits[T_key]["xmin"]
            T_max = self.limits[T_key]["xmax"]
            P_key = [key for key in self.limits.keys() if ("Precipitation" in key and region in key)][0]
            P_min = self.limits[P_key]["xmin"]
            P_max = self.limits[P_key]["xmax"]
            V_min = self.limits[P_key]["ymin"]
            V_max = self.limits[P_key]["ymax"]
        
            if len(Ts) > 0 and len(Ps) > 0:
                for filename,data,name,T,T_plot_unit,T_label,P,P_plot_unit,P_label,Z_label in zip(filenames,datasets,names,
                                                                                                  Ts,T_plot_units,T_labels,
                                                                                                  Ps,P_plot_units,P_labels,
                                                                                                  Z_labels):
                    Z = [k for k in data.variables.keys() if "timeint_of" in k]

                    post.WhittakerDiagram(T,
                                          P,
                                          Variable(filename=filename,variable_name=Z[0]),
                                          region      = region,
                                          X_plot_unit =    T_plot_unit,
                                          Y_plot_unit =    P_plot_unit,
                                          Z_plot_unit = self.plot_unit,
                                          X_label     =    T_label,
                                          Y_label     =    P_label,
                                          Z_label     = self.longname,
                                          X_min = T_min, X_max = T_max,
                                          Y_min = P_min, Y_max = P_max,
                                          Z_min = V_min, Z_max = V_max,
                                          filename    = "%s/%s_%s_whittaker.png" % (self.output_path,name,region))
                    
                self.layout.addFigure("Period Mean Relationships",
                                      "whittaker",
                                      "MNAME_RNAME_whittaker.png",
                                      legend    = False,
                                      benchmark = True)
                
    def generateHtml(self):
        """Generate the HTML for the results of this confrontation.

        This routine opens all netCDF files and builds a table of
        metrics. Then it passes the results to the HTML generator and
        saves the result in the output directory. This only occurs on
        the confrontation flagged as master.

        """
        # only the master processor needs to do this
        if not self.master: return

        # build the metric dictionary
        metrics = {}
        
        for fname in glob.glob("%s/*.nc" % self.output_path):
            try:
                dataset = Dataset(fname)
            except:
                continue

            # if the dataset opens, we need to add the model (table row)
            mname = dataset.getncattr("name")
            metrics[mname] = {}

            # each model will need to have all regions
            for region in self.regions: metrics[mname][region] = {}
            
            # columns in the table will be in the scalars group
            if not dataset.groups.has_key("scalars"): continue

            # we add scalars to the model/region based on the region
            # name being in the variable name. If no region is found,
            # we assume it is the global region.
            grp = dataset.groups["scalars"]
            for vname in grp.variables.keys():
                found = False
                for region in self.regions:
                    if region in vname: 
                        found = True
                        var   = grp.variables[vname]
                        name  = vname.replace(region,"")
                        metrics[mname][region][name] = Variable(name = name,
                                                                unit = var.units,
                                                                data = var[...])
                if not found:
                    var = grp.variables[vname]
                    metrics[mname]["global"][vname] = Variable(name = vname,
                                                               unit = var.units,
                                                               data = var[...])
                    

        """
        print "\n\n",self.longname
        import sys
        def dump(obj, nested_level=0, output=sys.stdout):
            spacing = '   '
            if type(obj) == dict:
                print >> output, '%s{' % ((nested_level) * spacing)
                for k, v in obj.items():
                    if hasattr(v, '__iter__'):
                        print >> output, '%s%s:' % ((nested_level + 1) * spacing, k)
                        dump(v, nested_level + 1, output)
                    else:
                        print >> output, '%s%s: %g' % ((nested_level + 1) * spacing, k, v.data)
                print >> output, '%s}' % (nested_level * spacing)
            elif type(obj) == list:
                print >> output, '%s[' % ((nested_level) * spacing)
                for v in obj:
                    if hasattr(v, '__iter__'):
                        dump(v, nested_level + 1, output)
                    else:
                        print >> output, '%s%s' % ((nested_level + 1) * spacing, v.name)
                print >> output, '%s]' % ((nested_level) * spacing)
            else:
                print >> output, '%s%s' % (nested_level * spacing, obj.name)
        dump(metrics)
        """
        
        # write the HTML page
        f = file("%s/%s.html" % (self.output_path,self.name),"w")
        self.layout.setMetrics(metrics)
        f.write(str(self.layout))
        f.close()

