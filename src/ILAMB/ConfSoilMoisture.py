from . import ilamblib as il
from .Variable import *
from .Regions import Regions
from .constants import space_opts,time_opts,mid_months,bnd_months
import os,glob,re
from netCDF4 import Dataset
from . import Post as post
import pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpi4py import MPI
from sympy import sympify
import cftime as cf
from .Confrontation import getVariableList
from .Confrontation import Confrontation


import logging
logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)


class ConfSoilMoisture(Confrontation):

    def __init__(self,**keywords):
        # Calls the regular constructor
        super(ConfSoilMoisture,self).__init__(**keywords)

        # Get/modify depths
        with Dataset(self.source) as dset:
            v = dset.variables[self.variable]
            depth_name = [d for d in v.dimensions if d in ["layer","depth"]]
            if len(depth_name) == 0:
                # if there is no depth dimension, we assume the data is
                # top 10cm
                self.depths = np.asarray([[0., .1]],dtype=float)
                self.depths_units = 'm'
            else:
                # if there are depths, then make sure that the depths
                # at which we will compute are in the range of depths
                # of the data
                depth_name = depth_name[0]

                if 'bounds' in dset.variables[depth_name].ncattrs():
                    data = dset.variables[dset.variables[depth_name \
                                                         ].bounds][...]
                    self.depths = data
                    self.depths_units = dset.variables[dset.variables[depth_name \
                                                                      ].bounds].units
                else:
                    data = dset.variables[depth_name][...]

                    self.depths = np.asarray(self.keywords.get("depths_bnds",
                                                               [[0., .1],
                                                                [.1, .3],
                                                                [.3, .5],
                                                                [.5, 1.]]),
                                             dtype = float)
                    self.depths = self.depths[(self.depths[:,1]>=data.min()
                        )*(self.depths[:,0]<=data.max()), :]
                    self.depths_units = dset.variables[depth_name].units

    def stageData(self,m):
        """ Extract Model data with interpolation to the confrontation
            depth."""
        mem_slab = self.keywords.get("mem_slab",100000.) # Mb

        # peak at the reference dataset without reading much into memory
        info = ""
        unit = ""
        with Dataset(self.source) as dset:
            var = dset.variables[self.variable]
            obs_t,obs_tb,obs_cb,obs_b,obs_e,cal = il.GetTime(var)
            obs_nt = obs_t.size
            obs_mem = var.size*8e-6
            unit = var.units
            climatology = False if obs_cb is None else True
            if climatology:
                info += "[climatology]"
                obs_cb = (obs_cb-1850)*365.
                t0 = obs_cb[0]; tf = obs_cb[1]
            else:
                t0 = obs_tb[0,0]; tf = obs_tb[-1,1]
            info += " contents span years %.1f to %.1f, est memory %d [Mb]" % (t0/365.+1850,tf/365.+1850,obs_mem)
        logger.info("[%s][%s]%s" % (self.name,self.variable,info))

        # to peak at the model, we need any variable that could be
        # part of the expression to look at the time
        info = ""
        possible = [self.variable,] + self.alternate_vars
        if self.derived is not None: possible += [str(s) for s in sympify(self.derived).free_symbols]
        vname = [v for v in possible if v in m.variables.keys()]
        if len(vname) == 0:
            logger.debug("[%s] Could not find [%s] in the model results" % (self.name,",".join(possible)))
            raise il.VarNotInModel()
        vname = vname[0]

        # peak at the model dataset without reading much into memory
        mod_nt  =  0
        mod_mem =  0.
        mod_t0  =  2147483647
        mod_tf  = -2147483648
        for fname in m.variables[vname]:
            with Dataset(fname) as dset:
                var = dset.variables[vname]
                mod_t,mod_tb,mod_cb,mod_b,mod_e,cal = il.GetTime(var,t0=t0-m.shift,tf=tf-m.shift)
                if mod_t is None:
                    info += "\n      %s does not overlap the reference" % (fname)
                    continue
                mod_t += m.shift
                mod_tb += m.shift
                ind = np.where((mod_tb[:,0] >= t0)*(mod_tb[:,1] <= tf))[0]
                if ind.size == 0:
                    info += "\n      %s does not overlap the reference" % (fname)
                    continue
                mod_t  = mod_t [ind]
                mod_tb = mod_tb[ind]
                mod_t0 = min(mod_t0,mod_tb[ 0,0])
                mod_tf = max(mod_tf,mod_tb[-1,1])
                nt = mod_t.size
                mod_nt += nt
                mem = (var.size/var.shape[0]*nt)*8e-6
                mod_mem += mem
                info += "\n      %s spans years %.1f to %.1f, est memory in time bounds %d [Mb]" % (fname,mod_t.min()/365.+1850,mod_t.max()/365.+1850,mem)
        info += "\n      total est memory = %d [Mb]" % mod_mem
        logger.info("[%s][%s][%s] reading model data from possibly many files%s" % (self.name,m.name,vname,info))
        if mod_t0 > mod_tf:
            logger.debug("[%s] Could not find [%s] in the model results in the given time frame, tinput = [%.1f,%.1f]" % (self.name,",".join(possible),t0,tf))
            raise il.VarNotInModel()

        # if the reference is a climatology, then build a model climatology in slabs
        info = ""
        if climatology:

            # how many slabs
            ns   = int(np.floor(mod_mem/mem_slab))+1
            ns   = min(max(1,ns),mod_nt)
            logger.info("[%s][%s] building climatology in %d slabs" % (self.name,m.name,ns))

            # across what times?
            slab_t = (mod_tf-mod_t0)*np.linspace(0,1,ns+1)+mod_t0
            slab_t = np.floor(slab_t / 365)*365 + bnd_months[(np.abs(bnd_months[:,np.newaxis] - (slab_t % 365))).argmin(axis=0)]

            # ready to slab
            tb_prev = None
            data    = None
            dnum    = None
            for i in range(ns):
                v = m.extractTimeSeries(self.variable,
                                        alt_vars     = self.alternate_vars,
                                        expression   = self.derived,
                                        initial_time = slab_t[i],
                                        final_time   = slab_t[i+1]).convert(unit)

                # trim does not work properly so we will add a manual check ourselves
                if tb_prev is None:
                    tb_prev = v.time_bnds[...]
                else:
                    if np.allclose(tb_prev[-1],v.time_bnds[0]):
                        v.data = v.data[1:]
                        v.time = v.time[1:]
                        v.time_bnds = v.time_bnds[1:]
                    tb_prev = v.time_bnds[...]
                if v.time.size == 0: continue

                mind = (np.abs(mid_months[:,np.newaxis]-(v.time % 365))).argmin(axis=0)
                if data is None:
                    data = np.ma.zeros((12,)+v.data.shape[1:])
                    dnum = np.ma.zeros(data.shape,dtype=int)
                data[mind,...] += v.data
                dnum[mind,...] += 1
            with np.errstate(over='ignore',under='ignore'):
                data = data / dnum.clip(1)

            # return variables
            obs = Variable(filename       = self.source,
                           variable_name  = self.variable,
                           alternate_vars = self.alternate_vars)
            mod = Variable(name  = obs.name,
                           unit  = unit,
                           data  = data,
                           time  = obs.time,
                           lat   = v.lat,
                           lon   = v.lon,
                           depth = v.depth,
                           time_bnds  = obs.time_bnds,
                           lat_bnds   = v.lat_bnds,
                           lon_bnds   = v.lon_bnds,
                           depth_bnds = v.depth_bnds)
            yield obs,mod

        # if obs is historical, then we yield slabs of both
        else:
            obs_mem *= (mod_tf-mod_t0)/(tf-t0)
            mod_t0 = max(mod_t0,t0)
            mod_tf = min(mod_tf,tf)
            ns   = int(np.floor(max(obs_mem,mod_mem)/mem_slab))+1
            ns   = min(min(max(1,ns),mod_nt),obs_nt)
            logger.info("[%s][%s] staging data in %d slabs" % (self.name,m.name,ns))
            
            # across what times?
            slab_t = (mod_tf-mod_t0)*np.linspace(0,1,ns+1)+mod_t0
            slab_t = np.floor(slab_t / 365)*365 + bnd_months[(np.abs(bnd_months[:,np.newaxis] - (slab_t % 365))).argmin(axis=0)]
            
            obs_tb = None; mod_tb = None
            for i in range(ns):
                # get reference variable
                obs = Variable(filename       = self.source,
                               variable_name  = self.variable,
                               alternate_vars = self.alternate_vars,
                               t0             = slab_t[i],
                               tf             = slab_t[i+1]).trim(t=[slab_t[i],slab_t[i+1]])
                if obs_tb is None:
                    obs_tb = obs.time_bnds[...]
                else:
                    if np.allclose(obs_tb[-1],obs.time_bnds[0]):
                        obs.data = obs.data[1:]
                        obs.time = obs.time[1:]
                        obs.time_bnds = obs.time_bnds[1:]
                    assert np.allclose(obs.time_bnds[0,0],obs_tb[-1,1])
                    obs_tb = obs.time_bnds[...]

                # get model variable
                mod = m.extractTimeSeries(self.variable,
                                          alt_vars     = self.alternate_vars,
                                          expression   = self.derived,
                                          initial_time = slab_t[i],
                                          final_time   = slab_t[i+1]).trim(t=[slab_t[i],slab_t[i+1]]).convert(obs.unit)
                if mod_tb is None:
                    mod_tb = mod.time_bnds
                else:
                    if np.allclose(mod_tb[-1],mod.time_bnds[0]):
                        mod.data = mod.data[1:]
                        mod.time = mod.time[1:]
                        mod.time_bnds = mod.time_bnds[1:]
                    assert np.allclose(mod.time_bnds[0,0],mod_tb[-1,1])
                    mod_tb = mod.time_bnds[...]
                assert obs.time.size == mod.time.size
                
                yield obs,mod


    def confront(self,m):

        def _addDepth(v):
            v.depth = np.asarray([.05])
            v.depth_bnds = np.asarray([[0.,.1]])
            shp = list(v.data.shape)
            shp.insert(1,1)
            v.data.shape = shp
            v.layered = True
            return v

        mod_file = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        obs_file = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name,      ))
        with il.FileContextManager(self.master,mod_file,obs_file) as fcm:

            # Encode some names and colors
            fcm.mod_dset.setncatts({"name" :m.name,
                                    "color":m.color,
                                    "complete":0})
            if self.master:
                fcm.obs_dset.setncatts({"name" :"Benchmark",
                                        "color":np.asarray([0.5,0.5,0.5]),
                                        "complete":0})

            # Get the depth-integrated observation and model data for each slab.
            obs_timeint = {}; mod_timeint = {}
            for dind in range(self.depths.shape[0]):
                obs_timeint[dind] = []
                mod_timeint[dind] = []
            for obs,mod in self.stageData(m):
                # if the data has no depth, we assume it is surface
                if not obs.layered: obs = _addDepth(obs)
                if not mod.layered: mod = _addDepth(mod)

                # time bounds for this slab
                tb = obs.time_bnds[[0,-1],[0,1]].reshape((1,2))
                t  = np.asarray([tb.mean()])

                #
                for dind, z0 in enumerate(self.depths[:, 0]):
                    zf = self.depths[dind, 1]                    
                    z = obs.integrateInDepth(z0 = z0, zf = zf, mean = True).integrateInTime(mean = True)

                    #YW
                    print('Staging data ... %.2f-%.2f' % (z0, zf))
                    
                    obs_timeint[dind].append(Variable(name = "sm%.2f-%.2f" % (z0, zf),
                                                      unit = z.unit,
                                                      data = z.data.reshape((1,) +z.data.shape),
                                                      time = t, time_bnds = tb,
                                                      lat = z.lat, lat_bnds = z.lat_bnds,
                                                      lon = z.lon, lon_bnds = z.lon_bnds))
                    z = mod.integrateInDepth(z0 = z0, zf = zf, mean = True).integrateInTime(mean = True)
                    mod_timeint[dind].append(Variable(name = "sm%.2f-%.2f" % (z0, zf),
                                                      unit = z.unit,
                                                      data = z.data.reshape((1,)+z.data.shape),
                                                      time = t,     time_bnds = tb,
                                                      lat  = z.lat, lat_bnds  = z.lat_bnds,
                                                      lon  = z.lon, lon_bnds  = z.lon_bnds))

            # Read in some options and run the mean state analysis
            mass_weighting = self.keywords.get("mass_weighting",False)
            skip_rmse      = self.keywords.get("skip_rmse"     ,False)
            skip_iav       = self.keywords.get("skip_iav"      ,True )
            skip_cycle     = self.keywords.get("skip_cycle"    ,False)
            rmse_score_basis = self.keywords.get("rmse_score_basis","cycle")

            for dind in range(self.depths.shape[0]):
                obs_tmp = il.CombineVariables(obs_timeint[dind])
                mod_tmp = il.CombineVariables(mod_timeint[dind])
                print(obs_tmp.name)
                print(mod_tmp.name)
                obs_tmp.name = obs_tmp.name.split("_")[0]
                mod_tmp.name = mod_tmp.name.split("_")[0]

                if obs_tmp.spatial:
                    il.AnalysisMeanStateSpace(obs_tmp,mod_tmp,dataset   = fcm.mod_dset,
                                           regions           = self.regions,
                                           benchmark_dataset = fcm.obs_dset,
                                           table_unit        = self.table_unit,
                                           plot_unit         = self.plot_unit,
                                           space_mean        = self.space_mean,
                                           skip_rmse         = skip_rmse,
                                           skip_iav          = skip_iav,
                                           skip_cycle        = skip_cycle,
                                           mass_weighting    = mass_weighting,
                                           rmse_score_basis  = rmse_score_basis)
                else:
                    il.AnalysisMeanStateSites(obs_tmp,mod_tmp,dataset   = fcm.mod_dset,
                                           regions           = self.regions,
                                           benchmark_dataset = fcm.obs_dset,
                                           table_unit        = self.table_unit,
                                           plot_unit         = self.plot_unit,
                                           space_mean        = self.space_mean,
                                           skip_rmse         = skip_rmse,
                                           skip_iav          = skip_iav,
                                           skip_cycle        = skip_cycle,
                                           mass_weighting    = mass_weighting)
            fcm.mod_dset.setncattr("complete",1)
            if self.master: fcm.obs_dset.setncattr("complete",1)
        logger.info("[%s][%s] Success" % (self.longname,m.name))


    def computeOverallScore(self,m):
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
            scores     = {}
            variables = [v for v in scalars.variables.keys() if "Score" in v and "Overall" not in v]
            for region in self.regions:
                overall_score  = 0.
                sum_of_weights = 0.
                for v in variables:
                    if region not in v: continue
                    score = v.replace(region,"").strip()
                    weight = 1.
                    if score in self.weight: weight = self.weight[score]
                    overall_score  += weight*scalars.variables[v][...]
                    sum_of_weights += weight
                overall_score /= max(sum_of_weights,1e-12)
                scores["Overall Score %s" % region] = overall_score
            return scores

        fname = os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name))
        if not os.path.isfile(fname): return
        with Dataset(fname,mode="r+") as dataset:
            datasets = [dataset.groups[grp] for grp in dataset.groups if "scalars" not in grp]
            groups   = [grp                 for grp in dataset.groups if "scalars" not in grp]
            datasets.append(dataset)
            groups  .append(None)
            for dset,grp in zip(datasets,groups):
                if "scalars" in dset.groups:
                    scalars = dset.groups["scalars"]
                    score = _computeOverallScore(scalars)
                    for key in score.keys():
                        if key in scalars.variables:
                            scalars.variables[key][0] = score[key]
                        else:
                            Variable(data=score[key],name=key,unit="1").toNetCDF4(dataset,group=grp)


    def compositePlots(self):
        """Renders plots which display information of all models.

        This routine renders plots which contain information from all
        models. Thus only the master process will run this routine,
        and only after all analysis has finished.

        """
        if not self.master: return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        models = []
        colors = []
        corr   = {}
        std    = {}
        cycle  = {}
        has_cycle = False
        has_std   = False
        for fname in glob.glob(os.path.join(self.output_path,"*.nc")):
            dataset = Dataset(fname)
            if "MeanState" not in dataset.groups: continue
            dset    = dataset.groups["MeanState"]
            models.append(dataset.getncattr("name"))
            colors.append(dataset.getncattr("color"))
            for region in self.regions:
                if region not in cycle: cycle[region] = {}
                if region not in std: std[region] = {}
                if region not in corr: corr[region] = {}

                for dind, z0 in enumerate(self.depths[:,0]):
                    zf = self.depths[dind,1]
                    zstr = '%.2f-%.2f' % (z0, zf)

                    if zstr not in cycle[region]: cycle[region][zstr] = []

                    key = [v for v in dset.variables.keys() if ("cycle_"  in v and zstr in v and region in v)]
                    if len(key)>0:
                        has_cycle = True
                        cycle[region][zstr].append(Variable(filename=fname,groupname="MeanState",
                                                             variable_name=key[0]))

                    if zstr not in std[region]: std[region][zstr] = []
                    if zstr not in corr[region]: corr[region][zstr] = []

                    key = []
                    if "scalars" in dset.groups:
                        key = [v for v in dset.groups["scalars"].variables.keys() \
                               if ("Spatial Distribution Score" in v and zstr in v and region in v)]
                        if len(key) > 0:
                            has_std = True
                            sds     = dset.groups["scalars"].variables[key[0]]
                            corr[region][zstr].append(sds.getncattr("R"  ))
                            std [region][zstr].append(sds.getncattr("std"))

        # composite annual cycle plot
        if has_cycle and len(models) > 2:
            page.addFigure("Spatially integrated regional mean",
                           "compcycle",
                           "RNAME_compcycle.png",
                           side   = "ANNUAL CYCLE",
                           legend = False)

        for region in self.regions:
            if region not in cycle: continue
            fig, axes = plt.subplots(self.depths.shape[0], 1,
                                     figsize=(6.8,2.8 * self.depths.shape[0]),
                                     tight_layout=True)
            for dind, z0 in enumerate(self.depths[:,0]):
                zf = self.depths[dind, 1]
                zstr = '%.2f-%.2f' % (z0, zf)

                ax = axes[dind]
                for name,color,var in zip(models,colors,cycle[region][zstr]):
                    dy = 0.05*(self.limits["cycle"][region]["max"]-self.limits["cycle"][region]["min"])
                    var.plot(ax, lw=2, color=color, label=name,
                             ticks      = time_opts["cycle"]["ticks"],
                             ticklabels = time_opts["cycle"]["ticklabels"],
                             vmin       = self.limits["cycle"][region]["min"]-dy,
                             vmax       = self.limits["cycle"][region]["max"]+dy)
                    ylbl = time_opts["cycle"]["ylabel"]
                    if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                    ylbl = ylbl + ' ' + zstr + self.depths_units
                    ax.set_ylabel(ylbl)
            fig.savefig(os.path.join(self.output_path,"%s_compcycle.png" % (region)))
            plt.close()

        # plot legends with model colors (sorted with Benchmark data on top)
        page.addFigure("Spatially integrated regional mean",
                       "legend_compcycle",
                       "legend_compcycle.png",
                       side   = "MODEL COLORS",
                       legend = False)
        def _alphabeticalBenchmarkFirst(key):
            key = key[0].lower()
            if key == "BENCHMARK": return "A"
            return key
        tmp = sorted(zip(models,colors),key=_alphabeticalBenchmarkFirst)
        fig,ax = plt.subplots()
        for model,color in tmp:
            ax.plot(0,0,'o',mew=0,ms=8,color=color,label=model)
        handles,labels = ax.get_legend_handles_labels()
        plt.close()

        ncol = np.ceil(float(len(models))/11.).astype(int)
        if ncol > 0:
            fig,ax = plt.subplots(figsize=(3.*ncol,2.8),tight_layout=True)
            ax.legend(handles,labels,loc="upper right",ncol=ncol,fontsize=10,numpoints=1)
            ax.axis(False)
            fig.savefig(os.path.join(self.output_path,"legend_compcycle.png"))
            fig.savefig(os.path.join(self.output_path,"legend_spatial_variance.png"))
            fig.savefig(os.path.join(self.output_path,"legend_temporal_variance.png"))
            plt.close()

        # spatial distribution Taylor plot
        if has_std:
            for dind, z0 in enumerate(self.depths[:,0]):
                zf = self.depths[dind, 1]
                zstr = '%.2f-%.2f' % (z0, zf)
                page.addFigure("Temporally integrated period mean",
                               "spatial_variance",
                               "RNAME_spatial_variance_" + zstr + ".png",
                               side   = "SPATIAL TAYLOR DIAGRAM",
                               legend = False)
            page.addFigure("Temporally integrated period mean",
                           "legend_spatial_variance",
                           "legend_spatial_variance.png",
                           side   = "MODEL COLORS",
                           legend = False)
        if "Benchmark" in models: colors.pop(models.index("Benchmark"))
        for region in self.regions:
            if not (region in std and region in corr): continue

            for dind, z0 in enumerate(self.depths[:,0]):
                zf = self.depths[dind, 1]
                zstr = '%.2f-%.2f' % (z0, zf)

                if not (zstr in std[region] and zstr in corr[region]): continue
                if len(std[region][zstr]) != len(corr[region][zstr]): continue
                if len(std[region][zstr]) == 0: continue

                fig = plt.figure(figsize=(6.0,6.0))
                post.TaylorDiagram(np.asarray(std[region][zstr]),
                                   np.asarray(corr[region][zstr]),
                                   1.0,fig,colors)
                fig.savefig(os.path.join(self.output_path,
                                         "%s_spatial_variance_%s.png" % (region, zstr)))
                plt.close()

    def modelPlots(self,m):
        """For a given model, create the plots of the analysis results.

        This routine will extract plotting information out of the
        netCDF file which results from the analysis and create
        plots. Note that determinePlotLimits should be called before
        this routine.

        """
        bname     = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name       ))
        fname     = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        with Dataset(fname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")
            for vname in variables:
                # The other depths will be handled in plotting
                zstr_0 = '%.2f-%.2f' % (self.depths[0,0], self.depths[0,1])
                if not zstr_0 in vname: continue
                
                # is this a variable we need to plot?
                pname = vname.split("_")[0]
                if group.variables[vname][...].size <= 1: continue
                var = Variable(filename=fname,groupname="MeanState",variable_name=vname)

                if (var.spatial or (var.ndata is not None)) and not var.temporal:

                    # grab plotting options
                    if pname not in self.limits.keys(): continue
                    if pname not in space_opts: continue
                    opts = space_opts[pname]

                    # add to html layout
                    page.addFigure(opts["section"],
                                   pname,
                                   opts["pattern"],
                                   side   = opts["sidelbl"],
                                   legend = opts["haslegend"])

                    # plot variable
                    for region in self.regions:
                        fig, axes = plt.subplots(self.depths.shape[0], 1,
                                                 figsize = (6.5, 2.8*self.depths.shape[0]))

                        for dind, z0 in enumerate(self.depths[:,0]):
                            zf = self.depths[dind,1]
                            zstr = '%.2f-%.2f' % (z0, zf)
                            ax = axes.flat[dind]
                            var2 = Variable(filename=fname, groupname = "MeanState",
                                            variable_name=vname.replace(zstr_0, zstr))
                            var2.plot(ax, region = region,
                                      vmin   = self.limits[pname]["min"],
                                      vmax   = self.limits[pname]["max"],
                                      cmap   = self.limits[pname]["cmap"])
                        fig.savefig(os.path.join(self.output_path,"%s_%s_%s.png" % (m.name,region,pname)))
                        plt.close()

                    # Jumping through hoops to get the benchmark plotted and in the html output
                    if self.master and (pname == "timeint" or pname == "phase" or pname == "iav"):
                        opts = space_opts[pname]

                        # add to html layout
                        page.addFigure(opts["section"],
                                       "benchmark_%s" % pname,
                                       opts["pattern"].replace("MNAME","Benchmark"),
                                       side   = opts["sidelbl"].replace("MODEL","BENCHMARK"),
                                       legend = True)

                        # plot variable
                        for region in self.regions:
                            fig, axes = plt.subplots(self.depths.shape[0], 1,
                                                     figsize = (6.5, 2.8*self.depths.shape[0]))
                            for dind, z0 in enumerate(self.depths[:,0]):
                                zf = self.depths[dind,1]
                                zstr = '%.2f-%.2f' % (z0, zf)
                                ax = axes.flat[dind]
                                obs = Variable(filename=bname,groupname="MeanState",
                                               variable_name=vname.replace(zstr_0, zstr))
                                obs.plot(ax, region = region,
                                         vmin   = self.limits[pname]["min"],
                                         vmax   = self.limits[pname]["max"],
                                         cmap   = self.limits[pname]["cmap"])
                            fig.savefig(os.path.join(self.output_path,"Benchmark_%s_%s.png" % (region,pname)))
                            plt.close()

                    if not (var.spatial or (var.ndata is not None)) and var.temporal:
                        # grab the benchmark dataset to plot along with
                        try:
                            obs = Variable(filename=bname,groupname="MeanState",
                                           variable_name=vname).convert(var.unit)
                        except:
                            continue

                        # grab plotting options
                        if pname not in time_opts: continue
                        opts = time_opts[pname]

                        # add to html layout
                        page.addFigure(opts["section"],
                                       pname,
                                       opts["pattern"],
                                       side   = opts["sidelbl"],
                                       legend = opts["haslegend"])

                        # plot variable
                        for region in self.regions:
                            if region not in vname: continue
                            fig, axes = plt.subplots(self.depths.shape[0], 1,
                                                     figsize = (6.5, 2.8*self.depths.shape[0]))
                            for dind, z0 in enumerate(self.depths[:,0]):
                                zf = self.depths[dind,1]
                                zstr = '%.2f-%.2f' % (z0, zf)
                                ax = axes.flat[dind]

                                var2 = Variable(filename=fname, groupname = "MeanState",
                                                variable_name=vname.replace(zstr_0, zstr))
                                obs = Variable(filename=bname,groupname="MeanState",
                                               variable_name=vname.replace(zstr_0, zstr)).convert(var2.unit)
                                obs.plot(ax, lw = 2, color = 'k', alpha = 0.5)
                                var2.plot(ax, lw = 2, color = color, label = m.name,
                                          ticks     =opts["ticks"],
                                          ticklabels=opts["ticklabels"])
                                dy = 0.05*(self.limits[pname][region]["max"]-self.limits[pname][region]["min"])
                                ax.set_ylim(self.limits[pname][region]["min"]-dy,
                                            self.limits[pname][region]["max"]+dy)
                                ylbl = opts["ylabel"]
                                if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                                ax.set_ylabel(ylbl)
                            fig.savefig(os.path.join(self.output_path,"%s_%s_%s.png" % (m.name,region,pname)))
                            plt.close()

        logger.info("[%s][%s] Success" % (self.longname,m.name))
