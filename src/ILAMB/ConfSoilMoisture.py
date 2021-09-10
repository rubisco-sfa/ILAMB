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
import numpy as np
import time # DEBUG


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
                depth_bnd_name = [d for d in dset.variables.keys() \
                                  if depth_name in d and ("bound" in d or "bnd" in d)]

                if len(depth_bnd_name) > 0:
                    depth_bnd_name = depth_bnd_name[0]
                    data = dset.variables[depth_bnd_name][...].data
                    self.depths = data
                    self.depths_units = dset.variables[depth_bnd_name].units
                else:
                    data = dset.variables[depth_name][...]

                    self.depths = np.asarray(self.keywords.get("depths_bnds",
                                                               [[0., .1]]),
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

            print('stage observation ' + self.variable) # DEBUG
            tstart = time.time() #DEBUG
            obs_t,obs_tb,obs_cb,obs_b,obs_e,cal = il.GetTime(var)
            tend = time.time() # DEBUG
            print( "il.GetTime took " + str((tend - tstart)/60) + " minutes." ) # DEBUG

            obs_nt = obs_t.size
            obs_mem = var.size*8e-6
            unit = var.units
            climatology = False if obs_cb is None else True
            if climatology:
                info += "[climatology]"
                obs_cb = (obs_cb-1850)*365.
                obs_t0 = obs_cb[0]; obs_tf = obs_cb[1]
            else:
                obs_t0 = obs_tb[0,0]; obs_tf = obs_tb[-1,1]

            obs_dname = [name for name in dset.variables.keys() \
                         if name.lower() in ["depth_bnds", "depth_bounds"]]
            if len(obs_dname) == 0:
                # if there is no depth, assume the data is surface
                obs_z0 = 0; obs_zf = 0.1; obs_z_bnd = np.array([[0, 0.1]]); obs_nd = 0
                obs_dname = None
            else:
                obs_dname = obs_dname[0]
                obs_z0 = np.min(dset.variables[obs_dname])
                obs_zf = np.max(dset.variables[obs_dname])
                obs_z_bnd = dset.variables[obs_dname][...]
                obs_nd = dset.variables[obs_dname].shape[0]

            info += " contents span years %.1f to %.1f and depths %.1f to %.1f, est memory %d [Mb]" % (obs_t0/365.+1850,obs_tf/365.+1850,obs_z0,obs_zf,obs_mem)
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
        mod_t0  = 2147483647
        mod_tf  = -2147483648
        mod_z0  =  2147483647
        mod_zf  = -2147483648
        mod_nd = 999
        for fname in m.variables[vname]:

            print('stage model ' + vname) # DEBUG

            with Dataset(fname) as dset:
                var = dset.variables[vname]

                tstart = time.time() # DEBUG
                mod_t,mod_tb,mod_cb,mod_b,mod_e,cal = il.GetTime(var,t0=obs_t0-m.shift,
                                                                 tf=obs_tf-m.shift)
                tend = time.time() # DEBUG
                print( "il.GetTime took " + str((tend - tstart)/60) + " minutes." ) # DEBUG

                if mod_t is None:
                    info += "\n      %s does not overlap the reference" % (fname)
                    continue
                mod_t += m.shift
                mod_tb += m.shift
                ind = np.where((mod_tb[:,0] >= obs_t0)*(mod_tb[:,1] <= obs_tf))[0]
                if ind.size == 0:
                    info += "\n      %s does not overlap the reference" % (fname)
                    continue
                mod_t  = mod_t [ind]
                mod_tb = mod_tb[ind]
                mod_t0 = min(mod_t0,mod_tb[ 0,0])
                mod_tf = max(mod_tf,mod_tb[-1,1])

                mod_dname = [name for name in dset.variables.keys() \
                             if name.lower() in ["depth_bnds", "depth_bounds"]]
                if len(mod_dname) == 0:
                    # if there is no depth, assume the data is surface
                    z0 = 0; zf = 0.1; mod_nd = 0; mod_dname = None
                else:
                    mod_dname = mod_dname[0]
                    temp = dset.variables[mod_dname][...]
                    ind = (temp[:,1] > obs_z0)*(temp[:,0] < obs_zf)
                    if sum(ind) == 0:
                        info += "\n      %s does not overlap the reference" % (fname)
                        continue
                    z0 = np.min(temp[ind, :])
                    zf = np.max(temp[ind, :])
                    mod_nd = min(mod_nd, sum(ind))
                mod_z0 = min(mod_z0,z0)
                mod_zf = max(mod_zf,zf)

                nt = mod_t.size
                mod_nt += nt
                mem = (var.size/var.shape[0]*nt)*8e-6
                mod_mem += mem
                info += "\n      %s spans years %.1f to %.1f and depths %.1f to %.1f, est memory in time bounds %d [Mb]" % (fname,mod_t.min()/365.+1850,mod_t.max()/365.+1850,mod_z0,mod_zf,mem)
        info += "\n      total est memory = %d [Mb]" % mod_mem
        logger.info("[%s][%s][%s] reading model data from possibly many files%s" % (self.name,m.name,vname,info))
        if mod_t0 > mod_tf:
            logger.debug("[%s] Could not find [%s] in the model results in the given time frame, tinput = [%.1f,%.1f]" % (self.name,",".join(possible),mod_t0,mod_tf))
            raise il.VarNotInModel()

        # yield the results by observational depths
        def _addDepth(v):
            v.depth = np.asarray([.05])
            v.depth_bnds = np.asarray([[0.,.1]])
            shp = list(v.data.shape)
            shp.insert(1,1)
            v.data.shape = shp
            v.layered = True
            return v

        info = ""
        for i in range(obs_z_bnd.shape[0]):
            ind = (self.depths[:,0] < obs_z_bnd[i,1]) & \
                  (self.depths[:,1] > obs_z_bnd[i,0]) & \
                  (self.depths[:,0] < mod_zf) & \
                  (self.depths[:,1] > mod_z0)
            if sum(ind) == 0:
                continue
            z0 = min(self.depths[ind,0])
            zf = max(self.depths[ind,1])

            mod_t0 = max(mod_t0,obs_t0)
            mod_tf = min(mod_tf,obs_tf)
            logger.info("[%s][%s] building depths %.1f to %.1f in loop %d" % (self.name,m.name,z0,zf,i))

            # get reference variable
            if obs_dname is None:
                obs = Variable(filename       = self.source,
                               variable_name  = self.variable,
                               alternate_vars = self.alternate_vars,
                               t0             = mod_t0,
                               tf             = mod_tf).trim(t = [mod_t0,mod_tf])
                obs = _addDepth(obs)
            else:
                obs = Variable(filename       = self.source,
                               variable_name  = self.variable,
                               alternate_vars = self.alternate_vars,
                               t0             = mod_t0,
                               tf             = mod_tf,
                               z0             = z0,
                               zf             = zf).trim(t = [mod_t0,mod_tf])
                obs = obs.integrateInDepth(z0 = z0, zf = zf, mean = True)
            obs.name = "depthint%.2f-%.2f" % (z0, zf)

            print("obs ", obs.name, obs.unit, obs.data, obs.time, obs_tb, obs.lat, obs.lat_bnds, 
                  obs.lon, obs.lon_bnds) # DEBUG

            # get model variable
            if mod_dname is None:
                mod = m.extractTimeSeries(self.variable,
                                          alt_vars     = self.alternate_vars,
                                          expression   = self.derived,
                                          initial_time = mod_t0,
                                          final_time   = mod_tf).trim(t=[mod_t0,mod_tf]).convert(obs.unit)
                mod = _addDepth(mod)
            else:
                mod = m.extractTimeSeries(self.variable,
                                          alt_vars     = self.alternate_vars,
                                          expression   = self.derived,
                                          initial_time = mod_t0,
                                          final_time   = mod_tf,
                                          initial_depth= z0,
                                          final_depth  = zf).trim(t=[mod_t0,mod_tf]).convert(obs.unit)
                mod = mod.trim(d = [z0, zf]).integrateInDepth(z0 = z0, zf = zf, mean = True)
            mod.name = "depthint%.2f-%.2f" % (z0, zf)

            print("mod ", mod.name, mod.unit, mod.data, mod.time, mod_tb, mod.lat, mod.lat_bnds, 
                  mod.lon, mod.lon_bnds) # DEBUG

            assert obs.time.size == mod.time.size

            obs.name = obs.name.split("_")[0]
            mod.name = mod.name.split("_")[0]

            yield obs, mod, z0, zf


    def confront(self,m):
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

            # Read in some options and run the mean state analysis
            mass_weighting = self.keywords.get("mass_weighting",False)
            skip_rmse      = self.keywords.get("skip_rmse"     ,False)
            skip_iav       = self.keywords.get("skip_iav"      ,False)
            skip_cycle     = self.keywords.get("skip_cycle"    ,False)
            rmse_score_basis = self.keywords.get("rmse_score_basis","cycle")

            # Get the depth-integrated observation and model data for each slab.
            for obs,mod,z0,zf in self.stageData(m):
                #YW
                print('Staging data ... %.2f-%.2f' % (z0, zf))
                print(obs.name)
                print(mod.name)

                if obs.spatial:
                    il.AnalysisMeanStateSpace(obs, mod, dataset   = fcm.mod_dset,
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
                    il.AnalysisMeanStateSites(obs, mod, dataset   = fcm.mod_dset,
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
        if has_cycle and len(models) > 0:
            page.addFigure("Spatially integrated regional mean",
                           "compcycle",
                           "RNAME_compcycle.png",
                           side   = "ANNUAL CYCLE",
                           legend = False)

        for region in self.regions:
            if region not in cycle: continue
            fig, axes = plt.subplots(self.depths.shape[0], 1,
                                     figsize = (6.5, 2.8*self.depths.shape[0]))
            for dind, z0 in enumerate(self.depths[:,0]):
                zf = self.depths[dind, 1]
                zstr = '%.2f-%.2f' % (z0, zf)

                if self.depths.shape[0] == 1:
                    ax = axes
                else:
                    ax = axes.flat[dind]

                for name,color,var in zip(models,colors,cycle[region][zstr]):
                    dy = 0.05*(self.limits["cycle"][region]["max"] - \
                               self.limits["cycle"][region]["min"])

                    var.plot(ax, lw=2, color=color, label=name,
                             ticks      = time_opts["cycle"]["ticks"],
                             ticklabels = time_opts["cycle"]["ticklabels"],
                             vmin       = self.limits["cycle"][region]["min"]-dy,
                             vmax       = self.limits["cycle"][region]["max"]+dy)
                    #ylbl = time_opts["cycle"]["ylabel"]
                    #if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                    ylbl = zstr + ' '+ self.depths_units
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
            page.addFigure("Temporally integrated period mean",
                           "spatial_variance",
                           "RNAME_spatial_variance.png",
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

            fig = plt.figure(figsize=(12.0,12.0))
            for dind, z0 in enumerate(self.depths[:,0]):
                zf = self.depths[dind, 1]
                zstr = '%.2f-%.2f' % (z0, zf)

                if not (zstr in std[region] and zstr in corr[region]): continue
                if len(std[region][zstr]) != len(corr[region][zstr]): continue
                if len(std[region][zstr]) == 0: continue
                ax, aux = post.TaylorDiagram(np.asarray(std[region][zstr]),
                                             np.asarray(corr[region][zstr]),
                                             1.0,fig,colors,True,220+dind+1)
                ax.set_title(zstr + ' ' + self.depths_units)
            fig.savefig(os.path.join(self.output_path,
                                     "%s_spatial_variance.png" % (region)))
            plt.close()

    def modelPlots(self,m):
        """For a given model, create the plots of the analysis results.

        This routine will extract plotting information out of the
        netCDF file which results from the analysis and create
        plots. Note that determinePlotLimits should be called before
        this routine.

        """
        self._relationship(m)
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

                # YW
                ##print(self.limits.keys())
                ##print(pname)

                if (var.spatial or (var.ndata is not None)) and not var.temporal:

                    # grab plotting options
                    if pname not in self.limits.keys(): continue
                    if pname not in space_opts: continue
                    opts = space_opts[pname]

                    ##print('... is used in space_opts') # DEBUG

                    # add to html layout
                    page.addFigure(opts["section"],
                                   pname,
                                   opts["pattern"],
                                   side   = opts["sidelbl"],
                                   legend = opts["haslegend"])

                    # plot variable
                    for region in self.regions:
                        nax = self.depths.shape[0]
                        fig = plt.figure()
                        for dind, z0 in enumerate(self.depths[:,0]):
                            zf = self.depths[dind,1]
                            zstr = '%.2f-%.2f' % (z0, zf)
                            var2 = Variable(filename=fname, groupname = "MeanState",
                                            variable_name=vname.replace(zstr_0, zstr))
                            ax = var2.plot(None, fig, nax, region = region,
                                           vmin   = self.limits[pname]["min"],
                                           vmax   = self.limits[pname]["max"],
                                           cmap   = self.limits[pname]["cmap"])
                            ax.set_title(zstr + ' ' + self.depths_units)
                        fig.savefig(os.path.join(self.output_path,
                                                 "%s_%s_%s.png" % (m.name,region,pname)))
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
                            nax = self.depths.shape[0]
                            fig = plt.figure()
                            for dind, z0 in enumerate(self.depths[:,0]):
                                zf = self.depths[dind,1]
                                zstr = '%.2f-%.2f' % (z0, zf)
                                obs = Variable(filename=bname,groupname="MeanState",
                                               variable_name=vname.replace(zstr_0, zstr))
                                ax = obs.plot(None, fig, nax, region = region,
                                              vmin   = self.limits[pname]["min"],
                                              vmax   = self.limits[pname]["max"],
                                              cmap   = self.limits[pname]["cmap"])
                                ax.set_title(zstr + ' ' + self.depths_units)
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
                            if self.depths.shape[0] == 1:
                                ax = axes
                            else:
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
