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

            obs_t,obs_tb,obs_cb,obs_b,obs_e,cal = il.GetTime(var)

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
                obs_z0 = 0; obs_zf = 0.1; obs_nd = 0
                obs_dname = None
            else:
                obs_dname = obs_dname[0]
                obs_z0 = np.min(dset.variables[obs_dname])
                obs_zf = np.max(dset.variables[obs_dname])
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
            with Dataset(fname) as dset:
                var = dset.variables[vname]
                mod_t,mod_tb,mod_cb,mod_b,mod_e,cal = il.GetTime(var,t0=obs_t0-m.shift,
                                                                 tf=obs_tf-m.shift)

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
        for i in range(self.depths.shape[0]):
            z0 = max(self.depths[i,0], obs_z0, mod_z0)
            zf = min(self.depths[i,1], obs_zf, mod_zf)
            if z0 >= zf:
                continue

            mod_t0 = max(mod_t0,obs_t0)
            mod_tf = min(mod_tf,obs_tf)
            logger.info("[%s][%s] building depths %.1f to %.1f in loop %d" % (self.name,m.name,z0,zf,i))

            # get reference variable
            print('Loading obs ' + str(z0) + '-' + str(zf))
            tstart = time.time() # DEBUG

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

            tend = time.time() # DEBUG
            print("Loading obs " + str(z0) + '-' + str(zf) + ' took ' + str((tend - tstart) / 60)) # DEBUG
            print("obs ", obs.name, obs.unit, obs.data, obs.time, obs_tb, obs.lat, obs.lat_bnds, 
                  obs.lon, obs.lon_bnds) # DEBUG

            # get model variable
            print('Loading model ' + str(z0) + '-' + str(zf))
            tstart = time.time() # DEBUG

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

            tend = time.time() # DEBUG
            print("Loading model " + str(z0) + '-' + str(zf) + ' took ' + str((tend - tstart) / 60)) # DEBUG
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
                print('Staging data ... %.2f-%.2f' % (z0, zf)) # DEBUG
                print(obs.name) # DEBUG
                print(mod.name) # DEBUG

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
                    ylbl = post.UnitStringToMatplotlib(var.unit)
                    ax.set_ylabel(ylbl)
                    ax.set_title(zstr + ' '+ self.depths_units)
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
                            ax.set_title(zstr + ' ' + self.depths_units)
                        fig.savefig(os.path.join(self.output_path,"%s_%s_%s.png" % (m.name,region,pname)))
                        plt.close()

        logger.info("[%s][%s] Success" % (self.longname,m.name))

    def _relationship(self,m,nbin=25):
        """
        Modified to plot by depths.
        """
        def _retrieveData(filename):
            key_list = []
            with Dataset(filename,mode="r") as dset:
                for dind, z0 in enumerate(self.depths[:,0]):
                    zf = self.depths[dind, 1]
                    zstr = '%.2f-%.2f' % (z0, zf)
                    key = [v for v in dset.groups["MeanState"].variables.keys() \
                           if ("timeint_" in v) and (zstr in v)]
                    if len(key) == 0:
                        raise "Unable to retrieve data for relationship " + zstr
                    key_list.append(key)
            return [Variable(filename      = filename,
                             groupname     = "MeanState",
                             variable_name = key) for key in key_list]
        
        def _applyRefMask(ref,com):
            tmp = ref.interpolate(lat=com.lat,lat_bnds=com.lat_bnds,
                                  lon=com.lon,lon_bnds=com.lon_bnds)
            com.data.mask += tmp.data.mask
            return com
        
        def _checkLim(data,lim):
            if lim is None:
                lim     = [min(data.min(),data.min()),
                           max(data.max(),data.max())]
                delta   = 1e-8*(lim[1]-lim[0])
                lim[0] -= delta
                lim[1] += delta
            else:
                assert type(lim) == type([])
                assert len (lim) == 2
            return lim

        def _limitExtents(vars):
            lim = [+1e20,-1e20]
            for v in vars:
                lmin,lmax = _checkLim(v.data,None)
                lim[0] = min(lmin,lim[0])
                lim[1] = max(lmax,lim[1])
            return lim

        def _buildDistributionResponse(ind,dep,ind_lim=None,dep_lim=None,region=None,nbin=25,eps=3e-3):

            r = Regions()

            # Checks on the input parameters
            assert np.allclose(ind.data.shape,dep.data.shape)
            ind_lim = _checkLim(ind.data,ind_lim)
            dep_lim = _checkLim(dep.data,dep_lim)

            # Mask data
            mask = ind.data.mask + dep.data.mask
            if region is not None: mask += r.getMask(region,ind)
            x = ind.data[mask==False].flatten()
            y = dep.data[mask==False].flatten()

            # Compute normalized 2D distribution
            dist,xedges,yedges = np.histogram2d(x,y,
                                                bins  = [nbin,nbin],
                                                range = [ind_lim,dep_lim])
            dist  = np.ma.masked_values(dist.T,0).astype(float)
            dist /= dist.sum()

            # Compute the functional response
            which_bin = np.digitize(x,xedges).clip(1,xedges.size-1)-1
            mean = np.ma.zeros(xedges.size-1)
            std  = np.ma.zeros(xedges.size-1)
            cnt  = np.ma.zeros(xedges.size-1)
            np.seterr(under='ignore')
            for i in range(mean.size):
                yi = y[which_bin==i]
                cnt [i] = yi.size
                mean[i] = yi.mean()
                std [i] = yi.std()
            mean = np.ma.masked_array(mean,mask = (cnt/cnt.sum()) < eps)
            std  = np.ma.masked_array( std,mask = (cnt/cnt.sum()) < eps)
            np.seterr(under='warn')
            return dist,xedges,yedges,mean,std

        def _scoreDistribution(ref,com):
            mask = ref.mask + com.mask
            ref  = np.ma.masked_array(ref.data,mask=mask).compressed()
            com  = np.ma.masked_array(com.data,mask=mask).compressed()
            return np.sqrt(((np.sqrt(ref)-np.sqrt(com))**2).sum())/np.sqrt(2)

        def _scoreFunction(ref,com):
            mask = ref.mask + com.mask
            ref  = np.ma.masked_array(ref.data,mask=mask).compressed()
            com  = np.ma.masked_array(com.data,mask=mask).compressed()
            return np.exp(-np.linalg.norm(ref-com)/np.linalg.norm(ref))

        def _plotDistribution(dist_list,xedges_list,yedges_list,
                              xlabel_list, ylabel_list, filename):
            fig, axes = plt.subplots(len(self.depths[:,0]), 1, 
                                     figsize=(3 * len(self.depths[:,0]), 3.),tight_layout=True)
            for ind, dist, xedges, yedges, xlabel, ylabel in \
                zip(range(len(self.depths[:,0])), dist_list, xedges_list, yedges_list,
                    xlabel_list, ylabel_list):
                ax = axes.flat[ind]
                pc = ax.pcolormesh(xedges, yedges, dist,
                                   norm = LogNorm(vmin = 1e-4,vmax = 1e-1),
                                   cmap = 'plasma' if 'plasma' in plt.cm.cmap_d else 'summer')
                ax.set_xlabel(xlabel,fontsize = 12)
                ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
                ax.set_xlim(xedges[0],xedges[-1])
                ax.set_ylim(yedges[0],yedges[-1])
            fig.colorbar(pc, cax = fig.add_axes([0.95, 0.1, 0.02, 0.8]),
                         orientation="vertical",label="Fraction of total datasites")
            fig.savefig(filename)
            plt.close()

        def _plotDifference(ref_list,com_list,xedges_list,yedges_list,xlabel_list,ylabel_list,filename):
            fig, axes = plt.subplots(len(self.depths[:,0]), 1,
                                     figsize=(3 * len(self.depths[:,0]), 3.),tight_layout=True)
            for ind, ref, com, xedges, yedges, xlabel, ylabel in \
                zip(range(len(self.depths[:,0])), ref_list, com_list, xedges_list, yedges_list,
                    xlabel_list, ylabel_list):
                ref = np.ma.copy(ref)
                com = np.ma.copy(com)
                ref.data[np.where(ref.mask)] = 0.
                com.data[np.where(com.mask)] = 0.
                diff = np.ma.masked_array(com.data-ref.data,mask=ref.mask*com.mask)
                lim = np.abs(diff).max()

                pc = ax.pcolormesh(xedges, yedges, diff,
                                   cmap = 'Spectral_r',
                                   vmin = -lim, vmax = +lim)
                ax.set_xlabel(xlabel,fontsize = 12)
                ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
                ax.set_xlim(xedges[0],xedges[-1])
                ax.set_ylim(yedges[0],yedges[-1])
            fig.colorbar(pc,cax = fig.add_axes([0.95, 0.1, 0.02, 0.8]),
                         orientation="vertical",label="Distribution Difference")
            fig.savefig(filename)
            plt.close()

        def _plotFunction(ref_mean_list,ref_std_list,com_mean_list,com_std_list,
                          xedges_list,yedges_list,xlabel_list,ylabel_list,color,filename):
            fig, axes = plt.subplots(len(self.depths[:,0]), 1, 
                                     figsize=(3 * len(self.depths[:,0]), 3.),tight_layout=True)
            for ind, dist, xedges, yedges, xlabel, ylabel in \
                zip(range(len(self.depths[:,0])), dist_list, xedges_list, yedges_list,
                    xlabel_list, ylabel_list):

                xe    = 0.5*(xedges[:-1]+xedges[1:])
                delta = 0.1*np.diff(xedges).mean()

                # reference function
                ref_x = xe - delta
                ref_y = ref_mean
                ref_e = ref_std
                if not (ref_mean.mask==False).all():
                    ind   = np.where(ref_mean.mask==False)
                    ref_x = xe      [ind]-delta
                    ref_y = ref_mean[ind]
                    ref_e = ref_std [ind]
    
                # comparison function
                com_x = xe + delta
                com_y = com_mean
                com_e = com_std
                if not (com_mean.mask==False).all():
                    ind   = np.where(com_mean.mask==False)
                    com_x = xe      [ind]-delta
                    com_y = com_mean[ind]
                    com_e = com_std [ind]
    
                ax = axes.flat[ind]
                ax.errorbar(ref_x,ref_y,yerr=ref_e,fmt='-o',color='k')
                ax.errorbar(com_x,com_y,yerr=com_e,fmt='-o',color=color)
                ax.set_xlabel(xlabel,fontsize = 12)
                ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
                ax.set_xlim(xedges[0],xedges[-1])
                ax.set_ylim(yedges[0],yedges[-1])
            fig.savefig(filename)
            plt.close()

        # If there are no relationships to analyze, get out of here
        if self.relationships is None: return

        # Get the HTML page
        page = [page for page in self.layout.pages if "Relationships" in page.name]
        if len(page) == 0: return
        page = page[0]

        # Try to get the dependent data from the model and obs
        try:
            ref_dep_list  = _retrieveData(os.path.join(self.output_path,"%s_%s.nc" % (self.name,"Benchmark")))
            com_dep_list  = _retrieveData(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name     )))
            com_dep_list  = [_applyRefMask(ref_dep, com_dep) for ref_dep,com_dep in zip(ref_dep_list,com_dep_list)]
            dep_name = self.longname.split("/")[0]
            dep_min  = self.limits["timeint"]["min"]
            dep_max  = self.limits["timeint"]["max"]
        except:
            return

        with Dataset(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name)),mode="r+") as results:

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
                    ref_ind_list  = _retrieveData(os.path.join(c.output_path,"%s_%s.nc" % (c.name,"Benchmark")))
                    com_ind_list  = _retrieveData(os.path.join(c.output_path,"%s_%s.nc" % (c.name,m.name     )))
                    com_ind_list  = _applyRefMask(ref_ind,com_ind)
                    ind_name = c.longname.split("/")[0]
                    ind_min  = c.limits["timeint"]["min"]-1e-12
                    ind_max  = c.limits["timeint"]["max"]+1e-12
                except:
                    continue


                # Add figures to the html page
                page.addFigure(c.longname,
                               "benchmark_rel_%s"            % ind_name,
                               "Benchmark_RNAME_rel_%s.png"  % ind_name,
                               legend    = False,
                               benchmark = False)
                page.addFigure(c.longname,
                               "rel_%s"                      % ind_name,
                               "MNAME_RNAME_rel_%s.png"      % ind_name,
                               legend    = False,
                               benchmark = False)
                page.addFigure(c.longname,
                               "rel_diff_%s"                 % ind_name,
                               "MNAME_RNAME_rel_diff_%s.png" % ind_name,
                               legend    = False,
                               benchmark = False)
                page.addFigure(c.longname,
                               "rel_func_%s"                 % ind_name,
                               "MNAME_RNAME_rel_func_%s.png" % ind_name,
                               legend    = False,
                               benchmark = False)

                # Try to get the dependent data from the model and obs
                try:
                    ref_dep_list  = _retrieveData(os.path.join(self.output_path,"%s_%s.nc" % (self.name,"Benchmark")))
                    com_dep_list  = _retrieveData(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name     )))
                    com_dep_list  = [_applyRefMask(ref_dep, com_dep) for ref_dep,com_dep in zip(ref_dep_list,com_dep_list)]
                    dep_name = self.longname.split("/")[0]
                    dep_min  = self.limits["timeint"]["min"]
                    dep_max  = self.limits["timeint"]["max"]
                except:
                    return

                # Analysis over regions
                lim_dep  = [dep_min,dep_max]
                lim_ind  = [ind_min,ind_max]
                longname = c.longname.split('/')[0]
                for region in self.regions:
                    ref_dist_list = []
                    ref_xedges_list = []
                    ref_yedges_list = []
                    ref_mean_list = []
                    ref_std_list = []

                    com_dist_list = []
                    com_xedges_list = []
                    com_yedges_list = []
                    com_mean_list = []
                    com_std_list = []

                    for dind, ref_dep, ref_ind in zip(range(len(ref_dep_list)), ref_dep_list, ref_ind_list):
                        # Check on data shape
                        if not np.allclose(ref_dep.data.shape,ref_ind.data.shape):
                            msg = "[%s][%s] Data size mismatch in relationship: %s %s vs. %s %s" % (self.longname,m.name,dep_name,str(ref_dep.data.shape),ind_name,str(ref_ind.data.shape))
                            logger.debug(msg)
                            raise ValueError

                        ref_dist, ref_xedges, ref_yedges, ref_mean, ref_std = _buildDistributionResponse(ref_ind,ref_dep,ind_lim=lim_ind,dep_lim=lim_dep,region=region)
                        com_dist, com_xedges, com_yedges, com_mean, com_std = _buildDistributionResponse(com_ind,com_dep,ind_lim=lim_ind,dep_lim=lim_dep,region=region)

                        ref_dist_list.append(ref_dist)
                        ref_xedges_list.append(ref_xedges)
                        ref_yedges_list.append(ref_yedges)
                        ref_mean_list.append(ref_mean)
                        ref_std_list.append(ref_std)

                        com_dist_list.append(com_dist)
                        com_xedges_list.append(com_xedges)
                        com_yedges_list.append(com_yedges)
                        com_mean_list.append(com_mean)
                        com_std_list.append(com_std)

                    # Make the plots
                    _plotDistribution(ref_dist_list,ref_xedges_list,ref_yedges_list,
                                      "%s/%s,  %s" % (ind_name,   c.name,post.UnitStringToMatplotlib(ref_ind.unit)),
                                      "%s/%s,  %s" % (dep_name,self.name,post.UnitStringToMatplotlib(ref_dep.unit)),
                                      os.path.join(self.output_path,"%s_%s_rel_%s.png" % ("Benchmark",region,ind_name)))
                    _plotDistribution(com_dist_list,com_xedges_list,com_yedges_list,
                                      "%s/%s,  %s" % (ind_name,m.name,post.UnitStringToMatplotlib(com_ind.unit)),
                                      "%s/%s,  %s" % (dep_name,m.name,post.UnitStringToMatplotlib(com_dep.unit)),
                                      os.path.join(self.output_path,"%s_%s_rel_%s.png" % (m.name,region,ind_name)))
                    _plotDifference  (ref_dist_list,com_dist_list,ref_xedges_list,ref_yedges_list,
                                      "%s/%s,  %s" % (ind_name,m.name,post.UnitStringToMatplotlib(com_ind.unit)),
                                      "%s/%s,  %s" % (dep_name,m.name,post.UnitStringToMatplotlib(com_dep.unit)),
                                      os.path.join(self.output_path,"%s_%s_rel_diff_%s.png" % (m.name,region,ind_name)))
                    _plotFunction(ref_mean_list,ref_std_list,com_mean_list,
                                  com_std_list,ref_xedges_list,ref_yedges_list,
                                  "%s,  %s" % (ind_name,post.UnitStringToMatplotlib(com_ind.unit)),
                                  "%s,  %s" % (dep_name,post.UnitStringToMatplotlib(com_dep.unit)),
                                  m.color,
                                  os.path.join(self.output_path,"%s_%s_rel_func_%s.png" % (m.name,region,ind_name)))

                # Score the distribution
                score_list = []
                for ref_dist, com_dist in zip(ref_dist_list, com_dist_list):
                    score = _scoreDistribution(ref_dist,com_dist)
                    score_list.append(score)
                score = np.mean(score_list) # !!!!!!! May be wrong?
                sname = "%s Hellinger Distance %s" % (longname,region)
                if sname in scalars.variables:
                    scalars.variables[sname][0] = score
                else:
                    Variable(name = sname,
                             unit = "1",
                             data = score).toNetCDF4(results,group="Relationships")

                # Score the functional response
                score = _scoreFunction(ref_dist[3],com_dist[3])
                sname = "%s RMSE Score %s" % (longname,region)
                if sname in scalars.variables:
                    scalars.variables[sname][0] = score
                else:
                    Variable(name = sname,
                             unit = "1",
                             data = score).toNetCDF4(results,group="Relationships")
