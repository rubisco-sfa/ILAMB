from ILAMB.Confrontation import getVariableList
from ILAMB.Confrontation import Confrontation
from ILAMB.constants import earth_rad,mid_months,lbl_months,bnd_months
from ILAMB.Variable import Variable
from ILAMB.Regions import Regions
import ILAMB.ilamblib as il
import ILAMB.Post as post
from netCDF4 import Dataset
from copy import deepcopy
import pylab as plt
import numpy as np
import os,glob,re
from sympy import sympify

from mpi4py import MPI
import logging
logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)	

def VariableReduce(var,region="global",time=None,depth=None,lat=None,lon=None):
    ILAMBregions = Regions()
    out = deepcopy(var)
    out.data.mask += ILAMBregions.getMask(region,out)
    if time  is not None:
        out = out.integrateInTime (t0=time[0] ,tf=time[1] ,mean=True)
    if depth is not None and var.layered:
        out = out.integrateInDepth(z0=depth[0],zf=depth[1],mean=True)
    if lat   is not None:
        lat0        = np.argmin(np.abs(var.lat-lat[0]))
        latf        = np.argmin(np.abs(var.lat-lat[1]))+1
        wgt         = earth_rad*(np.sin(var.lat_bnds[:,1])-np.sin(var.lat_bnds[:,0]))[lat0:latf]
        np.seterr(over='ignore',under='ignore')
        out.data    = np.ma.average(out.data[...,lat0:latf,:],axis=-2,weights=wgt/wgt.sum())
        np.seterr(over='raise',under='raise')
        out.lat     = None
        out.lat_bnd = None
        out.spatial = False
    if lon   is not None:
        lon0        = np.argmin(np.abs(var.lon-lon[0]))
        lonf        = np.argmin(np.abs(var.lon-lon[1]))+1
        wgt         = earth_rad*(var.lon_bnds[:,1]-var.lon_bnds[:,0])[lon0:lonf]
        np.seterr(over='ignore',under='ignore')
        out.data    = np.ma.average(out.data[...,lon0:lonf],axis=-1,weights=wgt/wgt.sum())
        np.seterr(over='raise',under='raise')
        out.lon     = None
        out.lon_bnd = None
        out.spatial = False
    return out

def TimeLatBias(ref,com):
    # composite depth axis
    d0 = max(ref.depth_bnds.min(),com.depth_bnds.min())
    df = min(ref.depth_bnds.max(),com.depth_bnds.max())
    d  = np.unique(np.hstack([ref.depth_bnds.flatten(),com.depth_bnds.flatten()]))
    d  = d[(d>=d0)*(d<=df)]
    db = np.asarray([d[:-1],d[1:]]).T
    d  = db.mean(axis=1)
    # composite lat axis
    l0 = max(ref.lat_bnds.min(),com.lat_bnds.min())
    lf = min(ref.lat_bnds.max(),com.lat_bnds.max())
    l  = np.unique(np.hstack([ref.lat_bnds.flatten(),com.lat_bnds.flatten()]))
    l  = l[(l>=l0)*(l<=lf)]
    lb = np.asarray([l[:-1],l[1:]]).T
    l  = lb.mean(axis=1)
    # interpolation / difference
    data  = il.NearestNeighborInterpolation(com.depth,com.lat,com.data,d,l)
    data -= il.NearestNeighborInterpolation(ref.depth,ref.lat,ref.data,d,l)
    area  = np.diff(db)[:,np.newaxis] * (earth_rad*(np.sin(lb[:,1])-np.sin(lb[:,0])))
    return Variable(name  = ref.name.replace("timelonint","timelonbias"),
                    unit  = ref.unit, 
                    data  = data,
                    area  = area,
                    lat   = l,
                    depth = d, 
                    lat_bnds   = lb, 
                    depth_bnds = db)

def CycleBias(ref,com):
    # composite depth axis
    d0 = max(ref.depth_bnds.min(),com.depth_bnds.min())
    df = min(ref.depth_bnds.max(),com.depth_bnds.max())
    d  = np.unique(np.hstack([ref.depth_bnds.flatten(),com.depth_bnds.flatten()]))
    d  = d[(d>=d0)*(d<=df)]
    db = np.asarray([d[:-1],d[1:]]).T
    d  = db.mean(axis=1)
    # interpolation / difference
    data  = il.NearestNeighborInterpolation(com.time,com.depth,com.data,com.time,d)
    data -= il.NearestNeighborInterpolation(ref.time,ref.depth,ref.data,ref.time,d)
    return Variable(name  = ref.name.replace("cycle","cyclebias"),
                    unit  = ref.unit, 
                    data  = data,
                    time  = mid_months,
                    time_bnds = np.asarray([bnd_months[:-1],bnd_months[1:]]).T,
                    depth      = d,
                    depth_bnds = db)

class ConfIOMB(Confrontation):

    def __init__(self,**keywords):

        # Calls the regular constructor
        super(ConfIOMB,self).__init__(**keywords)

        self.depths = np.asarray(self.keywords.get("depths",[0,100,250]),dtype=float)
        sections    = ["Period Mean at %d [m]" % d for d in self.depths]
        sections   += ["Mean regional depth profiles"]
        sections   += ["Overlapping mean regional depth profiles"]
        sections   += ["Mean regional annual cycle"]
        sections   += ["Overlapping mean regional annual cycle"]

        # Setup a html layout for generating web views of the results
        pages = []
        
        # Mean State page
        pages.append(post.HtmlPage("MeanState","Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(sections)
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation","Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        with Dataset(self.source) as dset:
            for attr in dset.ncattrs():
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (attr,dset.getncattr(attr).encode('ascii','ignore'))
        self.layout = post.HtmlLayout(pages,self.longname)

    def stageData(self,m):

        mem_slab = self.keywords.get("mem_slab",500.) # Mb
        
        # peak at the obs dataset without reading much into memory,
        # this assumes that the reference dataset was encoded using
        # our datum
        info = "obs: %s " % self.variable
        with Dataset(self.source) as dset:
            climatology = True if "climatology" in dset.variables["time"].ncattrs() else False
            ot          = dset.variables["time"       ].size
            om          = dset.variables[self.variable].size *8e-6
            unit        = dset.variables[self.variable].units
            if climatology:
                info += "climatology "
                t  = np.round(dset.variables[dset.variables["time"].climatology][...]/365.)*365.
                t0 = t[0,0]; tf = t[-1,1]
            else:
                if "bounds" in dset.variables["time"].ncattrs():
                    t = dset.variables[dset.variables["time"].bounds][...]
                    t0 = t[0,0]; tf = t[-1,1]
                else:
                    t = dset.variables["time"][...]
                    t0 = t[0]; tf = t[-1]
            info += "(%f,%f) %.0f" % (t0/365.+1850,tf/365.+1850,om)
        #print info

        # peak at the model dataset without reading much into memory,
        # it could be in the variable or alternate variables, or it
        # could be in the derived expression
        possible = [self.variable,] + self.alternate_vars
        if self.derived is not None: possible += [str(s) for s in sympify(self.derived).free_symbols]
        vname = ([v for v in possible if v in m.variables.keys()])
        if len(vname) == 0:
            logger.debug("[%s] Could not find [%s] in the model results" % (self.name,",".join(possible)))
            raise il.VarNotInModel()
        vname = vname[0]

        info = "mod: %s " % vname 
        mt = 0; mm = 0.; mt0 = 1e20; mtf = -1e20; shp = None
        for fname in m.variables[vname]:
            info += "\n  %s " % fname
            with Dataset(fname) as dset:
                t   = dset.variables["time"]
                tb  = dset.variables[dset.variables["time"].bounds] if "bounds" in dset.variables["time"].ncattrs() else None
                if tb:
                    t,tb = il.ConvertCalendar(t,tb)
                    t = tb
                else:
                    t = il.ConvertCalendar(t)
                t  += m.shift
                info += " (%.2f %.2f) " % (t.min()/365.+1850,t.max()/365.+1850)
                i   = (t >= t0)*(t <= tf)
                if i.any() == False: continue
                mt0 = min(mt0,t[i].min())
                mtf = max(mtf,t[i].max())
                info += " (%.2f %.2f) " % (mt0/365.+1850,mtf/365.+1850)
                nt  = i.sum()
                v   = dset.variables[vname]
                shp = v.shape
                mt += nt
                mm += (v.size / v.shape[0] * nt) * 8e-6
                info += "%.0f " % mm
        #print info

        if mt0 > mtf:
            logger.debug("[%s] Could not find [%s] in the model results in the given time frame, tinput = [%.1f,%.1f]" % (self.name,",".join(possible),t0,tf))
            raise il.VarNotInModel()

        # if obs is a climatology, then build a climatology in slabs
        if climatology:            
            mt0  = int(mt0/365)*365 + bnd_months[mid_months.searchsorted(mt0 % 365)  ]
            mtf  = int(mtf/365)*365 + bnd_months[mid_months.searchsorted(mtf % 365)+1]
            data = None
            dnum = None 
            ns   = int(mm/mem_slab)+1
            dt   = (mtf-mt0)/ns
            nm   = 0
            for i in range(ns):
                
                # find slab begin/end to the nearest month
                st0 = mt0 +  i   *dt
                st0 = int(st0/365)*365 + bnd_months[mid_months.searchsorted(st0 % 365)]
                stf = mt0 + (i+1)*dt
                stf = int(stf/365)*365 + bnd_months[mid_months.searchsorted(stf % 365)]
                v   = m.extractTimeSeries(vname,initial_time=st0,final_time=stf).trim(t=[st0,stf]).convert(unit)               
                nm += v.time.size

                # accumulate the sum for the mean cycle
                mind = mid_months.searchsorted(v.time % 365.)
                if data is None:
                    data = np.ma.zeros((12,)+v.data.shape[1:])
                    dnum = np.ma.zeros(data.shape,dtype=int)
                data[mind,...] += v.data
                dnum[mind,...] += 1

            data /= dnum
            obs = Variable(filename       = self.source,
                           variable_name  = self.variable,
                           alternate_vars = self.alternate_vars)
            mod = Variable(name  = obs.name,
                           unit  = obs.unit,
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
            info = "slabbing historical:\n"
            mt0  = int(mt0/365)*365 + bnd_months[mid_months.searchsorted(mt0 % 365)]
            mtf  = int(mtf/365)*365 + bnd_months[mid_months.searchsorted(mtf % 365)+1]
            info += "  (%f %f) (%f %f)\n" % (t0/365.+1850,tf/365.+1850,mt0/365.+1850,mtf/365.+1850)
            info += "  obs mem %f (%f) mod mem %f\n" % (om,om*(mtf-mt0)/(tf-t0),mm)
            om  *= (mtf-mt0)/(tf-t0)
            ns   = int(max(om,mm)/mem_slab)+1
            dt   = (mtf-mt0)/ns
            info += "  nslabs %d dt %f\n" % (ns,dt)
            #print info
            for i in range(ns):
                
                # find slab begin/end to the nearest month
                st0 = mt0 +  i   *dt
                st0 = int(st0/365)*365 + bnd_months[mid_months.searchsorted(st0 % 365)]
                stf = mt0 + (i+1)*dt
                stf = int(stf/365)*365 + bnd_months[mid_months.searchsorted(stf % 365)]
                obs = Variable(filename       = self.source,
                               variable_name  = self.variable,
                               alternate_vars = self.alternate_vars,
                               t0             = st0,
                               tf             = stf).trim(t=[st0,stf])
                mod = m.extractTimeSeries(vname,initial_time=st0,final_time=stf).trim(t=[st0,stf]).convert(obs.unit)                     
                yield obs,mod
              
    def confront(self,m):

        mod_file = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        obs_file = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name,      ))
        with il.FileContextManager(self.master,mod_file,obs_file) as fcm:

            # Encode some names and colors
            fcm.mod_dset.setncatts({"name" :m.name,
                                    "color":m.color})
            if self.master:
                fcm.obs_dset.setncatts({"name" :"Benchmark",
                                        "color":np.asarray([0.5,0.5,0.5])})
        
            obs_timeint = {}; mod_timeint = {}
            obs_depth   = {}; mod_depth   = {}
            ocyc        = {}; oN          = {}
            mcyc        = {}; mN          = {}
            for depth in self.depths:
                dlbl = "%d" % depth
                obs_timeint[dlbl] = []
                mod_timeint[dlbl] = []
            for region in self.regions:
                obs_depth[region] = []
                mod_depth[region] = []
            unit = None
            for obs,mod in self.stageData(m):

                # time bounds for this slab
                tb = obs.time_bnds[[0,-1],[0,1]].reshape((1,2))
                t  = np.asarray([tb.mean()])
                
                # mean lat/lon slices at various depths
                for depth in self.depths:
                    
                    dlbl = "%d" % depth
                    z  = obs.integrateInDepth(z0=depth-1.,zf=depth+1,mean=True).integrateInTime(mean=True)
                    unit = z.unit
                    obs_timeint[dlbl].append(Variable(name = "timeint%s" % dlbl,
                                                      unit = z.unit,
                                                      data = z.data.reshape((1,)+z.data.shape),
                                                      time = t,     time_bnds = tb,
                                                      lat  = z.lat, lat_bnds  = z.lat_bnds,
                                                      lon  = z.lon, lon_bnds  = z.lon_bnds))
                    z  = mod.integrateInDepth(z0=depth-1.,zf=depth+1,mean=True).integrateInTime(mean=True)
                    mod_timeint[dlbl].append(Variable(name = "timeint%s" % dlbl,
                                                      unit = z.unit,
                                                      data = z.data.reshape((1,)+z.data.shape),
                                                      time = t,     time_bnds = tb,
                                                      lat  = z.lat, lat_bnds  = z.lat_bnds,
                                                      lon  = z.lon, lon_bnds  = z.lon_bnds))

                # mean
                for region in self.regions:
                    z = VariableReduce(obs,region=region,time=tb[0],lon=[-180.,+180.])                    
                    z.time = t; z.time_bnds  = tb; z.temporal = True; z.data.shape = (1,)+z.data.shape
                    obs_depth[region].append(z)
                    
                    z = VariableReduce(mod,region=region,time=tb[0],lon=[-180.,+180.])                    
                    z.time = t; z.time_bnds  = tb; z.temporal = True; z.data.shape = (1,)+z.data.shape
                    mod_depth[region].append(z)

                # annual cycle in slabs
                for region in self.regions:
                    z = obs.integrateInSpace(region=region,mean=True)
                    if not ocyc.has_key(region): 
                        ocyc[region] = np.ma.zeros((12,)+z.data.shape[1:])
                        oN  [region] = np.ma.zeros((12,)+z.data.shape[1:],dtype=int)
                    i = mid_months.searchsorted(z.time % 365.)
                    (ocyc[region])[i,...] += z.data
                    (oN  [region])[i,...] += 1

                    z = mod.integrateInSpace(region=region,mean=True)
                    if not mcyc.has_key(region): 
                        mcyc[region] = np.ma.zeros((12,)+z.data.shape[1:])
                        mN  [region] = np.ma.zeros((12,)+z.data.shape[1:],dtype=int)
                    i = mid_months.searchsorted(z.time % 365.)
                    (mcyc[region])[i,...] += z.data
                    (mN  [region])[i,...] += 1
                
            # combine time slabs from the different depths
            for dlbl in obs_timeint.keys():

                # period means and bias
                obs_tmp = il.CombineVariables(obs_timeint[dlbl]).integrateInTime(mean=True)
                mod_tmp = il.CombineVariables(mod_timeint[dlbl]).integrateInTime(mean=True)
                obs_tmp.name = obs_tmp.name.split("_")[0]
                mod_tmp.name = mod_tmp.name.split("_")[0]
                bias = obs_tmp.spatialDifference(mod_tmp)
                bias.name = mod_tmp.name.replace("timeint","bias")
                mod_tmp.toNetCDF4(fcm.mod_dset,group="MeanState")
                bias.toNetCDF4(fcm.mod_dset,group="MeanState")
                for region in self.regions:
                    
                    sval = mod_tmp.integrateInSpace(region=region,mean=True)
                    sval.name = "Period Mean at %s %s" % (dlbl,region)
                    sval.toNetCDF4(fcm.mod_dset,group="MeanState")

                    sval = bias.integrateInSpace(region=region,mean=True)
                    sval.name = "Bias at %s %s" % (dlbl,region)
                    sval.toNetCDF4(fcm.mod_dset,group="MeanState")
                    
                if self.master:
                    obs_tmp.toNetCDF4(fcm.obs_dset,group="MeanState")
                    for region in self.regions:
                        sval = obs_tmp.integrateInSpace(region=region,mean=True)
                        sval.name = "Period Mean at %s %s" % (dlbl,region)
                        sval.toNetCDF4(fcm.obs_dset,group="MeanState")

            # combine depth/lat slabs for different regions
            for region in self.regions:
                mod_tmp = il.CombineVariables(mod_depth[region]).integrateInTime(mean=True)
                mod_tmp.name = "timelonint_of_%s_over_%s" % (self.variable,region)
                mod_tmp.toNetCDF4(fcm.mod_dset,group="MeanState")
                obs_tmp = il.CombineVariables(obs_depth[region]).integrateInTime(mean=True)
                obs_tmp.name = "timelonint_of_%s_over_%s" % (self.variable,region)
                mod_bias = TimeLatBias(obs_tmp,mod_tmp)
                mod_bias.toNetCDF4(fcm.mod_dset,group="MeanState")
                np.seterr(over='ignore',under='ignore')
                ocyc[region] = ocyc[region]/(oN[region].clip(1))
                mcyc[region] = mcyc[region]/(mN[region].clip(1))
                np.seterr(over='raise',under='raise')
                mcyc[region] = Variable(name = "cycle_of_%s_over_%s" % (self.variable,region),
                                        unit = mod.unit,
                                        data = mcyc[region],
                                        depth = mod.depth,
                                        depth_bnds = mod.depth_bnds,
                                        time = mid_months)
                ocyc[region] = Variable(name = "cycle_of_%s_over_%s" % (self.variable,region),
                                        unit = obs.unit,
                                        data = ocyc[region],
                                        depth = obs.depth,
                                        depth_bnds = obs.depth_bnds,
                                        time = mid_months)
                cyc_bias = CycleBias(ocyc[region],mcyc[region])
                cyc_bias    .toNetCDF4(fcm.mod_dset,group="MeanState")
                mcyc[region].toNetCDF4(fcm.mod_dset,group="MeanState")
                if self.master:
                    obs_tmp     .toNetCDF4(fcm.obs_dset,group="MeanState")
                    ocyc[region].toNetCDF4(fcm.obs_dset,group="MeanState")

    def modelPlots(self,m):
        
        def _fheight(region):
            if region in ["arctic","southern"]: return 6.8
            return 2.8
        
        bname  = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname  = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return
        
        # get the HTML page and set table priorities
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        page.priority  = [" %d " % d for d in self.depths]
        page.priority += ["Period","Bias"]
        page.priority += ["Score","Overall"]
        
        # model plots
        cmap = { "timeint" : self.cmap,
                 "bias"    : "seismic" }
        plbl = { "timeint" : "MEAN",
                 "bias"    : "BIAS" }
        with Dataset(fname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")
            for ptype in ["timeint","bias"]:
                for vname in [v for v in variables if ptype in v]:
                    var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                    try:
                        z = int(vname.replace(ptype,"")) 
                    except:
                        continue
                    page.addFigure("Period Mean at %d [m]" % z,
                                   vname,
                                   "MNAME_RNAME_%s.png" % vname,
                                   side   = "MODEL %s AT %d [m]" % (plbl[ptype],z),
                                   legend = True)
                    for region in self.regions:
                        fig = plt.figure()
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        var.plot(ax,
                                 region = region,
                                 vmin   = self.limits[vname]["min"],
                                 vmax   = self.limits[vname]["max"],
                                 cmap   = cmap[ptype],
                                 land   = 0.750,
                                 water  = 0.875)
                        fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,vname))
                        plt.close()


        for region in self.regions:

            vname = "timelonint_of_%s_over_%s" % (self.variable,region)
            if vname in variables:
                var0 = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                var  = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                bias = Variable(filename=fname,variable_name=vname.replace("timelonint","timelonbias"),groupname="MeanState")
                if region == "global":
                    page.addFigure("Mean regional depth profiles",
                                   "timelonint",
                                   "MNAME_RNAME_timelonint.png",
                                   side     = "MODEL DEPTH PROFILE",
                                   legend   = True,
                                   longname = "Time/longitude averaged profile")
                    page.addFigure("Overlapping mean regional depth profiles",
                                   "timelonints",
                                   "MNAME_RNAME_timelonints.png",
                                   side     = "MODEL DEPTH PROFILE",
                                   legend   = True,
                                   longname = "Overlapping Time/longitude averaged profile")
                    page.addFigure("Overlapping mean regional depth profiles",
                                   "timelonbias",
                                   "MNAME_RNAME_timelonbias.png",
                                   side     = "MODEL DEPTH PROFILE BIAS",
                                   legend   = True,
                                   longname = "Overlapping Time/longitude averaged profile bias")
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                l   = np.hstack([var .lat_bnds  [:,0],var .lat_bnds  [-1,1]])
                d0  = np.hstack([var0.depth_bnds[:,0],var0.depth_bnds[-1,1]])
                d   = np.hstack([var .depth_bnds[:,0],var .depth_bnds[-1,1]])
                ind = np.all(var.data.mask,axis=0)
                ind = np.ma.masked_array(range(ind.size),mask=ind,dtype=int)
                b   = ind.min()
                e   = ind.max()+1
                ax.pcolormesh(l[b:(e+1)],d,var.data[:,b:e],
                              vmin = self.limits["timelonint"]["global"]["min"],
                              vmax = self.limits["timelonint"]["global"]["max"],
                              cmap = self.cmap)
                ax.set_xlabel("latitude")
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                fig.savefig("%s/%s_%s_timelonint.png" % (self.output_path,m.name,region))
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/%s_%s_timelonints.png" % (self.output_path,m.name,region))                
                plt.close()
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                l   = np.hstack([bias.lat_bnds  [:,0],bias.lat_bnds  [-1,1]])
                d0  = np.hstack([var0.depth_bnds[:,0],var0.depth_bnds[-1,1]])
                d   = np.hstack([bias.depth_bnds[:,0],bias.depth_bnds[-1,1]])
                ind = np.all(bias.data.mask,axis=0)
                ind = np.ma.masked_array(range(ind.size),mask=ind,dtype=int)
                b   = ind.min()
                e   = ind.max()+1
                ax.pcolormesh(l[b:(e+1)],d,bias.data[:,b:e],
                              vmin = self.limits["timelonbias"]["global"]["min"],
                              vmax = self.limits["timelonbias"]["global"]["max"],
                              cmap = "seismic")
                ax.set_xlabel("latitude")
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/%s_%s_timelonbias.png" % (self.output_path,m.name,region))                
                plt.close()                    


            vname = "cycle_of_%s_over_%s" % (self.variable,region)
            if vname in variables:
                var0 = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                var  = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                bias = Variable(filename=fname,variable_name=vname.replace("cycle","cyclebias"),groupname="MeanState")
                if region == "global":
                    page.addFigure("Mean regional annual cycle",
                                   "cycle",
                                   "MNAME_RNAME_cycle.png",
                                   side     = "MODEL ANNUAL CYCLE",
                                   legend   = True,
                                   longname = "Annual cycle")
                    page.addFigure("Overlapping mean regional annual cycle",
                                   "cycles",
                                   "MNAME_RNAME_cycles.png",
                                   side     = "MODEL ANNUAL CYCLE",
                                   legend   = True,
                                   longname = "Overlapping annual cycle")
                    page.addFigure("Overlapping mean regional annual cycle",
                                   "cyclebias",
                                   "MNAME_RNAME_cyclebias.png",
                                   side     = "MODEL ANNUAL CYCLE BIAS",
                                   legend   = True,
                                   longname = "Overlapping annual cycle bias")
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                d0  = np.hstack([var0.depth_bnds[:,0],var0.depth_bnds[-1,1]])
                d   = np.hstack([var .depth_bnds[:,0],var .depth_bnds[-1,1]])
                ax.pcolormesh(bnd_months,d,var.data.T,
                              vmin = self.limits["cycle"]["global"]["min"],
                              vmax = self.limits["cycle"]["global"]["max"],
                              cmap = self.cmap)
                ax.set_xticks     (mid_months)
                ax.set_xticklabels(lbl_months)
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                fig.savefig("%s/%s_%s_cycle.png" % (self.output_path,m.name,region))
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/%s_%s_cycles.png" % (self.output_path,m.name,region))                
                plt.close()
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                ax.pcolormesh(bnd_months,
                              np.hstack([bias.depth_bnds[:,0],bias.depth_bnds[-1,1]]),
                              bias.data.T,
                              vmin = self.limits["cyclebias"]["global"]["min"],
                              vmax = self.limits["cyclebias"]["global"]["max"],
                              cmap = "seismic")
                ax.set_xticks     (mid_months)
                ax.set_xticklabels(lbl_months)
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/%s_%s_cyclebias.png" % (self.output_path,m.name,region))
                plt.close()


        # benchmark plots
        if not self.master: return
        with Dataset(bname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")            
            for ptype in ["timeint"]:
                for vname in [v for v in variables if ptype in v]:
                    var = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                    z   = int(vname.replace(ptype,"")) 
                    page.addFigure("Period Mean at %d [m]" % z,
                                   "benchmark_%s" % vname,
                                   "Benchmark_RNAME_%s.png" % vname,
                                   side   = "BENCHMARK %s AT %d [m]" % (plbl[ptype],z),
                                   legend = True)
                    for region in self.regions:
                        fig = plt.figure()
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        var.plot(ax,
                                 region = region,
                                 vmin   = self.limits[vname]["min"],
                                 vmax   = self.limits[vname]["max"],
                                 cmap   = cmap[ptype],
                                 land   = 0.750,
                                 water  = 0.875)
                        fig.savefig("%s/Benchmark_%s_%s.png" % (self.output_path,region,vname))                        
                        plt.close()

        for region in self.regions:

            vname = "timelonint_of_%s_over_%s" % (self.variable,region)
            if vname in variables:
                var0 = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                var  = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                if region == "global":
                    page.addFigure("Mean regional depth profiles",
                                   "benchmark_timelonint",
                                   "Benchmark_RNAME_timelonint.png",
                                   side   = "BENCHMARK DEPTH PROFILE",
                                   legend = True,
                                   longname = "Time/longitude averaged profile")
                    page.addFigure("Overlapping mean regional depth profiles",
                                   "benchmark_timelonints",
                                   "Benchmark_RNAME_timelonints.png",
                                   side   = "BENCHMARK DEPTH PROFILE",
                                   legend = True,
                                   longname = "Overlapping Time/longitude averaged profile")
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                l   = np.hstack([var .lat_bnds  [:,0],var .lat_bnds  [-1,1]])
                d0  = np.hstack([var0.depth_bnds[:,0],var0.depth_bnds[-1,1]])
                d   = np.hstack([var .depth_bnds[:,0],var .depth_bnds[-1,1]])
                ind = np.all(var.data.mask,axis=0)
                ind = np.ma.masked_array(range(ind.size),mask=ind,dtype=int)
                b   = ind.min()
                e   = ind.max()+1
                ax.pcolormesh(l[b:(e+1)],d,var.data[:,b:e],
                              vmin = self.limits["timelonint"]["global"]["min"],
                              vmax = self.limits["timelonint"]["global"]["max"],
                              cmap = self.cmap)
                ax.set_xlabel("latitude")
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                fig.savefig("%s/Benchmark_%s_timelonint.png" % (self.output_path,region))
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/Benchmark_%s_timelonints.png" % (self.output_path,region))                                
                plt.close()

            vname = "cycle_of_%s_over_%s" % (self.variable,region)
            if vname in variables:
                var0 = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                var  = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                if region == "global":
                    page.addFigure("Mean regional annual cycle",
                                   "benchmark_cycle",
                                   "Benchmark_RNAME_cycle.png",
                                   side     = "BENCHMARK ANNUAL CYCLE",
                                   legend   = True,
                                   longname = "Annual cycle")
                    page.addFigure("Overlapping mean regional annual cycle",
                                   "benchmark_cycles",
                                   "Benchmark_RNAME_cycles.png",
                                   side     = "BENCHMARK ANNUAL CYCLE",
                                   legend   = True,
                                   longname = "Overlapping annual cycle")
                fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                d  = np.hstack([var0.depth_bnds[:,0],var0.depth_bnds[-1,1]])
                d0 = np.hstack([var .depth_bnds[:,0],var .depth_bnds[-1,1]])
                ax.pcolormesh(bnd_months,d,var0.data.T,
                              vmin = self.limits["cycle"]["global"]["min"],
                              vmax = self.limits["cycle"]["global"]["max"],
                              cmap = self.cmap)
                ax.set_xticks     (mid_months)
                ax.set_xticklabels(lbl_months)
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                fig.savefig("%s/%s_%s_cycle.png" % (self.output_path,"Benchmark",region))
                ax.set_ylim((min(d0.max(),d.max()),max(d0.min(),d.min())))
                fig.savefig("%s/%s_%s_cycles.png" % (self.output_path,"Benchmark",region))                
                plt.close()
                        
    def determinePlotLimits(self):
        
        # Pick limit type
        max_str = "up99"; min_str = "dn99"
        if self.keywords.get("limit_type","99per") == "minmax":
            max_str = "max"; min_str = "min"
            
        # Determine the min/max of variables over all models
        limits = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            with Dataset(fname) as dataset:
                if "MeanState" not in dataset.groups: continue
                group     = dataset.groups["MeanState"]
                variables = [v for v in group.variables.keys() if (v not in group.dimensions.keys() and
                                                                   "_bnds" not in v                 and
                                                                   group.variables[v][...].size > 1)]
                for vname in variables:
                    var    = group.variables[vname]
                    pname  = vname.split("_")[ 0]
                    if "_over_" in vname:
                        region = vname.split("_over_")[-1]
                        if not limits.has_key(pname): limits[pname] = {}
                        if not limits[pname].has_key(region):
                            limits[pname][region] = {}
                            limits[pname][region]["min"]  = +1e20
                            limits[pname][region]["max"]  = -1e20
                            limits[pname][region]["unit"] = post.UnitStringToMatplotlib(var.getncattr("units"))
                        limits[pname][region]["min"] = min(limits[pname][region]["min"],var.getncattr("min"))
                        limits[pname][region]["max"] = max(limits[pname][region]["max"],var.getncattr("max"))
                    else:
                        if not limits.has_key(pname):
                            limits[pname] = {}
                            limits[pname]["min"]  = +1e20
                            limits[pname]["max"]  = -1e20
                            limits[pname]["unit"] = post.UnitStringToMatplotlib(var.getncattr("units"))
                        limits[pname]["min"] = min(limits[pname]["min"],var.getncattr(min_str))
                        limits[pname]["max"] = max(limits[pname]["max"],var.getncattr(max_str))

        # Another pass to fix score limits
        for pname in limits.keys():
            if "score" in pname:
                if "min" in limits[pname].keys():
                    limits[pname]["min"] = 0.
                    limits[pname]["max"] = 1.
                else:
                    for region in limits[pname].keys():
                        limits[pname][region]["min"] = 0.
                        limits[pname][region]["max"] = 1.
        self.limits = limits
        
        # Second pass to plot legends
        cmaps = {"bias"       :"seismic",
                 "timelonbias":"seismic",
                 "cyclebias"  :"seismic",
                 "rmse"       :"YlOrRd"}
        for pname in limits.keys():

            base_pname = pname
            m = re.search("(\D+)\d+",pname)
            if m: base_pname = m.group(1)
            
            # Pick colormap
            cmap = self.cmap
            if cmaps.has_key(base_pname):
                cmap = cmaps[base_pname]
            elif "score" in pname:
                cmap = "RdYlGn"

            # Need to symetrize?
            if base_pname in ["bias","timelonbias","cyclebias"]:
                if limits[pname].has_key("min"):
                    vabs =  max(abs(limits[pname]["max"]),abs(limits[pname]["min"]))
                    limits[pname]["min"] = -vabs
                    limits[pname]["max"] =  vabs
                else:
                    vabs =  max(abs(limits[pname]["global"]["max"]),abs(limits[pname]["global"]["min"]))
                    limits[pname]["global"]["min"] = -vabs
                    limits[pname]["global"]["max"] =  vabs

            # Some plots need legends
            if base_pname in ["timeint","bias","biasscore","rmse","rmsescore","timelonint","timelonbias","cycle","cyclebias"]:
                if limits[pname].has_key("min"):
                    fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
                    post.ColorBar(ax,
                                  vmin  = limits[pname]["min" ],
                                  vmax  = limits[pname]["max" ],
                                  label = limits[pname]["unit"],
                                  cmap  = cmap)
                    fig.savefig("%s/legend_%s.png" % (self.output_path,pname))
                    if base_pname == "timelonint" or base_pname == "cycle":
                        fig.savefig("%s/legend_%ss.png" % (self.output_path,pname))
                    plt.close()
                else:
                    fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
                    post.ColorBar(ax,
                                  vmin  = limits[pname]["global"]["min" ],
                                  vmax  = limits[pname]["global"]["max" ],
                                  label = limits[pname]["global"]["unit"],
                                  cmap  = cmap)
                    fig.savefig("%s/legend_%s.png" % (self.output_path,pname))
                    if base_pname == "timelonint" or base_pname == "cycle":
                        fig.savefig("%s/legend_%ss.png" % (self.output_path,pname))
                    plt.close()

    def compositePlots(self):
        pass
