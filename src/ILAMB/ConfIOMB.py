from .Confrontation import getVariableList
from .Confrontation import Confrontation
from .constants import earth_rad,mid_months,lbl_months,bnd_months
from .Variable import Variable
from .Regions import Regions
from . import ilamblib as il
from . import Post as post
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

        # Get/modify depths
        self.depths = np.asarray(self.keywords.get("depths",[0,100,250]),dtype=float)
        with Dataset(self.source) as dset:
            v = dset.variables[self.variable]
            depth_name = [d for d in v.dimensions if d in ["layer","depth"]]

            if len(depth_name) == 0:
                # if there is no depth dimension, we assume the data is surface
                self.depths = np.asarray([0],dtype=float)
            else:
                # if there are depths, then make sure that the depths
                # at which we will compute are in the range of depths
                # of the data
                depth_name = depth_name[0]
                data = dset.variables[dset.variables[depth_name].bounds][...] if "bounds" in dset.variables[depth_name].ncattrs() else dset.variables[depth_name][...]
                self.depths = self.depths[(self.depths>=data.min())*(self.depths<=data.max())]

        # Setup a html layout for generating web views of the results
        pages       = []
        sections    = ["Period Mean at %d [m]" % d for d in self.depths]
        sections   += ["Mean regional depth profiles"]
        sections   += ["Overlapping mean regional depth profiles"]
        sections   += ["Mean regional annual cycle"]
        sections   += ["Overlapping mean regional annual cycle"]
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
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (attr,str(dset.getncattr(attr)).encode('ascii','ignore'))
        self.layout = post.HtmlLayout(pages,self.longname)

    def stageData(self,m):

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
                    mod_tb = mod.time_bnds[...]
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
            v.depth = np.asarray([5.])
            v.depth_bnds = np.asarray([[0.,10.]])
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
            max_obs = -1e20
            for obs,mod in self.stageData(m):

                # if the data has no depth, we assume it is surface
                if not obs.layered: obs = _addDepth(obs)
                if not mod.layered: mod = _addDepth(mod)
                max_obs = max(max_obs,obs.data.max())

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
                    z = mod
                    if mod.layered: z = z.integrateInDepth(z0=depth-1.,zf=depth+1,mean=True)
                    z = z.integrateInTime(mean=True)
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
                    if region not in ocyc:
                        ocyc[region] = np.ma.zeros((12,)+z.data.shape[1:])
                        oN  [region] = np.ma.zeros((12,)+z.data.shape[1:],dtype=int)
                    i = (np.abs(mid_months[:,np.newaxis]-(z.time % 365))).argmin(axis=0)
                    (ocyc[region])[i,...] += z.data
                    (oN  [region])[i,...] += 1

                    z = mod.integrateInSpace(region=region,mean=True)
                    if region not in mcyc:
                        mcyc[region] = np.ma.zeros((12,)+z.data.shape[1:])
                        mN  [region] = np.ma.zeros((12,)+z.data.shape[1:],dtype=int)
                    i = (np.abs(mid_months[:,np.newaxis]-(z.time % 365))).argmin(axis=0)
                    (mcyc[region])[i,...] += z.data
                    (mN  [region])[i,...] += 1

            # combine time slabs from the different depths
            large_bias = float(self.keywords.get("large_bias",0.1*max_obs))

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
                bias_score = None
                if dlbl == "0":
                    with np.errstate(all="ignore"):
                        bias_score = Variable(name  = bias.name.replace("bias","biasscore"),
                                              data  = np.exp(-np.abs(bias.data)/large_bias),
                                              unit  = "1",
                                              ndata = bias.ndata,
                                              lat   = bias.lat, lat_bnds = bias.lat_bnds,
                                              lon   = bias.lon, lon_bnds = bias.lon_bnds,
                                              area  = bias.area)
                        bias_score.toNetCDF4(fcm.mod_dset,group="MeanState")

                for region in self.regions:

                    sval = mod_tmp.integrateInSpace(region=region,mean=True)
                    sval.name = "Period Mean at %s %s" % (dlbl,region)
                    sval.toNetCDF4(fcm.mod_dset,group="MeanState")

                    sval = bias.integrateInSpace(region=region,mean=True)
                    sval.name = "Bias at %s %s" % (dlbl,region)
                    sval.toNetCDF4(fcm.mod_dset,group="MeanState")

                    if bias_score is not None:
                        sval = bias_score.integrateInSpace(region=region,mean=True)
                        sval.name = "Bias Score at %s %s" % (dlbl,region)
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
            fcm.mod_dset.setncattr("complete",1)
            if self.master: fcm.obs_dset.setncattr("complete",1)

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
        cmap = { "timeint"    : self.cmap,
                 "bias"       : "seismic",
                 "biasscore"  : "RdYlGn" }
        plbl = { "timeint"    : "MEAN",
                 "bias"       : "BIAS",
                 "biasscore"  : "BIAS SCORE" }
        with Dataset(fname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")
            for ptype in ["timeint","bias","biasscore"]:
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
                        ax = var.plot(None,
                                      region = region,
                                      vmin   = self.limits[vname]["min"],
                                      vmax   = self.limits[vname]["max"],
                                      cmap   = cmap[ptype],
                                      land   = 0.750,
                                      water  = 0.875)
                        fig = ax.get_figure()
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
                        ax = var.plot(None,
                                      region = region,
                                      vmin   = self.limits[vname]["min"],
                                      vmax   = self.limits[vname]["max"],
                                      cmap   = cmap[ptype],
                                      land   = 0.750,
                                      water  = 0.875)
                        fig = ax.get_figure()
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
                    if "_score" in vname:
                        pname = "_".join(vname.split("_")[:2])
                    if "_over_" in vname:
                        region = vname.split("_over_")[-1]
                        if pname not in limits: limits[pname] = {}
                        if region not in limits[pname]:
                            limits[pname][region] = {}
                            limits[pname][region]["min"]  = +1e20
                            limits[pname][region]["max"]  = -1e20
                            limits[pname][region]["unit"] = post.UnitStringToMatplotlib(var.getncattr("units"))
                        limits[pname][region]["min"] = min(limits[pname][region]["min"],var.getncattr("min"))
                        limits[pname][region]["max"] = max(limits[pname][region]["max"],var.getncattr("max"))
                    else:
                        if pname not in limits:
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
            if base_pname in cmaps:
                cmap = cmaps[base_pname]
            elif "score" in pname:
                cmap = "RdYlGn"

            # Need to symetrize?
            if base_pname in ["bias","timelonbias","cyclebias"]:
                if "min" in limits[pname]:
                    vabs =  max(abs(limits[pname]["max"]),abs(limits[pname]["min"]))
                    limits[pname]["min"] = -vabs
                    limits[pname]["max"] =  vabs
                else:
                    vabs =  max(abs(limits[pname]["global"]["max"]),abs(limits[pname]["global"]["min"]))
                    limits[pname]["global"]["min"] = -vabs
                    limits[pname]["global"]["max"] =  vabs

            # Some plots need legends
            if base_pname in ["timeint","bias","biasscore","rmse","rmsescore","timelonint","timelonbias","cycle","cyclebias"]:
                if "min" in limits[pname]:
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
