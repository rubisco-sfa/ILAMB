from ILAMB.Confrontation import Confrontation
from ILAMB.Confrontation import getVariableList
from ILAMB.constants import earth_rad,mid_months,lbl_months
from ILAMB.Variable import Variable
from ILAMB.Regions import Regions
import ILAMB.ilamblib as il
import ILAMB.Post as post
from netCDF4 import Dataset
from copy import deepcopy
import pylab as plt
import numpy as np
import os,glob

def VariableReduce(var,region="global",time=None,depth=None,lat=None,lon=None):
    ILAMBregions = Regions()
    out = deepcopy(var)
    out.data.mask += ILAMBregions.getMask(region,out)
    if time  is not None:
        out = out.integrateInTime (t0=time[0] ,tf=time[1] ,mean=True)
    if depth is not None:
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

class ConfIOMB(Confrontation):

    def __init__(self,**keywords):

        # Calls the regular constructor
        super(ConfIOMB,self).__init__(**keywords)

        # Setup a html layout for generating web views of the results
        pages = []

        # Mean State page
        pages.append(post.HtmlPage("MeanState","Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Period mean at surface",
                               "Mean regional depth profiles"])
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

        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars)
        t0  = obs.time_bnds[ 0,0]
        tf  = obs.time_bnds[-1,1]
        if obs.cbounds is not None:
            t0 = (obs.cbounds[0]  -1850)*365.
            tf = (obs.cbounds[1]+1-1850)*365.
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars     = self.alternate_vars,
                                  expression   = self.derived,
                                  initial_time = t0,
                                  final_time   = tf,
                                  lats         = None if obs.spatial else obs.lat,
                                  lons         = None if obs.spatial else obs.lon).convert(obs.unit)

        # push into MakeComparable
        dmin = max(obs.depth_bnds.min(),mod.depth_bnds.min())
        dmax = min(obs.depth_bnds.max(),mod.depth_bnds.max())
        obs.trim(d=[dmin,dmax])
        mod.trim(d=[dmin,dmax])
        mod = mod.annualCycle()
        
        return obs,mod
        
    def confront(self,m):

        def _profileScore(ref,com,region):
            db  = np.unique(np.hstack([np.unique(ref.depth_bnds),np.unique(com.depth_bnds)]))
            d   = 0.5*(db[:-1]+db[1:])
            w   = np.diff(db)
            r   = ref.data[np.argmin(np.abs(d[:,np.newaxis]-ref.depth),axis=1)]
            c   = com.data[np.argmin(np.abs(d[:,np.newaxis]-com.depth),axis=1)]
            err = np.sqrt( (((r-c)**2)*w).sum() / ((r**2)*w).sum() ) # relative L2 error
            return Variable(name = "Profile Score %s" % region,
                            unit = "1",
                            data = np.exp(-err))
        
        # get the data
        obs,mod = self.stageData(m)

        # Reduction 1: Surface states
        ds   = [   0.,  10.]
        ts   = [-1e20,+1e20]
        o1   = VariableReduce(obs,time=ts,depth=ds)
        m1   = VariableReduce(mod,time=ts,depth=ds)
        d1   = o1.bias(m1)
        s1   = il.Score(d1,o1.interpolate(lat=d1.lat,lon=d1.lon))
        m1.name = "timeint_surface_%s"    % self.variable
        d1.name = "bias_surface_%s"       % self.variable
        o1.name = "timeint_surface_%s"    % self.variable
        s1.name = "bias_score_surface_%s" % self.variable

        o2 = {}; m2 = {}; o3 = {}; m3 = {}; o4 = {}; m4 = {}
        op = {}; mp = {}; mb = {}; sb = {}; sp = {}
        for region in self.regions:

            op[region] = o1.integrateInSpace(mean=True,region=region)
            mp[region] = m1.integrateInSpace(mean=True,region=region)
            mb[region] = d1.integrateInSpace(mean=True,region=region)
            sb[region] = s1.integrateInSpace(mean=True,region=region)
            op[region].name = "Period Mean %s" % region
            mp[region].name = "Period Mean %s" % region
            mb[region].name = "Bias %s"        % region
            sb[region].name = "Bias Score %s"  % region
            
            # Reduction 2/3: Zonal depth profiles
            o2[region] = VariableReduce(obs,region,time=ts,lon=[-180.,180.])
            m2[region] = VariableReduce(mod,region,time=ts,lon=[-180.,180.])
            o3[region] = obs.integrateInSpace(region=region,mean=True).integrateInTime(t0=ts[0],tf=ts[1],mean=True)
            m3[region] = mod.integrateInSpace(region=region,mean=True).integrateInTime(t0=ts[0],tf=ts[1],mean=True)
            sp[region] = _profileScore(o3[region],m3[region],region)
            o2[region].name = "timelonint_of_%s_over_%s" % (self.variable,region)
            m2[region].name = "timelonint_of_%s_over_%s" % (self.variable,region)
            o3[region].name = "profile_of_%s_over_%s"    % (self.variable,region)
            m3[region].name = "profile_of_%s_over_%s"    % (self.variable,region)
            
            # Reduction 4: Temporal depth profile
            o4[region] = obs.integrateInSpace(region=region,mean=True)
            m4[region] = mod.integrateInSpace(region=region,mean=True)
            o4[region].name = "latlonint_of_%s_over_%s" % (self.variable,region)
            m4[region].name = "latlonint_of_%s_over_%s" % (self.variable,region)

        # Dump to files
        def _write(out_vars,results):
            for var in out_vars:
                if type(var) == type({}):
                    for key in var.keys(): var[key].toNetCDF4(results,group="MeanState")
                else:
                    var.toNetCDF4(results,group="MeanState")

        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        _write([m1,d1,s1,sb,mp,mb,m2,m3,sp,m4],results)
        results.close()
        if self.master:
            results = Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w")
            results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
            _write([o1,op,o2,o3,o4],results)
            results.close()

    def compositePlots(self):

        if not self.master: return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        models = []
        colors = []
        f1     = {}
        a1     = {}
        u1     = None
        for fname in glob.glob("%s/*.nc" % self.output_path):
            with Dataset(fname) as dset:
                models.append(dset.getncattr("name"))
                colors.append(dset.getncattr("color"))
                if "MeanState" not in dset.groups: continue
                group     = dset.groups["MeanState"]
                variables = getVariableList(group)
                for region in self.regions:

                    vname = "profile_of_%s_over_%s" % (self.variable,region)
                    if vname in variables:
                        if not f1.has_key(region):
                            f1[region],a1[region] = plt.subplots(figsize=(5,5),tight_layout=True)
                        var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                        u1  = var.unit
                        page.addFigure("Mean regional depth profiles",
                                       "profile",
                                       "RNAME_profile.png",
                                       side   = "REGIONAL MEAN PROFILE",
                                       legend = False)
                        a1[region].plot(var.data,var.depth,'-',
                                        color = dset.getncattr("color"))
        for key in f1.keys():
            a1[key].set_xlabel("%s [%s]" % (self.variable,u1))
            a1[key].set_ylabel("depth [m]")
            a1[key].invert_yaxis()
            f1[key].savefig("%s/%s_profile.png" % (self.output_path,key))
        plt.close()

    def modelPlots(self,m):

        bname  = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname  = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        with Dataset(fname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")

            vname = "timeint_surface_%s" % self.variable
            if vname in variables:
                var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                page.addFigure("Period mean at surface",
                               "timeint",
                               "MNAME_RNAME_timeint.png",
                               side   = "MODEL SURFACE MEAN",
                               legend = True)
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,
                             region = region,
                             vmin   = self.limits["timeint"]["min"],
                             vmax   = self.limits["timeint"]["max"],
                             cmap   = self.cmap)
                    fig.savefig("%s/%s_%s_timeint.png" % (self.output_path,m.name,region))
                    plt.close()

            vname = "bias_surface_%s" % self.variable
            if vname in variables:
                var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                page.addFigure("Period mean at surface",
                               "bias",
                               "MNAME_RNAME_bias.png",
                               side   = "SURFACE MEAN BIAS",
                               legend = True)
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,
                             region = region,
                             vmin   = self.limits["bias"]["min"],
                             vmax   = self.limits["bias"]["max"],
                             cmap   = "seismic")
                    fig.savefig("%s/%s_%s_bias.png" % (self.output_path,m.name,region))
                    plt.close()

            for region in self.regions:

                vname = "timelonint_of_%s_over_%s" % (self.variable,region)
                if vname in variables:
                    var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                    if region == "global":
                        page.addFigure("Mean regional depth profiles",
                                       "timelonint",
                                       "MNAME_RNAME_timelonint.png",
                                       side   = "MODEL DEPTH PROFILE",
                                       legend = True)
                    fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                    l   = np.hstack([var.lat_bnds  [:,0],var.lat_bnds  [-1,1]])
                    d   = np.hstack([var.depth_bnds[:,0],var.depth_bnds[-1,1]])
                    ind = np.all(var.data.mask,axis=0)
                    ind = np.ma.masked_array(range(ind.size),mask=ind,dtype=int)
                    b   = ind.min()
                    e   = ind.max()+1
                    ax.pcolormesh(l[b:(e+1)],d,var.data[:,b:e],
                                  cmap = self.cmap)
                    ax.set_xlabel("latitude")
                    ax.set_ylim((d.max(),d.min()))
                    ax.set_ylabel("depth [m]")
                    fig.savefig("%s/%s_%s_timelonint.png" % (self.output_path,m.name,region))
                    plt.close()

        if not self.master: return

        with Dataset(bname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")

            vname = "timeint_surface_%s" % self.variable
            if vname in variables:
                var = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                page.addFigure("Period mean at surface",
                               "benchmark_timeint",
                               "Benchmark_RNAME_timeint.png",
                               side   = "BENCHMARK SURFACE MEAN",
                               legend = True)
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,
                             region = region,
                             vmin   = self.limits["timeint"]["min"],
                             vmax   = self.limits["timeint"]["max"],
                             cmap   = self.cmap)
                    fig.savefig("%s/Benchmark_%s_timeint.png" % (self.output_path,region))
                    plt.close()

            for region in self.regions:

                vname = "timelonint_of_%s_over_%s" % (self.variable,region)
                if vname in variables:
                    var = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                    if region == "global":
                        page.addFigure("Mean regional depth profiles",
                                       "benchmark_timelonint",
                                       "Benchmark_RNAME_timelonint.png",
                                       side   = "BENCHMARK DEPTH PROFILE",
                                       legend = True)
                    fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                    l   = np.hstack([var.lat_bnds  [:,0],var.lat_bnds  [-1,1]])
                    d   = np.hstack([var.depth_bnds[:,0],var.depth_bnds[-1,1]])
                    ind = np.all(var.data.mask,axis=0)
                    ind = np.ma.masked_array(range(ind.size),mask=ind,dtype=int)
                    b   = ind.min()
                    e   = ind.max()+1
                    ax.pcolormesh(l[b:(e+1)],d,var.data[:,b:e],
                                  cmap = self.cmap)#,
                    #vmin = self.limits["timeint"]["min"],
                    #vmax = self.limits["timeint"]["max"])
                    ax.set_xlabel("latitude")
                    ax.set_ylim((d.max(),d.min()))
                    ax.set_ylabel("depth [m]")
                    fig.savefig("%s/Benchmark_%s_timelonint.png" % (self.output_path,region))
                    plt.close()
