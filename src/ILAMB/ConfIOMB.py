from ILAMB.Confrontation import Confrontation
from ILAMB.Confrontation import getVariableList
from ILAMB.constants import earth_rad,mid_months,lbl_months
from ILAMB.Variable import Variable
from ILAMB.Regions import Regions
import ILAMB.Post as post
from netCDF4 import Dataset
from copy import deepcopy
import pylab as plt
import numpy as np
import os

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
        pages[-1].setSections(["Mean at surface and over time period",
                               "Mean regional depth profiles"])            
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
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
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars     = self.alternate_vars,
                                  expression   = self.derived,
                                  initial_time = obs.time_bnds[ 0,0],
                                  final_time   = obs.time_bnds[-1,1],
                                  lats         = None if obs.spatial else obs.lat,
                                  lons         = None if obs.spatial else obs.lon).convert(obs.unit)
        # push into MakeComparable
        mod.trim(d=[obs.depth_bnds.min(),
                    obs.depth_bnds.max()]) 
        return obs,mod
    
    def confront(self,m):
        """Sections

        1) They look at surface states (mean maps over time) for some
        varables. We have a few of these (evap, shf) and could obtain the datasets
        they use for others.

        2) They also look at Global zonal average. (spaceint) looking
        at surface. we would include seasonal cycle here.
        
        They look in the depth dimension also (mean over time and
        lon), they do this over different regions (Atlantic, Pacficic,
        Indian, Southern, Arctic) maybe defined by lat/lon bounds. A
        plot of var vs depth/lat as well as vs depth. 

        4) (optional) look at states like in (1) but at preset
        depths. Maybe this is just part of 1 where by default there is
        only the surface depth. Do we include another pulldown for
        that which picks the depth? 0 50 100 200 300 500 750 1000 1500
        20000 2500 3000 4000 (also over regions) maybe use standard
        depths instead from WOA.

        Action items
        ------------

        * identify / obtain the obs datasets used in the CESM diagnostic ocean package.
        * Get ocean mask dataset (find or invent)

        """        
        # Looking at a representative year 
        y0 = 2000.
        yf = 2001.
        
        # get the data
        obs,mod = self.stageData(m)
        
        # Reduction 1: Surface states
        ds   = [   0.,  10.]
        ts   = [(y0-1850.)*365.,
                (yf-1850.)*365.]
        o1   = VariableReduce(obs,time=ts,depth=ds)
        m1   = VariableReduce(mod,time=ts,depth=ds)
        d1   = o1.bias(m1)
        m1.name = "timeint_surface_%s" % self.variable
        d1.name = "bias_surface_%s"    % self.variable
        o1.name = "timeint_surface_%s" % self.variable
        
        o2 = {}; m2 = {}; o3 = {}; m3 = {}; o4 = {}; m4 = {}
        op = {}; mp = {}
        for region in self.regions:

            op[region] = o1.integrateInSpace(mean=True,region=region)
            mp[region] = m1.integrateInSpace(mean=True,region=region)
            op[region].name = "Period Mean %s" % region
            mp[region].name = "Period Mean %s" % region
            
            # Reduction 2/3: Zonal depth profiles
            o2[region] = VariableReduce(obs,region,time=ts,lon=[-180.,180.])
            m2[region] = VariableReduce(mod,region,time=ts,lon=[-180.,180.])
            o3[region] = obs.integrateInSpace(region=region,mean=True).integrateInTime(t0=ts[0],tf=ts[1],mean=True)
            m3[region] = mod.integrateInSpace(region=region,mean=True).integrateInTime(t0=ts[0],tf=ts[1],mean=True)
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
        _write([m1,d1,mp,m2,m3,m4],results)
        results.close()
        if self.master:
            results = Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w")
            results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
            _write([o1,op,o2,o3,o4],results)
            results.close()

    def compositePlots(self):
        pass
        
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

            print variables
            
            vname = "timeint_surface_%s" % self.variable 
            if vname in variables:
                var = Variable(filename=fname,variable_name=vname,groupname="MeanState")
                page.addFigure("Mean at surface and over time period",
                               "MNAME_RNAME_timeint",
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
                page.addFigure("Mean at surface and over time period",
                               "MNAME_RNAME_bias",
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

        if not self.master: return
        
        with Dataset(bname) as dataset:
            group     = dataset.groups["MeanState"]
            variables = getVariableList(group)
            color     = dataset.getncattr("color")
            
            vname = "timeint_surface_%s" % self.variable 
            if vname in variables:
                var = Variable(filename=bname,variable_name=vname,groupname="MeanState")
                page.addFigure("Mean at surface and over time period",
                               "Benchmark_RNAME_timeint",
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
            




                    
            """
            
            vname = "%s_vs_time_and_depth" % self.variable 
            if vname in variables:
                page.addFigure("Spatially integrated regional mean",
                               "MNAME_RNAME_r1",
                               "MNAME_RNAME_r1.png",
                               side   = "MODEL MEAN OVER LAT=[30,90], LON=[-180,180]",
                               legend = False)       
                fig,ax = plt.subplots(figsize=(6.0,6.0),tight_layout=True)
                v = Variable(filename = fname, variable_name = vname, groupname = "MeanState")
                t = np.hstack([v.time_bnds [:,0],v.time_bnds [-1,1]])-150.*365.
                d = np.hstack([v.depth_bnds[:,0],v.depth_bnds[-1,1]])
                ax.pcolormesh(t,d,v.data.T,cmap=self.cmap,vmin=0,vmax=35.)
                ax.set_ylim((d.max(),d.min()))
                ax.set_ylabel("depth [m]")
                ax.set_xticks(mid_months)
                ax.set_xticklabels(lbl_months)
                fig.savefig("%s/%s_global_r1.png" % (self.output_path,m.name))            

            """
            
if __name__ == "__main__":
    from ILAMB.ModelResult import ModelResult
    m    = ModelResult("./Fake")    
    opts = {"source"         : "./obs.nc",
            "variable"       : "Nitrate",
            "alternate_vars" : ["NO3"],
            "output_path"    : "./junk"}
    c    = ConfIOMB(**opts)
    c.confront(m)




    """
    
    ######
    #
    # We could abandon the use of classic confrontation and then do
    #

    #x5 = mod.reduce(depth = [0,10],
    #                time  = [t0,tf])
    
    #x6 = mod.reduce(depth = [100,110],
    #                time  = [t0,tf])
    
    #####
    #
    # We may want to reduce a variable across isodensities, masks
    # the variable if it is close to the densities.
    #
    #x7 = _reduceIsopycnal(m,mod,density)
    
    #####
    #
    # From Clara
    
    # Depth of Max Chl in a region in a season (Aug-Sept)
    #
    # reduce lat = [narrow band across canyon], trim away time = [ single month] and lon = [162,154]
    #
    # reduce by ship track
    #
    
    """
    
