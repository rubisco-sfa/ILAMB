from .Confrontation import Confrontation
from .Variable import Variable
from .Relationship import Relationship
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

def getSource(filename,unit):
    vname = filename.split("/")[-1].split("_")[0]
    if vname == "soilc": vname = "cSoilAbove1m"
    files = filename.split(",") if "," in filename else [filename]
    data,lat,lon = None,None,None
    for f in files:
        v = Variable(filename = os.path.join(os.environ["ILAMB_ROOT"],f.strip()),
                     variable_name = vname)
        if v.temporal: v = v.integrateInTime(mean=True)
        v.convert(unit)
        if data is None:
            data = v.data
            lat  = v.lat
            lon  = v.lon
        else:
            assert np.allclose(v.lat,lat)*np.allclose(v.lon,lon)
            ind = np.where(data.mask + (data < 1e-12*data.max()))
            data[ind] = v.data[ind]
    return lat,lon,data
    
class ConfSoilCarbon(Confrontation):
    """A soil carbon temperature sensivity benchmark.

    For details of the metric, see the following publication:

    Koven, Hugelius, Lawrence, Wieder, Higher climatological
    temperature sensitivity of soil carbon in cold than warm
    climates. Nature Climate Change, October 2017, doi:
    10.1038/NCLIMATE3421

    """
    def __init__(self,**keywords):
        super(ConfSoilCarbon,self).__init__(**keywords)
        self.regions        = ["global"]
        self.layout.regions = self.regions
    
    def stageData(self,m):
        pass
    
    def confront(self,m):

        # Constants
        soilc_threshold   =  1e-12 # kg m-2
        npp_threshold     =  1e-4  # kg m-2 yr-1
        aridity_threshold = -1000  # mm yr-1
        sat_threshold     =  0.5   # 1
        gauss_critval     =  0.674 # for 50%
        relationship_bins = self.keywords.get("relationship_bins",np.arange(-15.5,28.6,1))
        if type(relationship_bins) == str:
            relationship_bins = np.asarray(relationship_bins.split(","),dtype=float)
        T10 = np.asarray([-10,+10,25]) # temperatures at which to report Q10
        
        # Get the source datafiles
        lat,lon,soilc = getSource(self.keywords.get("soilc_source"),"kg m-2")
        LAT,LON,npp = getSource(self.keywords.get("npp_source"),"kg m-2 yr-1")
        assert np.allclose(LAT,lat)*np.allclose(LON,lon)
        LAT,LON,tas = getSource(self.keywords.get("tas_source"),"degC")
        assert np.allclose(LAT,lat)*np.allclose(LON,lon)
        LAT,LON,pr = getSource(self.keywords.get("pr_source"),"mm yr-1")
        assert np.allclose(LAT,lat)*np.allclose(LON,lon)
        LAT,LON,pet = getSource(self.keywords.get("pet_source"),"mm yr-1")
        assert np.allclose(LAT,lat)*np.allclose(LON,lon)
        LAT,LON,fracpeat = getSource(self.keywords.get("fracpeat_source"),"1")
        assert np.allclose(LAT,lat)*np.allclose(LON,lon)

        # Determine what will be masked
        mask  = soilc.mask + tas.mask + npp.mask + pr.mask  # where any source is masked
        mask += (soilc < soilc_threshold)  # where there is no soilc
        mask += (npp < npp_threshold)  # where npp is small or negative
        mask += ((pr-pet) < aridity_threshold)  # where aridity dominates
        mask += (fracpeat > sat_threshold)  # where mostly peatland
        soilc = np.ma.masked_array(soilc,mask=mask).compressed()
        npp   = np.ma.masked_array(npp  ,mask=mask).compressed()
        tas   = Variable(name="Mean air temperature",unit="degC",data=np.ma.masked_array(tas,mask=mask).compressed())
        pr    = Variable(name="Precipitation",unit="mm yr-1",data=np.ma.masked_array(pr,mask=mask).compressed())
        tau   = Variable(name="Inferred turnover time",unit="yr",data=soilc/npp)
        r     = Relationship(tas,tau,dep_log=True,order=2,color=pr)
        r.limits = [[1.,1e3],[-22.,30.]]
        
        # Get model results
        y0 = self.keywords.get("y0",1980.)
        yf = self.keywords.get("yf",2006.)
        t0 = (y0-1850  )*365
        tf = (yf-1850+1)*365
        mod_soilc = m.extractTimeSeries("cSoilAbove1m",
                                        alt_vars     = ["soilc","cSoil"],
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("kg m-2")
        mod_npp   = m.extractTimeSeries("npp",
                                        initial_time = t0,
                                        final_time   = tf,
                                        expression   = "gpp-ra").integrateInTime(mean=True).convert("kg m-2 yr-1")
        mod_tas   = m.extractTimeSeries("tas",
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("degC")
        mod_pr    = m.extractTimeSeries("pr",
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("mm yr-1")
        mod_pet   = Variable(lat=LAT,lon=LON,unit="mm yr-1",data=pet).interpolate(lat=mod_pr.lat,lon=mod_pr.lon)
        
        # Determine what will be masked
        mask  = mod_soilc.data.mask + mod_npp.data.mask + mod_tas.data.mask + mod_pr.data.mask
        mask += (mod_soilc.data < soilc_threshold)
        mask += (mod_npp.data < npp_threshold)
        mask += ((mod_pr.data-mod_pet.data) < aridity_threshold)
        mod_soilc = np.ma.masked_array(mod_soilc.data,mask=mask).compressed()
        mod_npp   = np.ma.masked_array(mod_npp.data  ,mask=mask).compressed()
        mod_tas   = Variable(name="Mean air temperature",unit="degC",data=np.ma.masked_array(mod_tas.data,mask=mask).compressed())
        mod_pr    = Variable(name="Precipitation",unit="mm yr-1",data=np.ma.masked_array(mod_pr.data,mask=mask).compressed())

        # Compute inferred turnover and fit quadratic
        mod_tau = Variable(name="Inferred turnover time",unit="yr",data=mod_soilc/mod_npp)
        mod_r   = Relationship(mod_tas,mod_tau,dep_log=True,order=2,color=mod_pr)
        mod_r.limits = r.limits
        
        # Outputs and plots
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]        
        if self.master:
            page.addFigure("Temporally integrated period mean",
                           "benchmark_timeint",
                           "Benchmark_global_timeint.png",
                           side   = "BENCHMARK",
                           legend = False)
            
            fig,ax = plt.subplots(figsize=(5,5.5),tight_layout=True,dpi=100)
            r.plotPointCloud(ax,vmin=0,vmax=2000,cmap='wetdry')
            r.plotModel(ax,color='k',prediction=True)
            ax.set_xlim(-22,30)
            ax.set_ylim(1,3e3)
            plt.savefig("%s/Benchmark_global_timeint.png" % (self.output_path))
            plt.close()
    
            with Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w") as results:
                results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5]),"complete":0})
                p = r.dist["default"][5]
                Q10 = 10**(-10*(np.polyval(np.polyder(p),T10)))
                for q,t in zip(Q10,T10):
                    Variable(name = "Q10(%+d [C])" % int(t),unit="1",data=q).toNetCDF4(results,group="MeanState")
                results.setncattr("complete",1)

        page.addFigure("Temporally integrated period mean",
                       "timeint",
                       "MNAME_global_timeint.png",
                       side   = "MODEL",
                       legend = False)
        fig,ax = plt.subplots(figsize=(5,5.5),tight_layout=True,dpi=100)
        mod_r.plotPointCloud(ax,vmin=0,vmax=2000,cmap='wetdry')
        r    .plotModel(ax,color='k',prediction=True)
        ax.set_xlim(-22,30)
        ax.set_ylim(1,3e3)
        plt.savefig("%s/%s_global_timeint.png" % (self.output_path,m.name))
        plt.close()
        
        page.addFigure("Temporally integrated period mean",
                       "rel_tas",
                       "MNAME_RNAME_rel_tas.png",
                       side   = "MODEL",
                       legend = False)
        fig,ax = plt.subplots(figsize=(5,4.5),tight_layout=True,dpi=100)
        r    .plotFunction(ax,color='k'    ,shift=-0.1)
        mod_r.plotFunction(ax,color=m.color,shift=+0.1)
        ax.set_xlim(-22,30)
        ax.set_ylim(1,3e3)
        plt.savefig("%s/%s_global_rel_tas.png" % (self.output_path,m.name))
        plt.close()
        
        with Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w") as results:
            results.setncatts({"name" :m.name, "color":m.color,"complete":0})
            mod_p = mod_r.dist["default"][5]
            Q10 = 10**(-10*(np.polyval(np.polyder(mod_p),T10)))
            for q,t in zip(Q10,T10):
                Variable(name = "Q10(%+d [C])" % int(t),unit="1",data=q).toNetCDF4(results,group="MeanState")
            Variable(name = "RMSE Score global",unit="1",data=r.scoreRMSE(mod_r)).toNetCDF4(results,group="MeanState")
            Variable(name = "Distribution Score global",unit="1",data=r.scoreHellinger(mod_r)).toNetCDF4(results,group="MeanState")
            results.setncattr("complete",1)
        
    def determinePlotLimits(self):
        pass
    
    def compositePlots(self):
        pass
    
    def modelPlots(self,m):
        
        # Outputs and plots
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]        
        if self.master:
            page.addFigure("Temporally integrated period mean",
                           "benchmark_timeint",
                           "Benchmark_global_timeint.png",
                           side   = "BENCHMARK",
                           legend = False)
        page.addFigure("Temporally integrated period mean",
                       "timeint",
                       "MNAME_global_timeint.png",
                       side   = "MODEL",
                       legend = False)        
        page.addFigure("Temporally integrated period mean",
                       "rel_tas",
                       "MNAME_RNAME_rel_tas.png",
                       side   = "MODEL",
                       legend = False)
