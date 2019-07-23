from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

def getSource(filename,unit):
    vname = filename.split("/")[-1].split("_")[0]
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

def sensitivityPlot(tas,tau,pr,X,Y,std,gauss_critval,filename):
    fig,ax = plt.subplots(figsize=(5,5.5),tight_layout=True,dpi=100)
    sc = ax.scatter(tas,tau,c=pr,s=0.1,alpha=1,vmin=0,vmax=2000,cmap='wetdry')
    ax.semilogy(X,Y,'-k',lw=2)
    ax.semilogy(X,10**(np.log10(Y)-gauss_critval*std),'-k',lw=1)
    ax.semilogy(X,10**(np.log10(Y)+gauss_critval*std),'-k',lw=1)
    ax.set_yscale('log')
    ax.set_xlim(-22,30)
    ax.set_ylim(1,3e3)
    ax.set_xlabel("Mean air temperature [$^{\circ}$C]")
    ax.set_ylabel("Inferred turnover time [yr]")
    fig.colorbar(sc,orientation='horizontal',pad=0.15,label='Precipitation [mm yr$^{-1}$]')
    plt.savefig(filename)
    plt.close()
    
class ConfSoilCarbon(Confrontation):
    """
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
        tas   = np.ma.masked_array(tas  ,mask=mask).compressed()
        pr    = np.ma.masked_array(pr   ,mask=mask).compressed()

        # Compute inferred turnover and fit quadratic
        tau = soilc/npp
        p   = np.polyfit(tas,np.log10(tau),2)
        std = np.sqrt(((np.log10(tau) - np.polyval(p,tas))**2).sum()/tas.size)
        X   = np.linspace(-22,30,100)
        Y   = 10**np.polyval(p,X)
        
        # Get model results
        y0 = self.keywords.get("y0",1980.)
        yf = self.keywords.get("yf",2006.)
        t0 = (y0-1850  )*365
        tf = (yf-1850+1)*365
        mod_soilc = m.extractTimeSeries("soilc",
                                        alt_vars     = ["cSoil"],
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("kg m-2")
        mod_npp   = m.extractTimeSeries("npp",
                                        initial_time = t0,
                                        final_time   = tf,
                                        expression   = "gpp-ra-rh").integrateInTime(mean=True).convert("kg m-2 yr-1")
        mod_tas   = m.extractTimeSeries("tas",
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("degC")
        mod_pr    = m.extractTimeSeries("pr",
                                        initial_time = t0,
                                        final_time   = tf).integrateInTime(mean=True).convert("mm yr-1")

        # Determine what will be masked
        mask  = mod_soilc.data.mask + mod_npp.data.mask + mod_tas.data.mask + mod_pr.data.mask
        mask += (mod_soilc.data < soilc_threshold)
        mask += (mod_npp.data < npp_threshold)
        mod_soilc = np.ma.masked_array(mod_soilc.data,mask=mask).compressed()
        mod_npp   = np.ma.masked_array(mod_npp.data  ,mask=mask).compressed()
        mod_tas   = np.ma.masked_array(mod_tas.data  ,mask=mask).compressed()
        mod_pr    = np.ma.masked_array(mod_pr.data   ,mask=mask).compressed()

        # Compute inferred turnover and fit quadratic
        mod_tau = mod_soilc/mod_npp
        mod_p   = np.polyfit(mod_tas,np.log10(mod_tau),2)
        mod_std = np.sqrt(((np.log10(mod_tau) - np.polyval(mod_p,mod_tas))**2).sum()/mod_tas.size)
        mod_X   = np.linspace(-22,30,100)
        mod_Y   = 10**np.polyval(p,X)

        # Binned Relationship RMSE
        mean_obs = np.ma.masked_array(np.zeros(relationship_bins.size-1),mask=True)
        mean_mod = np.ma.masked_array(np.zeros(relationship_bins.size-1),mask=True)
        bins = np.digitize(mod_tas,relationship_bins).clip(1,relationship_bins.size-1)-1
        rmse = 0.
        for i in np.unique(bins):
            tmid = 0.5*(relationship_bins[i]+relationship_bins[i+1])
            mean_obs[i] = np.polyval(p,tmid)
            mean_mod[i] = np.log10(mod_tau[bins==i]).mean()
            rmse += mean_obs[i]**2 + mean_mod[i]**2
        rmse = np.sqrt(rmse)
        
        # Outputs and plots
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        
        if self.master:
            page.addFigure("Temporally integrated period mean",
                           "benchmark_timeint",
                           "Benchmark_global_timeint.png",
                           side   = "BENCHMARK",
                           legend = False)
            sensitivityPlot(tas,tau,pr,X,Y,std,gauss_critval,"%s/Benchmark_global_timeint.png" % (self.output_path))
            with Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w") as results:
                results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
                Variable(name = "T^2",unit="1",data=p[0]).toNetCDF4(results,group="MeanState")
                Variable(name = "T"  ,unit="1",data=p[1]).toNetCDF4(results,group="MeanState")
                Variable(name = "1"  ,unit="1",data=p[2]).toNetCDF4(results,group="MeanState")
                
        page.addFigure("Temporally integrated period mean",
                       "timeint",
                       "MNAME_global_timeint.png",
                       side   = "MODEL",
                       legend = False)
        sensitivityPlot(mod_tas,mod_tau,mod_pr,mod_X,mod_Y,mod_std,gauss_critval,
                        "%s/%s_global_timeint.png" % (self.output_path,m.name))
        with Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w") as results:
            results.setncatts({"name" :m.name, "color":m.color})
            Variable(name = "T^2" ,unit="1",data=mod_p[0]).toNetCDF4(results,group="MeanState")
            Variable(name = "T"   ,unit="1",data=mod_p[1]).toNetCDF4(results,group="MeanState")
            Variable(name = "1"   ,unit="1",data=mod_p[2]).toNetCDF4(results,group="MeanState")
            Variable(name = "RMSE",unit="1",data=rmse    ).toNetCDF4(results,group="MeanState")

if __name__ == "__main__":
    from ILAMB.ModelResult import ModelResult
    from ILAMB.Post import RegisterCustomColormaps
    RegisterCustomColormaps()
    m = ModelResult("/home/nate/data/ILAMB/MODELS/esmHistorical/CESM1-BGC/")
    c = ConfSoilCarbon(source = "/home/nate/data/ILAMB/DATA/gpp/GBAF/gpp_0.5x0.5.nc",
                       soilc_source = "DATA/soilc/NCSCDV22/soilc_0.5x0.5.nc, DATA/soilc/HWSD/soilc_0.5x0.5.nc",
                       tas_source = "DATA/tas/CRU/tas_0.5x0.5.nc",
                       pr_source = "DATA/pr/GPCC/pr_0.5x0.5.nc",
                       npp_source = "DATA/soilc/Koven/npp_0.5x0.5.nc",
                       pet_source = "DATA/soilc/Koven/pet_0.5x0.5.nc",
                       fracpeat_source = "DATA/soilc/Koven/fracpeat_0.5x0.5.nc")
    c.confront(m)
