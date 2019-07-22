from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable
import matplotlib.pyplot as plt
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

class ConfSoilCarbon(Confrontation):
    """
    """
    def stageData(self,m):

        # Constants
        small_npp         =  1e-4  # kg m-2 yr-1
        aridity_threshold = -1000  # mm yr-1
        sat_threshold     =  0.5   # 1
        gauss_critval     =  0.674 # for 50%
        
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
        mask += (soilc < 1e-12)  # where there is no soilc
        mask += (npp < small_npp)  # where npp is small or negative
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

        # Plot for checking
        fig,ax = plt.subplots(figsize=(9,8),nrows=2,ncols=2,tight_layout=True)
        with np.errstate(under='ignore'):
            ax[0,0].scatter(tas,soilc,color='k',s=0.1,alpha=1)
            ax[0,1].scatter(tas,npp  ,color='k',s=0.1,alpha=1)
            ax[1,0].scatter(tas,tau  ,c    = pr,s=0.1,alpha=1,vmin=0,vmax=2000,cmap='wetdry')
            ax[1,1].scatter(tas,tau  ,color='k',s=0.1,alpha=1)
            ax[1,1].semilogy(X,Y,'-r',lw=2)
            ax[1,1].semilogy(X,10**(np.log10(Y)-gauss_critval*std),'-r',lw=1)
            ax[1,1].semilogy(X,10**(np.log10(Y)+gauss_critval*std),'-r',lw=1)
            for i in range(2):
                for j in range(2):
                    ax[i,j].set_yscale('log')
                    ax[i,j].set_xlim(-22,30)
                    ax[i,j].set_xlabel("Mean air temperature [$^{\circ}$C]")
        ax[0,0].set_ylim(0.1,300)
        ax[0,1].set_ylim(1e-3,3)
        ax[1,0].set_ylim(1,3e3)
        ax[1,1].set_ylim(1,3e3)
        ax[0,0].set_ylabel("NCSCD/HWSD Soil carbon to 1 m [kg m$^{-2}$]")
        ax[0,1].set_ylabel("MODIS npp [kg m$^{-2}$ yr$^{-1}$]")
        ax[1,0].set_ylabel("Inferred turnover time [yr]")
        ax[1,1].set_ylabel("Inferred turnover time [yr]")
        plt.savefig("fig1.pdf")
        plt.close()

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

        return None,None
    
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
    obs,mod = c.stageData(m)







    

    
