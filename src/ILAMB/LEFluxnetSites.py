from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from Variable import Variable
from netCDF4 import Dataset
import pylab as plt
import numpy as np
import os

class LEFluxnetSites():
    """Confront models with the latent heat (LE) product from Fluxnet.
    """
    def __init__(self):
        self.name = "LEFluxnetSites"
        self.path = os.environ["ILAMB_ROOT"] + "/DATA/le/FLUXNET/derived/"
        try:
            os.stat(self.path)
        except:
            msg  = "I am looking for data for the %s confrontation here\n\n" % self.name
            msg += "%s\n\nbut I cannot find it. " % self.path
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?"
            raise il.MisplacedData(msg)
        self.data = {}
        self.regions = []

        # build output path if not already built
        self.output_path = "_build/%s" % self.name
        dirs = self.output_path.split("/")
        for i,d in enumerate(dirs):
            dname = "/".join(dirs[:(i+1)])
            if not os.path.isdir(dname): os.mkdir(dname)

        self.weights = {"RMSEScore"                  :2.,
                        "BiasScore"                  :1.,
                        "SeasonalCycleScore"         :1.,
                        "SpatialDistributionScore"   :1.,
                        "InterannualVariabilityScore":1.}

    def getData(self,output_unit=None):
        """Retrieves the confrontation data in the desired unit.

        Parameters
        ----------
        output_unit : string, optional
            if specified, will try to convert the units of the variable
            extract to these units given (see convert in ILAMB.constants)

        Returns
        -------
        var : ILAMB.Variable.Variable
            the requested variable
        """
        f = Dataset("%s/le.nc" % self.path,mode="r")
        self.siteClass = f.getncattr("IGBP_class").split(',')
        self.siteName  = f.getncattr("site_name").split(',')
        lat  = f.variables['lat'][...]
        lon  = f.variables['lon'][...]
        t    = f.variables['time'][...]
        v    = f.variables['le']
        var  = v[...]
        unit = v.units
        f.close()
        if output_unit is not None:
            try:
                var *= convert["le"][output_unit][unit]
                unit = output_unit
            except:
                msg  = "The le variable is in units of [%s]. " % unit
                msg += "You asked for units of [%s] but I do not know how to convert" % output_unit
                raise il.UnknownUnit(msg)
        v = Variable(var,unit,time=t,lat=lat,lon=lon,name="le")
        v.spatial = False # individual site data does not constitute areas

        return v

    def confront(self,m):
        
        # write confrontation result file
        f = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")

        # some data we will store to send to plot routine
        cdata = {}
        scores = {}

        # get the observational data
        if self.data.has_key("le"):
            obs_le = self.data["le"]
        else:
            obs_le = self.getData()
            obs_le.unit = "J m-2 s-1"
            obs_le.name = "hfls"
            self.data["le"] = obs_le

        # time limits for this confrontation (with a little padding)
        t0,tf = obs_le.time.min(),obs_le.time.max()
        ndays = tf-t0

        # get the model data
        mod_le = m.extractTimeSeries("hfls",
                                     lats=obs_le.lat,lons=obs_le.lon,
                                     initial_time=t0,final_time=tf,
                                     output_unit="J m-2 s-1")

        clr = ['r','g','b']
        for i in range(3):
            plt.plot(obs_le.time,obs_le.data[:,i],'-',label="(%.2f %.2f)" % (obs_le.lat[i],obs_le.lon[i]),color=clr[i])
        #plt.plot(mod_le.time,mod_le.data[:,4],'-',label="(%.2f %.2f)" % (mod_le.lat[4],mod_le.lon[4]))
        #plt.legend()
        
        
        cdata["le"] = mod_le

        # integrate over the time period
        obs_le_timeint = obs_le.integrateInTime(mean=True).convert("J m-2 s-1")
        print obs_le_timeint.data
        
        for i in range(3):
            plt.plot(55500,obs_le_timeint.data[i],'o',color=clr[i])
        
        
        plt.savefig("%s.png" % m.name)
        plt.close()

        mod_le_timeint = mod_le.integrateInTime(mean=True).convert("J m-2 s-1")
        obs_le_timeint.unit = "W m-2"
        self.data["timeint_le"] = obs_le_timeint
        mod_le_timeint.unit = "W m-2"
        mod_le_timeint.toNetCDF4(f)
        bias = obs_le_timeint.bias(mod_le_timeint)
        bias.name = "bias_of_le_integrated_over_time_and_divided_by_time_period"
        bias.toNetCDF4(f)
        bias = obs_le_timeint.bias(mod_le_timeint,normalize="score")
        bias.name = "bias_score_of_le_integrated_over_time_and_divided_by_time_period"
        bias.toNetCDF4(f)
        scores["BiasScore"] = bias.data
        rmse = obs_le_timeint.RMSE(mod_le_timeint)
        rmse.name = "rmse_of_le_integrated_over_time_and_divided_by_time_period"
        rmse.toNetCDF4(f)
        rmse = obs_le_timeint.RMSE(mod_le_timeint,normalize="score")
        rmse.name = "rmse_score_of_le_integrated_over_time_and_divided_by_time_period"
        rmse.toNetCDF4(f)
        scores["RMSEScore"] = rmse.data

        # observations phase
        if self.data.has_key("phase"):
            obs_phase = self.data["phase"]
        else:
            obs_phase = obs_le.phase()
            self.data["phase"] = obs_phase

        # model phase
        mod_phase = mod_le.phase()
        cdata["phase"] = mod_phase
        mod_phase.toNetCDF4(f)

        # shift
        shift  = mod_phase.data-obs_phase.data
        shift -= (shift > +0.5*365.)*365.
        shift += (shift < -0.5*365.)*365.
        shift  = Variable(shift,"d",name="phase_shift_of_hfls",
                         lat=mod_le.lat,lon=mod_le.lon)
        shift.toNetCDF4(f)
        cdata["shift"] = shift
        Variable(np.ma.masked_array(shift.data.mean()),"d",name="mean_phase_shift").toNetCDF4(f)

        # cycle score
        cycle = ((1-np.cos(np.abs(shift.data)/365*2*np.pi))*0.5).mean()
        Variable(np.ma.masked_array(cycle),"-",
                 name = "seasonal_cycle_score").toNetCDF4(f)
        scores["SeasonalCycleScore"] = cycle

        # spatial variation
        cor,std = obs_le.corrcoef(mod_le)
        sds = 2.*(1.+cor)/((std+1./std)**2)
        Variable(np.ma.masked_array(cor),"-",name="correlation").toNetCDF4(f)
        Variable(np.ma.masked_array(std),"-",name="std").toNetCDF4(f)
        Variable(np.ma.masked_array(sds),"-",name="spatial_distribution_score").toNetCDF4(f)
        scores["SpatialDistributionScore"] = sds

        # interannual variability
        iav  = np.exp(1.-np.abs(std-1))/np.exp(1)
        Variable(np.ma.masked_array(iav),"-",name="interannual_variability_score").toNetCDF4(f)
        scores["InterannualVariabilityScore"] = iav

        # overall score
        score = 0.
        sumw  = 0.
        for key in self.weights.keys():
            score += self.weights[key]*scores[key]
            sumw  += self.weights[key]
        score = np.ma.masked_array(score/sumw)
        Variable(score,"-",name="overall_score").toNetCDF4(f)

        self.plot(m,cdata)

    def plotFromFiles(self):
        pass

    def plot(self,m=None,data=None):
        
        # Correlation plot
        fig,ax = plt.subplots(figsize=(6.8,6.8),tight_layout=True)
        obs  = self.data["le"]
        mod  =      data["le"]
        absl = abs(obs.lat)
        hot  =  absl< 23.25
        cold =  absl> 66.50
        mid  = (absl>=23.25)*(absl<=66.50)        
        ax.scatter(obs.data[:,mid ].flatten(),mod.data[:,mid ].flatten(),color='g',alpha=0.75,label="Mid latitude")
        ax.scatter(obs.data[:,cold].flatten(),mod.data[:,cold].flatten(),color='b',alpha=0.75,label="High latitude")
        ax.scatter(obs.data[:,hot ].flatten(),mod.data[:,hot ].flatten(),color='r',alpha=0.75,label="Tropical latitude")
        ax.set_xlabel("Observation Latent Heat Flux [W/m$^2$]")
        ax.set_ylabel("Model Latent Heat Flux [W/m$^2$]")
        ax.legend(loc=4,scatterpoints=1)
        fig.savefig("%s/correlation_%s.png" % (self.output_path,m.name))
        plt.close()
        
        # Basemap plot of mean observational data (should happen once)
        fig,ax = plt.subplots(figsize=(6.8,2.8))
        bmap   = Basemap(projection='robin',lon_0=0,ax=ax)
        mle    = self.data["timeint_le"]
        print "mle lat/lon/unit:",mle.lat[4],mle.lon[4],mle.unit
        x,y    = bmap(mle.lon,mle.lat)
        mle    = mle.data
        print mle[4],obs.data[:,4].mean(),obs.unit
        print obs.data[:,4].mask.sum(),obs.data.shape[0]
        
        dmle   = mle.max()-mle.min()
        norm   = colors.Normalize(mle.min()-0.05*dmle,mle.max()+0.05*dmle)
        norm   = norm(mle)
        cmap   = plt.get_cmap('YlOrRd')
        clrs   = cmap(norm)
        size   = norm*30+20
        bmap.scatter(x,y,s=size,color=clrs,ax=ax,linewidths=0,cmap=cmap)
        bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")
        fig.savefig("%s/timeint_Benchmark.png" % (self.output_path))
        plt.close()



