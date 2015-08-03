from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from Variable import Variable,FromNetCDF4
from netCDF4 import Dataset
from Post import ColorBar,TaylorDiagram
from ilamblib import VarNotOnTimeScale
import pylab as plt
import numpy as np
import os,glob

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
        
        # some data we will store to send to plot routine
        cdata  = {}
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
        cdata["le"] = mod_le
        
        # the observational site data has many holes, mask the model
        # data so we are comparing apples to apples
        try:
            mod_le.data.mask += obs_le.data.mask
        except:
            raise VarNotOnTimeScale("Model has %d time points and observations have %d" % (mod_le.data.shape[0],obs_le.data.shape[0]))
        
        # write confrontation result file
        f = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        f.setncatts({"name":m.name,"color":m.color})
        
        # integrate over the time period
        obs_le_timeint = obs_le.integrateInTime(mean=True).convert("J m-2 s-1")
        mod_le_timeint = mod_le.integrateInTime(mean=True).convert("J m-2 s-1")
        obs_le_timeint.unit = "W m-2"
        mod_le_timeint.unit = "W m-2"
        self.data["timeint_le"] = obs_le_timeint
        cdata    ["timeint_le"] = mod_le_timeint
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
        
        def _UnitStringToMatplotlib(unit):
            import re
            # raise exponents using Latex
            match = re.findall("(-\d)",unit)
            for m in match: unit = unit.replace(m,"$^{%s}$" % m)
            return unit

        # Load data
        timeint = {}
        bias    = {}
        timeint["Benchmark"] = self.data["timeint_le"]
        pattern = "%s/%s*.nc" % (self.output_path,self.name)
        minLE   = timeint["Benchmark"].data.min() 
        maxLE   = timeint["Benchmark"].data.max()
        maxBias = 0.
        models  = []
        clr     = []
        cor     = []
        std     = []
        for fname in glob.glob(pattern):
            f     = Dataset(fname)
            mname = f.getncattr("name")
            models.append(mname)
            clr.append(f.getncattr("color"))
            f.close()
            timeint[mname] = FromNetCDF4(fname,"hfls_integrated_over_time_and_divided_by_time_period")
            bias   [mname] = timeint[mname].data-timeint["Benchmark"].data
            minLE   = min(minLE  ,timeint[mname].data.min())
            maxLE   = max(maxLE  ,timeint[mname].data.max())
            maxBias = max(maxBias,(np.abs(bias[mname])).max())
            cor.append(FromNetCDF4(fname,"correlation").data)
            std.append(FromNetCDF4(fname,"std"        ).data)
            
        # timeint
        minLE -= 0.05*(maxLE-minLE)
        maxLE += 0.05*(maxLE-minLE)
        for mname in timeint.keys():
            fig,ax = plt.subplots(figsize=(6.8,3.8),tight_layout=True)
            bmap   = Basemap(projection='robin',lon_0=0,ax=ax)
            mle    = timeint[mname]
            x,y    = bmap(timeint["Benchmark"].lon,timeint["Benchmark"].lat)
            mle    = mle.data
            norm   = colors.Normalize(minLE,maxLE)
            norm   = norm(mle)
            cmap   = plt.get_cmap('YlOrRd')
            clrs   = cmap(norm)
            size   = norm*30+20
            bmap.scatter(x,y,s=size,color=clrs,ax=ax,linewidths=0,cmap=cmap)
            bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")
            fig.savefig("%s/timeint_%s.png" % (self.output_path,mname))
            plt.close()
        fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
        ColorBar(ax,
                 vmin=minLE,
                 vmax=maxLE,
                 cmap="YlOrRd",
                 label=_UnitStringToMatplotlib(timeint[timeint.keys()[0]].unit))
        fig.savefig("%s/legend_timeint.png" % (self.output_path))
        plt.close()

        # bias
        maxBias *= 1.05
        minBias  = -maxBias
        for mname in bias.keys():
            fig,ax = plt.subplots(figsize=(6.8,3.8),tight_layout=True)
            bmap   = Basemap(projection='robin',lon_0=0,ax=ax)
            mle    = bias[mname]
            x,y    = bmap(timeint["Benchmark"].lon,timeint["Benchmark"].lat)
            norm   = colors.Normalize(minBias,maxBias)
            norm   = norm(mle)
            cmap   = plt.get_cmap('seismic')
            clrs   = cmap(norm)
            size   = norm*30+20
            bmap.scatter(x,y,s=size,color=clrs,ax=ax,linewidths=0,cmap=cmap)
            bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")
            fig.savefig("%s/bias_%s.png" % (self.output_path,mname))
            plt.close()
        fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
        ColorBar(ax,
                 vmin=minBias,
                 vmax=maxBias,
                 cmap="seismic",
                 label=_UnitStringToMatplotlib(timeint["Benchmark"].unit))
        fig.savefig("%s/legend_bias.png" % (self.output_path))
        plt.close()

        # spatial variation
        fig = plt.figure(figsize=(6.8,6.8))
        TaylorDiagram(np.asarray(std),np.asarray(cor),1.0,
                      fig,clr,normalize=False)
        fig.savefig("%s/spatial_variance.png" % (self.output_path))
        plt.close()
            
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
        
