import numpy as np
import ilamblib as il

class Confrontation():
    """
    A class for confronting model results with observational data.
    """
    def __init__(self,path):

        # Populate with observational data needed for the confrontation
        self.name = "CO2MaunaLoa"
        self.path = path
        mml       = np.genfromtxt("%s/monthly_mlo.csv" % path,delimiter=",",skip_header=57)
        self.t    = (mml[:,3]-1850.)*365. # days since 00:00:00 1/1/1850
        self.var  = np.ma.masked_where(mml[:,4]<0,mml[:,4])
        
        # This confrontation compares data at a point
        self.lat  = 19.4
        self.lon  = 24.4

        # A list of the output variables and units that this confrontation requires
        self.requires = ["co2"]
        self.units    = ["1e-6"]
        self.orders   = [300] # rough order of magnitudes

        self.close = {}
        self.close["co2"] = "co2mass"


    def getData(self,t0,tf):
        begin = np.argmin(np.abs(self.t-t0))
        end   = np.argmin(np.abs(self.t-tf))+1
        return self.t[begin:end],self.var[begin:end]

    def extractModelResult(self,M):
        """
        Extracts the model result on the time interval needed for this confrontation.

        Parameters
        ----------
        M : ILAMB.ModelResult.ModelResult
            the model results

        Returns
        -------
        t,var : numpy.ndarray
            the time series of the variable on the confrontation time interval
        """
        cdata = {}
        for rvar,runit,mag in zip(self.requires,self.units,self.orders):
            try:
                t,var,unit = M.extractPointTimeSeries(rvar,
                                                      self.lat,
                                                      self.lon,
                                                      initial_time=self.t.min(),
                                                      final_time  =self.t.max())
                if unit == runit: 

                    # some co2 comes in layers of the atmosphere, take the first non-null value
                    first = np.apply_along_axis(np.sum,1,var.mask)
                    var   = var[np.ix_(range(t.shape[0])),first][0,:]

                    # some files list unit as "ppm" but are really unitless, convert!
                    magdiff = int(round(np.log10(mag)-np.log10(np.ma.mean(var)),0))
                    cdata[rvar] = {}
                    cdata[rvar]["t"]    = t
                    cdata[rvar]["var"]  = var
                    if magdiff == 6: cdata[rvar]["var"] *= 1e6
                    cdata[rvar]["unit"] = unit
                    
            except il.VarNotInModel:
                pass # model variable doesn't exist, but that's ok
                
        # If data that is required isn't in our dictionary, then try 'close' variables
        for key in self.requires:
            if key not in cdata.keys():
                try:
                    t,var,unit = M.extractPointTimeSeries(self.close[rvar],
                                                          self.lat,
                                                          self.lon,
                                                          initial_time=self.t.min(),
                                                          final_time  =self.t.max())
                    cdata[rvar] = {}
                    cdata[rvar]["t"]    = t
                    cdata[rvar]["var"]  = var
                    cdata[rvar]["unit"] = unit
                    
                    if unit == "kg":
                        from constants import co2_ppm_per_kg
                        cdata[rvar]["var"] *= co2_ppm_per_kg
                        cdata[rvar]["unit"] = "1e-6"

                except il.VarNotInModel:
                    pass # model variable doesn't exist, but that's ok
                    
        if not cdata: 
            raise il.VarNotInModel("Required variables do not exist in this model on the confrontation time frame")
            
        # Now we assume that the dictionary has what we need and do some analysis

        # First we add the confrontation data to the dictionary, adjusted to the time of the model result
        begin = np.argmin(np.abs(self.t-cdata["co2"]["t"][0]))
        end   = begin+cdata["co2"]["t"].size
        cdata["data"]         = {}
        cdata["data"]["t"]    = self.t  [begin:end]
        cdata["data"]["var"]  = self.var[begin:end]
        cdata["data"]["unit"] = "1e-6"

        # Next we perform some analysis of the model and confrontation data
        for key in ["co2","data"]:
            annual_mean_times,annual_mean            = il.AnnualMean      (cdata[key]["t"],cdata[key]["var"])
            decade_mean_times,amplit_mean,amplit_std = il.DecadalAmplitude(cdata[key]["t"],cdata[key]["var"])
            trend = il.WindowedTrend (cdata[key]["t"],cdata[key]["var"])
            tmax  = il.DecadalMaxTime(cdata[key]["t"],cdata[key]["var"])
            tmin  = il.DecadalMinTime(cdata[key]["t"],cdata[key]["var"])
            cdata[key]["Annual"]          = {}        
            cdata[key]["Annual"]["t"]     = annual_mean_times
            cdata[key]["Annual"]["mean"]  = annual_mean
            cdata[key]["Decadal"]         = {}        
            cdata[key]["Decadal"]["t"]    = decade_mean_times
            cdata[key]["Decadal"]["amplitude_mean"] = amplit_mean
            cdata[key]["Decadal"]["amplitude_std"]  = amplit_std
            cdata[key]["Decadal"]["tmax"] = tmax
            cdata[key]["Decadal"]["tmin"] = tmin
            cdata[key]["Trend"]           = trend
            
        # Finally we compute some metrics
        cdata["metrics"] = {}; m = cdata["metrics"]
        m["RawBias"]   = il.Bias                (cdata["data"]["var"]  ,cdata["co2"]["var"])
        m["RawRMSE"]   = il.RootMeanSquaredError(cdata["data"]["var"]  ,cdata["co2"]["var"])
        m["TrendBias"] = il.Bias                (cdata["data"]["Trend"],cdata["co2"]["Trend"])
        m["TrendRMSE"] = il.RootMeanSquaredError(cdata["data"]["Trend"],cdata["co2"]["Trend"])
        m["AmpMeanBias"] = il.Bias(cdata["data"]["Decadal"]["amplitude_mean"],
                                   cdata["co2" ]["Decadal"]["amplitude_mean"])
        m["AmpMeanRMSE"] = il.RootMeanSquaredError(cdata["data"]["Decadal"]["amplitude_mean"],
                                                   cdata["co2" ]["Decadal"]["amplitude_mean"])
        m["AmpStdBias"] = il.Bias(cdata["data"]["Decadal"]["amplitude_std"],
                                  cdata["co2" ]["Decadal"]["amplitude_std"])
        m["AmpStdRMSE"] = il.RootMeanSquaredError(cdata["data"]["Decadal"]["amplitude_std"],
                                                  cdata["co2" ]["Decadal"]["amplitude_std"])

        # Define phase shift
        max_time_diff   = cdata["co2"]["Decadal"]["tmax"]-cdata["data"]["Decadal"]["tmax"]
        min_time_diff   = cdata["co2"]["Decadal"]["tmin"]-cdata["data"]["Decadal"]["tmin"]
        phase_shift     = 0.5*(max_time_diff + min_time_diff)
        m["PhaseShift"    ] = phase_shift
        m["PhaseShiftMean"] = phase_shift.mean()

        return cdata
