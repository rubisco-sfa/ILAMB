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
            
        

        return cdata
