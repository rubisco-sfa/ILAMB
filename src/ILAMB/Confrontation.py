import numpy as np
import ilamblib as il

class Confrontation():
    """
    A class for confronting model results with observational data.
    """
    def __init__(self,path):
        
        # Populate with observational data needed for the confrontation
        self.path = path
        mml       = np.genfromtxt("%s/monthly_mlo.csv" % path,delimiter=",",skip_header=57)
        self.t    = (mml[:,3]-1850.)*365. # days since 00:00:00 1/1/1850
        self.var  = np.ma.masked_where(mml[:,4]<0,mml[:,4])
        
        # This confrontation compares a point value
        self.lat  = 19.4
        self.lon  = 24.4

        # A list of the output variables and units that this confrontation requires
        self.requires = ["co2"]
        self.units    = ["ppm"]

        self.close = {}
        self.close["co2"] = ["co2mass"]


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
        t,var = M.extractPointTimeSeries(self.variable,
                                         self.lat,
                                         self.lon,
                                         initial_time=self.t.min(),
                                         final_time  =self.t.max())
        return t,var

    def computeNormalizedRootMeanSquaredError(self,M,t=[],var=[]):
        """
        

        """
        # if data wasn't passed in, grab it now
        if not (np.asarray(t).size or np.asarray(var).size):
            t,var = M.extractPointTimeSeries(self.variable,
                                             self.lat,
                                             self.lon,
                                             initial_time=self.t.min(),
                                             final_time  =self.t.max())
        begin = np.argmin(np.abs(self.t-t[0]))
        end   = begin+t.size
        return il.ComputeNormalizedRootMeanSquaredError(self.var[begin:end],var)

    def computeNormalizedBias(self,M,t=[],var=[]):
        """
        

        """
        # if data wasn't passed in, grab it now
        if not (np.asarray(t).size or np.asarray(var).size):
            t,var = M.extractPointTimeSeries(self.variable,
                                             self.lat,
                                             self.lon,
                                             initial_time=self.t.min(),
                                             final_time  =self.t.max())
        begin = np.argmin(np.abs(self.t-t[0]))
        end   = begin+t.size
        return il.ComputeNormalizedBias(self.var[begin:end],var)
