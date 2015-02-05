import numpy as np
import ilamblib as il
from constants import convert

class CO2MaunaLoa():
    """
    A class for confronting model results with observational data.
    """
    def __init__(self):

        # Populate with observational data needed for the confrontation
        mml           = np.genfromtxt("../demo/data/monthly_mlo.csv",delimiter=",",skip_header=57)
        self.name     = "CO2MaunaLoa"
        self.t        = (mml[:,3]-1850.)*365. # convert to days since 00:00:00 1/1/1850
        self.var      = np.ma.masked_where(mml[:,4]<0,mml[:,4])
        self.unit     = "1e-6"
        self.lat      = 19.4
        self.lon      = 24.4
        self.nlayers  = 3 

    def getData(self,initial_time=-1e20,final_time=1e20,output_unit=""):
        """Retrieves the confrontation data on the desired time frame and in
        the desired unit.

        Parameters
        ----------
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        output_unit : string, optional
            if specified, will try to convert the units of the variable 
            extract to these units given (see convert in ILAMB.constants)

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ma.core.MaskedArray
            an array of the extracted variable
        unit : string
            a description of the extracted unit
        """
        begin = np.argmin(np.abs(self.t-initial_time))
        end   = np.argmin(np.abs(self.t-final_time))+1
        if output_unit is not "":
            try:
                self.var *= convert["co2"][output_unit][self.unit]
                self.unit = output_unit
            except:
                raise il.UnknownUnit("Variable is in units of [%s], you asked for [%s] but I do not know how to convert" % (self.unit,output_unit))
        return self.t[begin:end],self.var[begin:end],self.unit

    def confront(self,m):
        """Confronts the input model with the observational data.

        This confrontation uses the Mauna Loa CO2 data and compares it
        to model results at that given latutide and longitude. If the
        model has CO2 in mass instead of parts per million, it will
        convert the value. If the CO2 is given at atmospheric levels,
        it will average the first "nlayer" unmasked levels to get a
        single value for each time.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        Returns
        -------
        cdata : dictionary
            contains all outputs/metrics
        """
        # time limits for this confrontation, with a little padding to
        # account for differences in monthly time representations
        t0,tf = self.t.min()-5, self.t.max()+5

        # extract the time, variable, and unit of the model result
        tm,vm,um = m.extractPointTimeSeries("co2",self.lat,self.lon,
                                            initial_time=t0,
                                            final_time=tf,
                                            alt_vars=["co2mass"],
                                            output_unit="1e-6")
        
        # update time limits, might be less model data than observations
        t0,tf = tm.min(), tm.max()

        # get the corresponding observational data on the same time frame
        to,vo,uo = self.getData(initial_time=t0,final_time=tf,output_unit="1e-6")

        # perform some assertion to verify data integrity, are the
        # times within 15 days of each other?
        assert np.allclose(tm,to,atol=15)

        # some CO2 results are given in atmospheric layers, so we need
        # to handle this...
        if vm.ndim == 2:
            # ...however, many of the early layers are masked, so
            # determine which index is the first non-masked
            index = np.apply_along_axis(np.sum,1,vm.mask)

            # now we will average the first nlayers non-masked layers
            tmp = np.zeros(tm.size)
            for i in range(self.nlayers):
                tmp += vm[np.ix_(range(tm.size)),(index+i).clip(0,vm.shape[1])][0,:]
            vm = tmp/self.nlayers

        # now we can do some analysis, the results of which we will
        # load into a dictionary which we return. The
        # plotting/visualization routines are written to operate on
        # this dictionary so its format is important.
        cdata = {}

        # put the extracted model data and manipulations here
        cdata["model"] = {} 
        cdata["model"]["t"] = tm; cdata["model"]["var"] = vm; cdata["model"]["unit"] = um
        cdata["model"]["tannual"],cdata["model"]["vannual"] = il.AnnualMean(tm,vm)

        # put the observation data and manipulations here
        cdata["obs"] = {} 
        cdata["obs"]["t"] = to; cdata["obs"]["var"] = vo; cdata["obs"]["unit"] = uo
        cdata["obs"]["tannual"],cdata["obs"]["vannual"] = il.AnnualMean(to,vo)
        
        # include metrics
        mw = il.MonthlyWeights(tm)
        cdata["metric"] = {}
        cdata["metric"]["MonthlyMean"] = {"bias":il.Bias                (vm,vo,normalize="score",weights=mw),
                                          "rmse":il.RootMeanSquaredError(vm,vo,normalize="score")}
        cdata["metric"]["AnnualMean"]  = {"bias":il.Bias                (cdata["obs"]["vannual"],cdata["model"]["vannual"]),
                                          "rmse":il.RootMeanSquaredError(cdata["obs"]["vannual"],cdata["model"]["vannual"])}
        
        return cdata
