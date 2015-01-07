import glob
import ilamblib as il
import numpy as np

class ModelResult():
    """
    A class for exploring model results.
    """
    def __init__(self,path):
        self.path = path
        
    def extractPointTimeSeries(self,variable,lat,lon,initial_time=-1e20,final_time=1e20,navg=1):
        """
        Extracts a time series of the given variable from the model
        results given a latitude and longitude.

        Parameters
        ----------
        variable : string
            name of the variable to extract
        lat : float
            latitude in degrees at which to extract field
        lon : float 
            longitude in degrees east of the international dateline at which to extract field
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        navg : int, optional
            if the model variable is layered, number of layers to average

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ndarray
            an array of the extracted variable

        Raises
        ------
        ValueError
            If no model result of that variable exists on the time frame given
        """
        # create a list of data which has a non-null intersection over the desired time range
        data   = []
        ntimes = 0
        for fname in glob.glob("%s/*.nc" % self.path):
            try:
                t,var = il.ExtractPointTimeSeries(fname,variable,lat,lon,navg=navg)
                nt    = ((t>=initial_time)*(t<=final_time)).sum()
                ntimes += nt
                if nt == 0: continue
            except:
                continue
            data.append((t,var))
        if ntimes == 0: raise ValueError("Model result does not exist in that time frame")

        # sort the list by the first time, create a composite array
        data = sorted(data,key=lambda entry: entry[0][0])
        tc   = np.zeros(ntimes)
        varc = np.zeros(ntimes)
        begin = 0
        for d in data:
            t,var = d
            mask = (t>=initial_time)*(t<=final_time)
            n = mask.sum(); end = begin+n
            tc  [begin:end] =   t[mask]
            varc[begin:end] = var[mask]
            begin = end
        return tc,varc
