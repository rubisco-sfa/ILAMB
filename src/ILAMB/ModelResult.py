import glob
import ilamblib as il
import numpy as np

class ModelResult():
    """
    A class for exploring model results.
    """
    def __init__(self,path,filter=""):
        self.path   = path
        self.filter = filter

    def explore(self):
        from netCDF4 import Dataset
        for fname in glob.glob("%s/*%s*.nc" % (self.path,self.filter)):
            f = Dataset(fname)
            for key in f.variables.keys():
                if key.count("co2") > 0: print fname,key
        
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
        unit : string
            a description of the extracted unit

        Raises
        ------
        ValueError
            If no model result of that variable exists on the time frame given
        """
        # create a list of data which has a non-null intersection over the desired time range
        data   = []
        ntimes = 0
        for fname in glob.glob("%s/*%s*.nc" % (self.path,self.filter)):
            try:
                t,var,unit = il.ExtractPointTimeSeries(fname,variable,lat,lon,navg=navg)
                nt      = ((t>=initial_time)*(t<=final_time)).sum()
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
        masc = np.zeros(ntimes,dtype=bool)
        begin = 0
        for d in data:
            t,var = d
            mask = (t>=initial_time)*(t<=final_time)
            n = mask.sum(); end = begin+n
            if var is not np.ma.masked:
                tc  [begin:end] =   t[mask]
                varc[begin:end] = var[mask]
                masc[begin:end] =     mask
            else:
                tc  [begin:end] =        t[mask]
                varc[begin:end] = var.data[mask]
                masc[begin:end] = var.mask[mask]
            begin = end
        return tc,np.ma.masked_array(varc,mask=masc),unit
