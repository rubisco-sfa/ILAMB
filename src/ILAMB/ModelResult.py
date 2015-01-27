import glob
import ilamblib as il
import numpy as np

class ModelResult():
    """
    A class for exploring model results.
    """
    def __init__(self,path,modelname="unamed",color=(0,0,0),filter=""):
        self.path   = path
        self.color  = color
        self.filter = filter
        self.name   = modelname
        self.variables = []
        self.explore()

    def __str__(self):
        out  = "Model Result\n"
        out += "------------\n"
        out += "  Name: %s\n" % self.name
        out += "  Variables: %s\n" % (",".join(self.variables))
        out += "\n"
        return out

    def explore(self):
        from netCDF4 import Dataset
        self.variables = []
        for fname in glob.glob("%s/*%s*.nc" % (self.path,self.filter)):
            f = Dataset(fname)
            for key in f.variables.keys():
                if key not in self.variables: self.variables.append(key)
        
    def extractPointTimeSeries(self,variable,lat,lon,initial_time=-1e20,final_time=1e20):
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

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ma.core.MaskedArray
            an array of the extracted variable
        unit : string
            a description of the extracted unit
        """
        # create a list of data which has a non-null intersection over the desired time range
        data   = []
        ntimes = 0
        for fname in glob.glob("%s/*%s*.nc" % (self.path,self.filter)):
            try:
                t,var,unit = il.ExtractPointTimeSeries(fname,variable,lat,lon)
                nt      = ((t>=initial_time)*(t<=final_time)).sum()
                ntimes += nt
                if nt == 0: continue
            except il.VarNotInFile: 
                continue
            data.append((t,var))
        if ntimes == 0: raise il.VarNotInModel("%s does not exist in this model on that time frame" % variable)

        # sort the list by the first time, create a composite array
        data = sorted(data,key=lambda entry: entry[0][0])
        mono = np.asarray([entry[0][-1] for entry in data])
        mono = mono[:-1]>mono[1:]
        if mono.sum() > 0:
            # there seems to be some overlapping data so I will remove it
            for i in range(mono.shape[0]): 
                if mono[i]: 
                    tmp     = data.pop(i)
                    ntimes -= ((tmp[0]>=initial_time)*(tmp[0]<=final_time)).sum()
        shp  = [ntimes]; shp.extend(data[0][1].shape[1:])
        tc   = np.zeros(ntimes)
        varc = np.zeros(shp)
        masc = np.zeros(shp,dtype=bool)
        begin = 0
        for d in data:
            t,var = d
            mask = (t>=initial_time)*(t<=final_time)
            n = mask.sum(); end = begin+n
            tc  [begin:end] =        t[mask]
            varc[begin:end] = var.data[mask]
            if var.mask.size == 1: # whole array is either completely masked or not
                masc[begin:end] = var.mask
            else:
                masc[begin:end] = var.mask[mask]
            begin = end
        return tc,np.ma.masked_array(varc,mask=masc),unit
