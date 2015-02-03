import glob
import ilamblib as il
from constants import convert
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
        self.confrontations = {}

    def __str__(self):
        out  = "Model Result\n"
        out += "------------\n"
        out += "  Name: %s\n" % self.name
        out += "\n"
        return out
        
    def extractPointTimeSeries(self,variable,lat,lon,alt_vars=[],initial_time=-1e20,final_time=1e20,output_unit=""):
        """Extracts a time series of the given variable from the model
        results given a latitude and longitude.

        This routine will look for netCDF files with the "nc" suffix
        in the model directory. It will open all such files looking
        for the specified variable name. If the variable is found at
        the given latitude and longitude as defined by
        ILAMB.ilamblib.ExtractPointTimeSeries and at least partially
        on the desired time interval, this data is added to a
        list. Optionally a user may specify alternative variables, or
        alternative names of variables and the function will look for
        these also, giving preference to the given variable. After
        examining all files, then the routine will sort the list in
        ascending time and then check/disgard overlapping time
        segments. Finally, a composite data array is returned.

        Parameters
        ----------
        variable : string
            name of the variable to extract
        lat : float
            latitude in degrees at which to extract field
        lon : float 
            longitude in degrees east of the international dateline at 
            which to extract field
        alt_vars: list of strings, optional
            alternate variables to search for if `variable' is not found
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        output_unit : string, optional
            if specified, will try to convert the units of the variable 
            extract to these units given. (See convert in ILAMB.constants)

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ma.core.MaskedArray
            an array of the extracted variable
        unit : string
            a description of the extracted unit

        """
        altvars = list(alt_vars)
        altvars.insert(0,variable)

        # create a list of data which has a non-null intersection over the desired time range
        data   = []
        ntimes = 0
        for fname in glob.glob("%s/*%s*.nc" % (self.path,self.filter)):
            found = False
            for vname in altvars:
                try:
                    t,var,unit = il.ExtractPointTimeSeries(fname,vname,lat,lon)
                    nt      = ((t>=initial_time)*(t<=final_time)).sum()
                    ntimes += nt
                    if nt == 0: continue
                    data.append((t,var,vname,nt))
                except il.VarNotInFile: 
                    continue
        if ntimes == 0: 
            raise il.VarNotInModel("These variable(s) do not exist in this model on that time frame: %s" % (",".join(altvars)))
            
        # a model might have the variable and its alternates, only use the highest preference variable present
        thin   = []
        for vname in altvars:
            for d in data:
                if d[2] == vname: thin.append(d)
            if len(thin) > 0: break
        data = thin

        # now check again that data exists on this time frame
        ntimes = 0
        for d in data: ntimes += d[3]
        if ntimes == 0: 
            raise il.VarNotInModel("These variable(s) do not exist in this model on that time frame: %s" % (",".join(altvars)))

        # sort the list by the first time, create a composite array
        data = sorted(data,key=lambda entry: entry[0][0])
        mono = np.asarray([entry[0][-1] for entry in data])
        mono = mono[:-1]>mono[1:]
        if mono.sum() > 0: # there seems to be some overlapping data so I will remove it
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
            t,var,vname,nt = d
            mask = (t>=initial_time)*(t<=final_time)
            n = mask.sum(); end = begin+n
            tc  [begin:end] =        t[mask]
            varc[begin:end] = var.data[mask]
            if var.mask.size == 1: # whole array is either completely masked or not
                masc[begin:end] = var.mask
            else:
                masc[begin:end] = var.mask[mask]
            begin = end

        # if you asked for a specific unit, try to convert
        if output_unit is not "":
            try:
                varc *= convert[variable][output_unit][unit]
                unit = output_unit
            except:
                raise il.UnknownUnit("Variable is in units of [%s], you asked for [%s] but I do not know how to convert" % (unit,output_unit))
        return tc,np.ma.masked_array(varc,mask=masc),unit
