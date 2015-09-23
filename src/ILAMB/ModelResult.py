import glob
import ilamblib as il
from constants import convert
import numpy as np
from netCDF4 import Dataset
import os
from Variable import Variable

def CombineVariables(V):
    # checks on data
    assert type(V) == type([])
    for v in V: assert v.temporal
    if len(V) == 1: return V[0]
    
    # Put list in order by initial time
    V.sort(key=lambda v: v.time[0])

    # Check the beginning and ends times for monotonicity
    nV  = len(V)
    t0  = np.zeros(nV)
    tf  = np.zeros(nV)
    nt  = np.zeros(nV,dtype=int)
    ind = [0]
    for i,v in enumerate(V):
        t0[i] = v.time[ 0]
        tf[i] = v.time[-1]
        nt[i] = v.time.size
        ind.append(nt[:(i+1)].sum())
        
    # Checks on monotonicity
    assert (t0[1:]-t0[:-1]).min() >= 0
    assert (tf[1:]-tf[:-1]).min() >= 0
    assert (t0[1:]-tf[:-1]).min() >= 0

    # Assemble the data
    shp  = (nt.sum(),)+V[0].data.shape[1:]
    time = np.zeros(shp[0])
    data = np.zeros(shp)
    mask = np.zeros(shp,dtype=bool)
    for i,v in enumerate(V):
        time[ind[i]:ind[i+1]]     = v.time
        data[ind[i]:ind[i+1],...] = v.data
        mask[ind[i]:ind[i+1],...] = v.data.mask
    v = V[0]
    return Variable(data  = np.ma.masked_array(data,mask=mask),
                    unit  = v.unit,
                    name  = v.name,
                    time  = time,
                    lat   = v.lat,
                    lon   = v.lon,
                    area  = v.area,
                    ndata = v.ndata)

class ModelResult():
    """A class for exploring model results.
    """
    def __init__(self,path,modelname="unamed",color=(0,0,0),filter=""):
        self.path           = path
        self.color          = color
        self.filter         = filter
        self.name           = modelname
        self.confrontations = {}
        self.cell_areas     = None
        self.land_fraction  = None
        self.land_areas     = None
        self.land_area      = None
        self.lat            = None
        self.lon            = None        
        self.lat_bnds       = None
        self.lon_bnds       = None        
        self._getGridInformation()

    def _fileExists(self,contains):
        """Looks through the model result path for a file that contains the text specified in "constains". Returns "" if not found.
        """
        fname = ""
        for subdir, dirs, files in os.walk(self.path):
            for f in files:
                if contains not in f: continue
                if ".nc" not in f: continue
                fname = "%s/%s" % (subdir,f)
                return fname
        return fname

    def _getGridInformation(self):
        # Look for a file named areacella...
        fname = self._fileExists("areacella")
        if fname == "": return # there are no areas associated with this model result

        # Now grab area information for this model
        f = Dataset(fname)
        self.cell_areas    = f.variables["areacella"][...]
        self.lat           = f.variables["lat"][...]
        self.lon           = f.variables["lon"][...]
        self.lat_bnds      = np.zeros(self.lat.size+1)
        self.lat_bnds[:-1] = f.variables["lat_bnds"][:,0]
        self.lat_bnds[-1]  = f.variables["lat_bnds"][-1,1]
        self.lon_bnds      = np.zeros(self.lon.size+1)
        self.lon_bnds[:-1] = f.variables["lon_bnds"][:,0]
        self.lon_bnds[-1]  = f.variables["lon_bnds"][-1,1]

        # Now we do the same for land fractions
        fname = self._fileExists("sftlf")
        if fname == "": 
            self.land_areas = self.cell_areas 
        else:
            self.land_fraction = (Dataset(fname).variables["sftlf"])[...]
            # some models represent the fraction as a percent 
            if np.ma.max(self.land_fraction) > 1: self.land_fraction *= 0.01 
            self.land_areas = self.cell_areas*self.land_fraction
        self.land_area = np.ma.sum(self.land_areas)
        return
                
    def extractTimeSeries(self,variable,lats=None,lons=None,alt_vars=[],initial_time=-1e20,final_time=1e20,output_unit=""):
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
        # prepend the target variable to the list of possible variables
        altvars = list(alt_vars)
        altvars.insert(0,variable)

        # create a list of datafiles which have a non-null intersection
        # over the desired time range
        data = []
        for subdir, dirs, files in os.walk(self.path):
            for fileName in files:
                if ".nc"       not in fileName: continue
                if self.filter not in fileName: continue
                pathName  = "%s/%s" % (subdir,fileName)
                dataset   = Dataset(pathName)
                variables = dataset.variables.keys()
                intersect = [v for v in altvars if v in variables]
                if len(intersect) == 0: continue
                time      = il._convertCalendar(dataset.variables["time"])
                if (time[0] > final_time or time[-1] < initial_time): continue
                data.append(pathName)
        
        # some data are marked "-derived", check for these and give preference
        derived = [f for f in data if "-derived" in f]
        for f in derived:
            f = f.replace("-derived","")
            if f in data: data.remove(f)

        # build variable and merge if in separate files
        if len(data) == 0:
            raise il.VarNotInModel()
        else:
            V = []
            for pathName in data:
                if lats is None:
                    V.append(Variable(filename       = pathName,
                                      variable_name  = variable,
                                      alternate_vars = altvars[1:],
                                      area           = self.land_areas))
                else:
                    V.append(Variable(filename       = pathName,
                                      variable_name  = variable,
                                      alternate_vars = altvars[1:],
                                      area           = self.land_areas).extractDatasites(lats,lons))
        v = CombineVariables(V)

        # adjust the time range
        begin  = np.argmin(np.abs(v.time-initial_time))
        end    = np.argmin(np.abs(v.time-final_time))+1
        v.time = v.time[begin:end]
        v.data = v.data[begin:end,...]
        return v
            
