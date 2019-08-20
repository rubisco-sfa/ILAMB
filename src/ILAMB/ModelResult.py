from .Variable import Variable
from netCDF4 import Dataset
from . import ilamblib as il
import numpy as np
import glob,os,re
from mpi4py import MPI
import logging

logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)

class ModelResult():
    """A class for exploring model results.

    This class provides a simplified way of accessing model
    results. It is essentially a pointer to a top level directory and
    defines the model as all netCDF4 files found in its
    subdirectories. If this directory contains model output from
    several runs or experiments, you may wish to specify a string (the
    *filter* argument) which we will require to be in the filename for
    it to be considered part of the model.

    Parameters
    ----------
    path : str
        the full path to the directory which contains the model result
        files
    modelname : str, optional
        a string representing the name of the model, will be used as a
        label in plot legends
    color : 3-tuple, optional
        a normalized tuple representing a color in RGB color space,
        will be used to color line plots
    filter : str, optional
        this string must be in file's name for it to be considered as
        part of the model results
    model_year : 2-tuple of int, optional
        used to shift model times, all model years at model_year[0]
        are shifted to model_year[1]
    """
    def __init__(self,path,modelname="unamed",color=(0,0,0),filter="",regex="",model_year=None):
        self.path           = path
        self.color          = color
        self.filter         = filter
        self.regex          = regex
        self.shift          = 0.
        if model_year is not None: self.shift = (model_year[1]-model_year[0])*365.
        self.name           = modelname
        self.confrontations = {}
        self.cell_areas     = None
        self.land_fraction  = None
        self.land_areas     = None
        self.land_area      = None
        self.variables      = None
        self.names          = None
        self.extents        = np.asarray([[-90.,+90.],[-180.,+180.]])
        self._findVariables()
        self._getGridInformation()

    def __str__(self):
        s  = ""
        s += "ModelResult: %s\n" % self.name
        s += "-"*(len(self.name)+13) + "\n"
        for key in self.names:
            s += "{0:>20}: {1:<50}".format(key,self.names[key]) + "\n"
        return s

    def _findVariables(self):
        """Loops through the netCDF4 files in a model's path and builds a dictionary of which variables are in which files.
        """
        def _get(key,dset):
            dim_name = key
            try:
                v = dset.variables[key]
                dim_bnd_name = v.getncattr("bounds")
            except:
                dim_bnd_name = None
            return dim_name,dim_bnd_name

        variables = {}
        names     = {}
        for subdir, dirs, files in os.walk(self.path,followlinks=True):
            for fileName in files:
                if not fileName.endswith(".nc"): continue
                if self.filter not in fileName: continue
                if self.regex is not "":
                    m = re.search(self.regex,fileName)
                    if not m: continue
                pathName  = os.path.join(subdir,fileName)

                try:
                    dataset = Dataset(pathName)
                except:
                    logger.debug("[%s] Error opening file %s" % (self.name,pathName))
                    continue

                # populate dictionary for which variables are in which files
                for key in dataset.variables.keys():
                    if key not in variables:
                        variables[key] = []
                    variables[key].append(pathName)

                    v = dataset.variables[key]
                    if key not in names:
                        if "long_name" in v.ncattrs():
                            names[key] = v.long_name
                            continue
                        if "standard_name" in v.ncattrs():
                            names[key] = v.standard_name
                            continue

        # determine spatial extents
        lats = [key for key in variables.keys() if (key.lower().startswith("lat" ) or
                                                    key.lower().  endswith("lat" ))]
        lons = [key for key in variables.keys() if (key.lower().startswith("lon" ) or
                                                    key.lower().  endswith("lon" ) or
                                                    key.lower().startswith("long") or
                                                    key.lower().  endswith("long"))]
        for key in lats:
            for pathName in variables[key]:
                with Dataset(pathName) as dset:
                    lat = dset.variables[key][...]
                    if lat.size == 1: continue
                    self.extents[0,0] = max(self.extents[0,0],lat.min())
                    self.extents[0,1] = min(self.extents[0,1],lat.max())
        for key in lons:
            for pathName in variables[key]:
                with Dataset(pathName) as dset:
                    lon = dset.variables[key][...]
                    if lon.size == 1: continue
                    if lon.ndim < 1 or lon.ndim > 2: continue
                    lon = (lon<=180)*lon + (lon>180)*(lon-360) + (lon<-180)*360
                    self.extents[1,0] = max(self.extents[1,0],lon.min())
                    self.extents[1,1] = min(self.extents[1,1],lon.max())

        # fix extents
        eps = 5.
        if self.extents[0,0] < (- 90.+eps): self.extents[0,0] = - 90.
        if self.extents[0,1] > (+ 90.-eps): self.extents[0,1] = + 90.
        if self.extents[1,0] < (-180.+eps): self.extents[1,0] = -180.
        if self.extents[1,1] > (+180.-eps): self.extents[1,1] = +180.
        self.variables = variables
        self.names = names

    def _getGridInformation(self):
        """Looks in the model output for cell areas as well as land fractions.
        """
        def _shiftLon(lon):
            return (lon<=180)*lon + (lon>180)*(lon-360) + (lon<-180)*360
        
        # Are there cell areas associated with this model?
        if "areacella" in self.variables.keys():
            with Dataset(self.variables["areacella"][0]) as f:
                self.cell_areas = f.variables["areacella"][...]
        else:
            if not ("lat_bnds" in self.variables.keys() and
                    "lon_bnds" in self.variables.keys()): return
            with Dataset(self.variables["lat_bnds"][0]) as f:
                x = f.variables["lat_bnds"][...]
            with Dataset(self.variables["lon_bnds"][0]) as f:
                y = f.variables["lon_bnds"][...]
                s = y.mean(axis=1).argmin()
                y = np.roll(_shiftLon(y),-s,axis=0)
                if y[ 0,0] > y[ 0,1]: y[ 0,0] = -180.
                if y[-1,0] > y[-1,1]: y[-1,1] = +180.
            self.cell_areas = il.CellAreas(None,None,lat_bnds=x,lon_bnds=y)
            
        # Now we do the same for land fractions
        if "sftlf" not in self.variables.keys():
            self.land_areas = self.cell_areas
        else:
            with Dataset(self.variables["sftlf"][0]) as f:
                self.land_fraction = f.variables["sftlf"][...]                
            # some models represent the fraction as a percent
            if np.ma.max(self.land_fraction) > 1: self.land_fraction *= 0.01
            with np.errstate(over='ignore',under='ignore'):
                if not np.allclose(self.cell_areas.shape,self.land_fraction.shape):
                    msg = "The model %s has areacella %s which is a different shape than sftlf %s" % (self.name,
                                                                                                      str(self.cell_areas.shape),
                                                                                                      str(self.land_fraction.shape))
                    raise ValueError(msg)
                self.land_areas = self.cell_areas*self.land_fraction
        self.land_area = np.ma.sum(self.land_areas)
        return

    def extractTimeSeries(self,variable,lats=None,lons=None,alt_vars=[],initial_time=-1e20,final_time=1e20,output_unit="",expression=None,convert_calendar=True):
        """Extracts a time series of the given variable from the model.

        Parameters
        ----------
        variable : str
            name of the variable to extract
        alt_vars : list of str, optional
            alternate variables to search for if *variable* is not found
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        output_unit : str, optional
            if specified, will try to convert the units of the variable
            extract to these units given.
        lats : numpy.ndarray, optional
            a 1D array of latitude locations at which to extract information
        lons : numpy.ndarray, optional
            a 1D array of longitude locations at which to extract information
        expression : str, optional
            an algebraic expression describing how to combine model outputs

        Returns
        -------
        var : ILAMB.Variable.Variable
            the extracted variable

        """
        # prepend the target variable to the list of possible variables
        altvars = list(alt_vars)
        altvars.insert(0,variable)

        # checks on input consistency
        if lats is not None: assert lons is not None
        if lons is not None: assert lats is not None
        if lats is not None: assert lats.shape == lons.shape

        # create a list of datafiles which have a non-null intersection
        # over the desired time range
        V = []
        tmin =  1e20
        tmax = -1e20
        same_site_epsilon = 0.1
        for v in altvars:
            if v not in self.variables: continue
            for pathName in self.variables[v]:
                var = Variable(filename       = pathName,
                               variable_name  = variable,
                               alternate_vars = altvars[1:],
                               area           = self.land_areas,
                               convert_calendar = convert_calendar,
                               t0             = initial_time - self.shift,
                               tf             = final_time   - self.shift)
                if var.time is None: continue
                tmin = min(tmin,var.time_bnds.min())
                tmax = max(tmax,var.time_bnds.max())
                if ((var.time_bnds.max() < initial_time - self.shift) or
                    (var.time_bnds.min() >   final_time - self.shift)): continue
                if lats is not None and var.ndata:
                    r = np.sqrt((lats[:,np.newaxis]-var.lat)**2 +
                                (lons[:,np.newaxis]-var.lon)**2)
                    imin = r.argmin(axis=1)
                    rmin = r.   min(axis=1)
                    imin = imin[np.where(rmin<same_site_epsilon)]
                    if imin.size == 0:
                        continue
                    var.lat   = var.lat [  imin]
                    var.lon   = var.lon [  imin]
                    var.data  = var.data[:,imin]
                    var.ndata = var.data.shape[1]
                if lats is not None and var.spatial: var = var.extractDatasites(lats,lons)
                var.time      += self.shift
                var.time_bnds += self.shift
                V.append(var)
            if len(V) > 0: break

        # If we didn't find any files, try to put together the
        # variable from a given expression
        if len(V) == 0:
            if expression is not None:
                v = self.derivedVariable(variable,
                                         expression,
                                         lats         = lats,
                                         lons         = lons,
                                         initial_time = initial_time,
                                         final_time   = final_time)
            else:
                tstr = ""
                if tmin < tmax: tstr = " in the given time frame, tinput = [%.1f,%.1f], tmodel = [%.1f,%.1f]" % (initial_time,final_time,tmin+self.shift,tmax+self.shift)
                logger.debug("[%s] Could not find [%s] in the model results%s" % (self.name,",".join(altvars),tstr))
                raise il.VarNotInModel()
        else:
            v = il.CombineVariables(V)


        return v

    def derivedVariable(self,variable_name,expression,lats=None,lons=None,initial_time=-1e20,final_time=1e20,convert_calendar=True):
        """Creates a variable from an algebraic expression of variables in the model results.

        Parameters
        ----------
        variable_name : str
            name of the variable to create
        expression : str
            an algebraic expression describing how to combine model outputs
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        lats : numpy.ndarray, optional
            a 1D array of latitude locations at which to extract information
        lons : numpy.ndarray, optional
            a 1D array of longitude locations at which to extract information

        Returns
        -------
        var : ILAMB.Variable.Variable
            the new variable

        """
        from sympy import sympify
        if expression is None: raise il.VarNotInModel()
        args  = {}
        units = {}
        unit  = expression
        mask  = None
        time  = None
        tbnd  = None
        lat   = None
        lon   = None
        ndata = None
        area  = None
        depth = None
        dbnds = None

        for arg in sympify(expression).free_symbols:

            var  = self.extractTimeSeries(arg.name,
                                          lats = lats,
                                          lons = lons,
                                          convert_calendar = convert_calendar,
                                          initial_time = initial_time,
                                          final_time   = final_time)
            units[arg.name] = var.unit
            args [arg.name] = var.data.data

            if mask is None:
                mask  = var.data.mask
            else:
                mask += var.data.mask
            if time is None:
                time  = var.time
            else:
                assert(np.allclose(time,var.time))
            if tbnd is None:
                tbnd  = var.time_bnds
            else:
                assert(np.allclose(tbnd,var.time_bnds))
            if lat is None:
                lat  = var.lat
            else:
                assert(np.allclose(lat,var.lat))
            if lon is None:
                lon  = var.lon
            else:
                assert(np.allclose(lon,var.lon))
            if area is None:
                area  = var.area
            else:
                assert(np.allclose(area,var.area))
            if ndata is None:
                ndata  = var.ndata
            else:
                assert(np.allclose(ndata,var.ndata))
            if depth is None:
                depth  = var.depth
            else:
                assert(np.allclose(depth,var.depth))
            if dbnds is None:
                dbnds  = var.depth_bnds
            else:
                assert(np.allclose(dbnds,var.depth_bnds))

        np.seterr(divide='ignore',invalid='ignore')
        result,unit = il.SympifyWithArgsUnits(expression,args,units)
        np.seterr(divide='raise',invalid='raise')
        mask  += np.isnan(result)
        result = np.ma.masked_array(np.nan_to_num(result),mask=mask)

        return Variable(data       = np.ma.masked_array(result,mask=mask),
                        unit       = unit,
                        name       = variable_name,
                        time       = time,
                        time_bnds  = tbnd,
                        lat        = lat,
                        lon        = lon,
                        area       = area,
                        ndata      = ndata,
                        depth      = depth,
                        depth_bnds = dbnds)
