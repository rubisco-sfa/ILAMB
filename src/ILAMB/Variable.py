from netCDF4 import Dataset
import numpy as np
import ilamblib as il
from constants import spd,dpy,mid_months,convert,regions as ILAMBregions
import Post as post
from copy import deepcopy
from cfunits import Units

import pylab as plt ### FIX: only import what I need

# Note concerning use of np.seterr: Currently, operations on masked
# arrays occurs on the whole array internally and then the masks are
# used to ignore these values at the numpy level. This causes spurious
# over/underflow warnings which have been suppressed around the calls
# that have been indentified as sometimes problematic. 

def FromNetCDF4(filename,variable_name,alternate_vars=[]):
    """Extracts data from a netCDF4 datafile for use in a Variable object.
    
    Intended to be used inside of the Variable constructor. Some of
    the return arguments will be None depending on the contents of the
    netCDF4 file.

    Parameters
    ----------
    filename : str
        Name of the netCDF4 file from which to extract a variable
    variable_name : str
        Name of the variable to extract from the netCDF4 file
    alternate_vars : list of str, optional
        A list of possible alternate variable names to find

    Returns
    -------
    data : numpy.ma.masked_array
        The array which contains the data which constitutes the variable
    unit : str
        The unit of the input data
    name : str
        The name of the variable, will be how it is saved in an output netCDF4 file
    time : numpy.ndarray
        A 1D array of times in days since 1850-01-01 00:00:00
    lat : numpy.ndarray
        A 1D array of latitudes of cell centroids
    lon : numpy.ndarray
        A 1D array of longitudes of cell centroids
    area : numpy.ndarray
        A 2D array of the cell areas
    ndata : int
        Number of data sites this data represents

    """
    try:
        f = Dataset(filename,mode="r")
    except RuntimeError:
        raise RuntimeError("Unable to open the file: %s" % filename)
    
    found     = False
    if variable_name in f.variables.keys():
        found = True
        var   = f.variables[variable_name]
    else:
        for var_name in alternate_vars:
            if var_name in f.variables.keys():
                found = True
                var   = f.variables[var_name]
    assert found == True
    time_name = None
    lat_name  = None
    lon_name  = None
    data_name = None
           
    for key in var.dimensions:
        if "time" in key: time_name = key
        if "lat"  in key: lat_name  = key
        if "lon"  in key: lon_name  = key
        if "data" in key: data_name = key
    if time_name is None:
        t = None
    else:
        t = il._convertCalendar(f.variables[time_name])
    if lat_name is None:
        lat = None
    else:
        lat = f.variables[lat_name][...]
    if lon_name is None:
        lon = None
    else:
        lon = f.variables[lon_name][...]
    if data_name is None:
        data = None
    else:
        data = len(f.dimensions[data_name])
        # if we have data sites, there may be lat/lon data to come
        # along with them although not a dimension of the variable
        for key in f.variables.keys():
            if "lat" in key: lat_name = key
            if "lon" in key: lon_name = key
        if lat_name is not None: lat = f.variables[lat_name][...]
        if lon_name is not None: lon = f.variables[lon_name][...]
        if lat.size != data: lat = None
        if lon.size != data: lon = None

    # handle incorrect or absent masking of arrays
    if type(var[...]) == type(np.ma.empty([])):
        v    = var[...]
    else:
        v    = var[...]
        mask = np.zeros(v.shape,dtype=int)
        if "_FillValue"    in var.ncattrs(): mask += (np.abs(v-var._FillValue   )<1e-12)
        if "missing_value" in var.ncattrs(): mask += (np.abs(v-var.missing_value)<1e-12)
        v    = np.ma.masked_array(v,mask=mask,copy=False)

    # handle units problems that cfunits doesn't
    units = var.units
    if units == "unitless": units = "1"
    
    return v,units,variable_name,t,lat,lon,data

class Variable:
    """A class for managing variables and their analysis.

    There are two ways to create a Variable object. Because python
    does not support multiple constructors, we will use keyword
    arguments. The first way to specify a Variable is by loading a
    netCDF4 file. You can achieve this by specifying the 'filename'
    and 'variable_name' keywords. The second way is to use the
    remaining keyword arguments to specify data arrays directly. If
    you use the second way, you must specify the keywords 'data' and
    'unit'. The rest are truly optional and depend on the nature of
    your data.

    """
    def __init__(self,**keywords):
        """Constructor for the variable class by specifying the data arrays.

        Parameters
        ----------
        filename : str, optional
            Name of the netCDF4 file from which to extract a variable
        variable_name : str, optional
            Name of the variable to extract from the netCDF4 file
        data : numpy.ndarray, optional
            The array which contains the data which constitutes the
            variable
        unit : str, optional
            The unit of the input data
        name : str, optional
            The name of the variable, will be how it is saved in the netCDF4 
            file
        time : numpy.ndarray, optional
            a 1D array of times in days since 1850-01-01 00:00:00
        lat : numpy.ndarray, optional
            a 1D array of latitudes of cell centroids
        lon : numpy.ndarray, optional
            a 1D array of longitudes of cell centroids
        area : numpy.ndarray, optional
            a 2D array of the cell areas
        ndata : int, optional
            number of data sites this data represents
        alternate_vars : list of str, optional
            A list of alternate acceptable variable names
        """
        # See if the user specified a netCDF4 file and variable
        filename       = keywords.get("filename"     ,None)
        variable_name  = keywords.get("variable_name",None)
        alternate_vars = keywords.get("alternate_vars",[])
        if filename is None: # if not pull data from other arguments
            data  = keywords.get("data" ,None)
            unit  = keywords.get("unit" ,None)
            name  = keywords.get("name" ,"unnamed")
            time  = keywords.get("time" ,None)
            lat   = keywords.get("lat"  ,None)
            lon   = keywords.get("lon"  ,None)
            ndata = keywords.get("ndata",None)
            assert data is not None
            assert unit is not None
        else:
            assert variable_name is not None
            data,unit,name,time,lat,lon,ndata = FromNetCDF4(filename,variable_name,alternate_vars)

        if not np.ma.isMaskedArray(data): data = np.ma.masked_array(data)
        self.data  = data 
        self.ndata = ndata
        self.unit  = unit
        self.name  = name
        
        # Handle time data
        self.time     = time   # time data
        self.temporal = False  # flag for temporal data
        self.dt       = 0.     # mean temporal spacing
        self.monthly  = False  # flag for monthly means
        if time is not None: 
            self.temporal = True
            self.dt = (time[1:]-time[:-1]).mean()
            if np.allclose(self.dt,30,atol=3): self.monthly = True

        # Handle space or multimember data
        self.spatial = False
        self.lat     = lat
        self.lon     = lon
        self.area    = keywords.get("area",None)
        if ((lat is     None) and (lon is     None)): return
        if ((lat is     None) and (lon is not None) or
            (lat is not None) and (lon is     None)):
            raise ValueError("If one of lat or lon is specified, they both must specified")
        self.lon = (self.lon<=180)*self.lon+(self.lon>180)*(self.lon-360)
        if data.ndim < 2: return
        if (data.shape[-2] == lat.size and data.shape[-1] == lon.size):
            self.spatial = True
            if self.area is None: self.area = il.CellAreas(self.lat,self.lon)
            # Some data arrays are arranged such that the first column
            # of data is arranged at the prime meridian. This does not
            # work well with some of the plotting and/or analysis
            # operations we will need to perform. These require that
            # the first column be coincident with the international
            # dateline. Thus we roll the data the required amount.
            shift     = self.lon.argmin()
            self.lon  = np.roll(self.lon ,-shift)
            self.data = np.roll(self.data,-shift,axis=-1)
            self.area = np.roll(self.area,-shift,axis=-1)
            
    def __str__(self):
        if self.data  is None: return "Uninitialized Variable"
        if self.ndata is None:
            ndata = "N/A"
        else:
            ndata = str(self.ndata)
        if self.time is None:
            time = ""
        else:
            time = " (%d)" % self.time.size
        if self.lat is None:
            space = ""
        else:
            space = " (%d,%d)" % (self.lat.size,self.lon.size)
        s  = "Variable: %s\n" % self.name
        s += "-"*(len(self.name)+10) + "\n"
        s += "{0:>20}: ".format("unit")       + self.unit          + "\n"
        s += "{0:>20}: ".format("isTemporal") + str(self.temporal) + time  + "\n"
        s += "{0:>20}: ".format("isSpatial")  + str(self.spatial)  + space + "\n"
        s += "{0:>20}: ".format("nDatasites") + ndata              + "\n"
        s += "{0:>20}: ".format("dataShape")  + "%s\n" % (self.data.shape,)
        s += "{0:>20}: ".format("dataMax")    + "%e\n" % self.data.max()
        s += "{0:>20}: ".format("dataMin")    + "%e\n" % self.data.min()
        s += "{0:>20}: ".format("dataMean")   + "%e\n" % self.data.mean()
        return s

    def integrateInTime(self,**keywords):
        """Integrates the variable over time period.

        Uses nodal integration to integrate the variable over the time
        domain. 
        
        Parameters
        ----------
        t0 : float, optional
            initial time in days since 1/1/1850
        tf : float, optional
            final time in days since 1/1/1850
        mean : boolean, optional
            enable to divide the integrand to get the mean function value

        Returns
        -------
        integral : ILAMB.Variable.Variable
            a Variable instance with the integrated value along with the
            appropriate name and unit change.

        """
        if not self.temporal: raise il.NotTemporalVariable()
        t0   = keywords.get("t0",self.time.min())
        tf   = keywords.get("tf",self.time.max())
        mean = keywords.get("mean",False)

        # perform the integration using ILAMB.ilamblib
        integral = il.TemporallyIntegratedTimeSeries(self.time,self.data,t0=t0,tf=tf)

        # special handling of Watts --> Joules per second (improve this)
        unit = self.unit
        if "W" in unit: unit = unit.replace("W","J s-1")
        
        # handle units (fragile, depends on user following this convention)
        if " s-1" in unit:
            integral *= spd
            unit      = unit.replace(" s-1","")
        elif " d-1" in self.unit:
            unit      = unit.replace(" d-1","")
        elif " y-1" in self.unit: 
            integral /= dpy["noleap"]
            unit      = unit.replace(" y-1","")
        else:
            unit += " d"
        name = self.name + "_integrated_over_time"

        # divide thru by the non-masked amount of time
        if mean:
            dt        = np.zeros(self.time.shape)
            dt[1:-1]  = 0.5*(self.time[2:]-self.time[:-2])
            dt[0]     = 0.5*(self.time[1] -self.time[0])  + self.time[0]-t0
            dt[-1]    = 0.5*(self.time[-1]-self.time[-2]) + tf-self.time[-1]
            dt       *= (self.time>=t0)*(self.time<=tf)
            for i in range(self.data.ndim-1): dt = np.expand_dims(dt,axis=-1)
            dt        = (dt*(self.data.mask==0)).sum(axis=0)

            np.seterr(over='ignore',under='ignore')
            integral /= dt
            np.seterr(over='raise',under='raise')
            
            unit     += " d-1"
            name     += "_and_divided_by_time_period"

        # special handling of Joules per day --> Watts (again, need to improve)
        if "J" in unit and "d-1" in unit:
            integral /= spd
            unit      = unit.replace("J","W").replace("d-1","")
        unit = unit.replace("d d-1","")
        
        return Variable(data=integral,unit=unit,name=name,lat=self.lat,lon=self.lon,area=self.area,ndata=self.ndata)

    def integrateInSpace(self,region=None,mean=False):
        """Integrates the variable over space.

        Uses nodal integration to integrate the variable over the
        specified region. If no region is specified, then perform the
        integration over the extent of the dataset.
        
        Parameters
        ----------
        region : str, optional
            name of the region overwhich you wish to integrate
        mean : bool, optional
            enable to divide the integrand to get the mean function value
        
        Returns
        -------
        integral : ILAMB.Variable.Variable
            a Variable instace with the integrated value along with the
            appropriate name and unit change.
        """
        if not self.spatial: raise il.NotSpatialVariable()
        if region is None:
            integral = il.SpatiallyIntegratedTimeSeries(self.data,self.area)
            if mean: integral /= self.area.sum()
            name = self.name + "_integrated_over_space"
        else:
            rem_mask  = np.copy(self.data.mask)
            lats,lons = ILAMBregions[region]
            mask      = (np.outer((self.lat>lats[0])*(self.lat<lats[1]),
                                  (self.lon>lons[0])*(self.lon<lons[1]))==0)
            self.data.mask += mask
            integral  = il.SpatiallyIntegratedTimeSeries(self.data,self.area)
            self.data.mask = rem_mask
            if mean:
                mask = rem_mask+mask
                if mask.ndim > 2: mask = np.all(mask,axis=0)
                area = np.ma.masked_array(self.area,mask=mask)
                integral /= area.sum()
            name = self.name + "_integrated_over_%s" % region
        unit = self.unit
        if mean:
            name += "_and_divided_by_area"
        else:
            unit  = unit.replace(" m-2","")
        return Variable(data=np.ma.masked_array(integral),unit=unit,time=self.time,name=name)

    def siteStats(self,region=None):
        """
        UNTESTED, What to do when no sites are in a region?
        """
        if self.ndata is None: raise il.NotDatasiteVariable()
        rem_mask = np.copy(self.data.mask)
        rname = ""
        if region is not None:
            lats,lons = ILAMBregions[region]
            mask      = ((self.lat>lats[0])*(self.lat<lats[1])*(self.lon>lons[0])*(self.lon<lons[1]))==False
            self.data.mask += mask
            rname = "_over_%s" % region
        np.seterr(over='ignore',under='ignore')
        mean = self.data.mean(axis=-1)
        std  = self.data.std (axis=-1)
        np.seterr(over='raise',under='raise')
        self.data.mask = rem_mask
        mean = Variable(data=mean,unit=self.unit,time=self.time,name="mean_%s%s" % (self.name,rname))
        std  = Variable(data=std ,unit=self.unit,time=self.time,name="std_%s%s"  % (self.name,rname))
        return mean,std
    
    def annualCycle(self):
        """Returns annual cycle information for the variable.
        
        For each site/cell in the variable, computes the mean,
        standard deviation, maximum and minimum values for each month
        of the year across all years.
        
        Returns
        -------
        mean : ILAMB.Variable.Variable
            The annual cycle mean values
        std : ILAMB.Variable.Variable
            The annual cycle standard deviations corresponding to the mean
        mx : ILAMB.Variable.Variable
            The annual cycle maximum values
        mn : ILAMB.Variable.Variable
            The annual cycle minimum values
        """
        if not self.temporal: raise il.NotTemporalVariable()
        assert self.monthly
        begin = np.argmin(self.time[:11]%365)
        end   = begin+int(self.time[begin:].size/12.)*12
        shp   = (-1,12) + self.data.shape[1:]
        v     = self.data[begin:end,...].reshape(shp)
        np.seterr(over='ignore',under='ignore')
        mean  = v.mean(axis=0)
        std   = v.std (axis=0)
        np.seterr(over='raise',under='raise')
        mx    = v.max (axis=0)
        mn    = v.min (axis=0)
        mean  = Variable(data=mean,unit=self.unit,name="annual_cycle_mean_of_%s" % self.name,
                         time=mid_months,lat=self.lat,lon=self.lon,ndata=self.ndata)
        std   = Variable(data=std ,unit=self.unit,name="annual_cycle_std_of_%s" % self.name,
                         time=mid_months,lat=self.lat,lon=self.lon,ndata=self.ndata)
        mx    = Variable(data=mx  ,unit=self.unit,name="annual_cycle_max_of_%s" % self.name,
                         time=mid_months,lat=self.lat,lon=self.lon,ndata=self.ndata)
        mn    = Variable(data=mn  ,unit=self.unit,name="annual_cycle_min_of_%s" % self.name,
                         time=mid_months,lat=self.lat,lon=self.lon,ndata=self.ndata)
        return mean,std,mx,mn

    def timeOfExtrema(self,etype="max"):
        """Returns the time of the specified extrema.
        
        Parameters
        ----------
        etype : str, optional
            The type of extrema to compute, either 'max' or 'min'

        Returns
        -------
        extrema : ILAMB.Variable.Variable
            The times of the extrema computed
        """
        if not self.temporal: raise il.NotTemporalVariable()
        fcn = {"max":np.argmax,"min":np.argmin}
        assert etype in fcn.keys()
        tid  = np.apply_along_axis(fcn[etype],0,self.data)
        mask = False
        if self.data.ndim > 1: mask = np.apply_along_axis(np.all,0,self.data.mask) # mask cells where all data is masked
        data = np.ma.masked_array(self.time[tid],mask=mask)
        return Variable(data=data,unit="d",name="time_of_%s_%s" % (etype,self.name),
                        lat=self.lat,lon=self.lon,area=self.area,ndata=self.ndata)

    def extractDatasites(self,lat,lon):
        """
        UNTESTED
        """
        assert lat.size == lon.size
        if not self.spatial: raise il.NotSpatialVariable()
        ilat = np.apply_along_axis(np.argmin,1,np.abs(lat[:,np.newaxis]-self.lat))
        ilon = np.apply_along_axis(np.argmin,1,np.abs(lon[:,np.newaxis]-self.lon))
        time = self.time
        if self.data.ndim == 2:
            data  = self.data[    ilat,ilon]
            ndata = 1
        else:
            data  = self.data[...,ilat,ilon]
            ndata = lat.size
        return Variable(data=data,unit=self.unit,name=self.name,lat=lat,lon=lon,ndata=ndata,time=time)
        
    def spatialDifference(self,var):
        """Computes the point-wise difference of two spatially defined variables.
        
        If the variable is spatial or site data and is defined on the
        same grid, this routine will simply compute the difference in
        the data arrays. If the variables are spatial but defined on
        separate grids, the routine will interpolate both variables to
        a composed grid via nearest-neighbor interpolation and then
        return the difference.

        Parameters
        ----------
        var : ILAMB.Variable.Variable
            The variable we wish to compare against this variable

        Returns
        -------
        diff : ILAMB.Variable.Variable
            A new variable object representing the difference

        UNTESTED: data sites difference
        """
        def _make_bnds(x):
            bnds       = np.zeros(x.size+1)
            bnds[1:-1] = 0.5*(x[1:]+x[:-1])
            bnds[0]    = max(x[0] -0.5*(x[ 1]-x[ 0]),-180)
            bnds[-1]   = min(x[-1]+0.5*(x[-1]-x[-2]),+180)
            return bnds
        assert Units(var.unit) == Units(self.unit)
        assert self.temporal == False
        assert self.ndata    == var.ndata
        # Perform a check on the spatial grid. If it is the exact same
        # grid, there is no need to interpolate.
        same_grid = False
        try:
            same_grid = np.allclose(self.lat,var.lat)*np.allclose(self.lon,var.lon)
        except:
            pass
        
        if same_grid:
            error     = np.ma.masked_array(var.data-self.data,mask=self.data.mask+var.data.mask)
            diff      = Variable(data=error,unit=var.unit,lat=var.lat,lon=var.lon,ndata=var.ndata,
                                 name="%s_minus_%s" % (var.name,self.name))
        else:
            if not self.spatial: raise il.NotSpatialVariable()
            lat_bnd1 = _make_bnds(self.lat)
            lon_bnd1 = _make_bnds(self.lon)
            lat_bnd2 = _make_bnds( var.lat)
            lon_bnd2 = _make_bnds( var.lon)
            lat_bnd,lon_bnd,lat,lon,error = il.TrueError(lat_bnd1,lon_bnd1,self.lat,self.lon,self.data,
                                                         lat_bnd2,lon_bnd2, var.lat, var.lon, var.data)
            diff = Variable(data=error,unit=var.unit,lat=lat,lon=lon,name="%s_minus_%s" % (var.name,self.name))
        return diff

    def convert(self,unit):
        """
        Parameter
        ---------
        unit : str
            the desired unit to convert to
        
        Return
        ------
        self : ILAMB.Variable.Variable
            this object with its unit converted
        """
        try:
            Units.conform(self.data,Units(self.unit),Units(unit),inplace=True)
            self.unit = unit
        except:
            print "Unit conversion error!!!!",self.name,self.unit,unit
            raise il.UnitConversionError()
        return self
    
    def toNetCDF4(self,dataset):
        """Adds the variable to the specified netCDF4 dataset.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            a dataset into which you wish to save this variable
        """
        def _checkTime(t,dataset):
            """A local function for ensuring the time dimension is saved in the dataset."""
            time_name = "time"
            while True:
                if time_name in dataset.dimensions.keys():
                    if (t.shape    == dataset.variables[time_name][...].shape and
                        np.allclose(t,dataset.variables[time_name][...],atol=0.5*self.dt)): 
                        return time_name
                    else:
                        time_name += "_"
                else:
                    dataset.createDimension(time_name)
                    T = dataset.createVariable(time_name,"double",(time_name))
                    T.setncattr("units","days since 1850-01-01 00:00:00")
                    T.setncattr("calendar","noleap")
                    T.setncattr("axis","T")
                    T.setncattr("long_name","time")
                    T.setncattr("standard_name","time")
                    T[...] = t
                    return time_name

        def _checkLat(lat,dataset):
            """A local function for ensuring the lat dimension is saved in the dataset."""
            lat_name = "lat"
            while True:
                if lat_name in dataset.dimensions.keys():
                    if (lat.shape == dataset.variables[lat_name][...].shape and
                        np.allclose(lat,dataset.variables[lat_name][...])): 
                        return lat_name
                    else:
                        lat_name += "_"
                else:
                    dataset.createDimension(lat_name,size=lat.size)
                    Y = dataset.createVariable(lat_name,"double",(lat_name))
                    Y.setncattr("units","degrees_north")
                    Y.setncattr("axis","Y")
                    Y.setncattr("long_name","latitude")
                    Y.setncattr("standard_name","latitude")
                    Y[...] = lat
                    return lat_name

        def _checkLon(lon,dataset):
            """A local function for ensuring the lon dimension is saved in the dataset."""
            lon_name = "lon"
            while True:
                if lon_name in dataset.dimensions.keys():
                    if (lon.shape == dataset.variables[lon_name][...].shape and
                    np.allclose(lon,dataset.variables[lon_name][...])): 
                        return lon_name
                    else:
                        lon_name += "_"
                else:
                    dataset.createDimension(lon_name,size=lon.size)
                    X = dataset.createVariable(lon_name,"double",(lon_name))
                    X.setncattr("units","degrees_east")
                    X.setncattr("axis","X")
                    X.setncattr("long_name","longitude")
                    X.setncattr("standard_name","longitude")
                    X[...] = lon
                    return lon_name

        def _checkData(ndata,dataset):
            """A local function for ensuring the data dimension is saved in the dataset."""
            data_name = "data"
            while True:
                if data_name in dataset.dimensions.keys():
                    if (ndata == len(dataset.dimensions[data_name])): 
                        return data_name
                    else:
                        data_name += "_"
                else:
                    dataset.createDimension(data_name,size=ndata)
                    return data_name

        dim = []
        if self.temporal:
            dim.append(_checkTime(self.time,dataset))
        if self.ndata is not None:
            dim.append(_checkData(self.ndata,dataset))
            dlat = _checkLat(self.lat,dataset)
            dlon = _checkLon(self.lon,dataset)
        if self.spatial:
            dim.append(_checkLat(self.lat,dataset))
            dim.append(_checkLon(self.lon,dataset))
            
        V = dataset.createVariable(self.name,"double",dim,zlib=True)
        V.setncattr("units",self.unit)
        V.setncattr("max",self.data.max())
        V.setncattr("min",self.data.min())
        V[...] = self.data

    def plot(self,ax,**keywords):
        """Plots the variable on the given matplotlib axis

        The behavior of this routine depends on the type of variable
        specified. If the data is purely temporal, then the plot will
        be a scatter plot versus time of the data. If it is purely
        spatial, then the plot will be a global plot of the data. The
        routine supports multiple keywords although some may not apply
        to the type of plot being generated.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object onto which you wish to plot the variable
        lw : float, optional
            The line width to use when plotting
        alpha : float, optional
            The degree of transparency when plotting, alpha \in [0,1]
        color : str or RGB tuple, optional
            The color to plot with in line plots
        label : str, optional
            The label to appear in the legend of line plots
        vmin : float, optional
            The minimum plotted value
        vmax : float, optional
            The maximum plotted value
        region : str, optional
            The region on which to display a spatial variable
        cmap : str
            The name of the colormap to be used in plotting the spatial variable
        """
        lw     = keywords.get("lw",1.0)
        alpha  = keywords.get("alpha",1.0)
        color  = keywords.get("color","k")
        label  = keywords.get("label",None)
        vmin   = keywords.get("vmin",self.data.min())
        vmax   = keywords.get("vmax",self.data.max())
        region = keywords.get("region","global")
        cmap   = keywords.get("cmap","jet")
        if self.temporal and not self.spatial:
            ticks      = keywords.get("ticks",None)
            ticklabels = keywords.get("ticklabels",None)
            t = self.time/365.+1850
            ax.plot(t,self.data,'-',
                    color=color,lw=lw,alpha=alpha,label=label)
            if ticks      is not None: ax.set_xticks(ticks)
            if ticklabels is not None: ax.set_xticklabels(ticklabels)
        elif not self.temporal and self.spatial:
            ax = post.GlobalPlot(self.lat,self.lon,self.data,ax,
                                 vmin   = vmin  , vmax = vmax,
                                 region = region, cmap = cmap)
        elif not self.temporal and self.ndata is not None:
            from mpl_toolkits.basemap import Basemap
            import matplotlib.colors as colors
            bmap = Basemap(projection='robin',lon_0=0,ax=ax)
            x,y  = bmap(self.lon,self.lat)
            norm = colors.Normalize(vmin,vmax)
            norm = norm(self.data)
            clmp = plt.get_cmap(cmap)
            clrs = clmp(norm)
            size = 35
            ax   = bmap.scatter(x,y,s=size,color=clrs,ax=ax,linewidths=0,cmap=cmap)
            bmap.drawcoastlines(linewidth=0.2,color="darkslategrey")
        return ax

    def interpolate(self,time=None,lat=None,lon=None):
        """Use nearest-neighbor interpolation to interpolate time and space at given values.

        Parameters
        ----------
        time : numpy.ndarray, optional
            Array of times at which to interpolate the variable
        lat : numpy.ndarray, optional
            Array of latitudes at which to interpolate the variable
        lon : numpy.ndarray, optional
            Array of longitudes at which to interpolate the variable

        Returns
        -------
        var : ILAMB.Variable.Variable
            The interpolated variable
        """
        if time is None and lat is None and lon is None: return self
        output_time = self.time if (time is None) else time
        output_lat  = self.lat  if (lat  is None) else lat
        output_lon  = self.lon  if (lon  is None) else lon
        output_area = self.area if (lat is None and lon is None) else None
        
        data = self.data
        if self.spatial and (lat is not None or lon is not None):
            if lat is None: lat = self.lat
            if lon is None: lon = self.lon
            rows  = np.apply_along_axis(np.argmin,1,np.abs(lat[:,np.newaxis]-self.lat))
            cols  = np.apply_along_axis(np.argmin,1,np.abs(lon[:,np.newaxis]-self.lon))
            if self.data.ndim == 2:
                mask  = data.mask[np.ix_(rows,cols)]
                data  = data.data[np.ix_(rows,cols)]
            else:
                mask  = data.mask[np.ix_(range(self.time.size),rows,cols)]
                data  = data.data[np.ix_(range(self.time.size),rows,cols)]
            data  = np.ma.masked_array(data,mask=mask)
        if self.temporal and time is not None:
            times = np.apply_along_axis(np.argmin,1,np.abs(time[:,np.newaxis]-self.time))
            mask  = data.mask
            if mask.size > 1: mask = data.mask[times,...]
            data  = data.data[times,...]
            data  = np.ma.masked_array(data,mask=mask)
        return Variable(data = data, unit = self.unit, name = self.name, ndata = self.ndata,
                        lat  = output_lat,
                        lon  = output_lon,
                        area = output_area,
                        time = output_time)

    def phaseShift(self,var,method="max_of_annual_cycle"):
        """Compute the phase shift between a variable and this variable.
        
        Finds the phase shift as the time between extrema of the
        annual cycles of the variables. Note that if this var and/or
        the given variable are not already annual cycles, they will be
        computed but not returned. The shift will then be returned as 

        Parameters
        ----------
        var : ILAMB.Variable.Variable
            The variable with which we will measure phase shift
        method : str, optional
            The name of the method used to compute the phase shift

        """
        assert method in ["max_of_annual_cycle","min_of_annual_cycle"]
        assert self.temporal == var.temporal
        v1 = self; v2 = var
        if not self.temporal:
            # If the data is not temporal, then the user may have
            # already found the extrema. If the units of the input
            # variable are days, then set the extrema to this data.
            if not (self.unit == "d" and var.unit == "d"): raise il.NotTemporalVariable
            e1 = v1
            e2 = v2
        else:
            # While temporal, the user may have passed in the mean
            # annual cycle as the variable. So if the leading
            # dimension is 12 we assume the variables are already the
            # annual cycles. If not, we compute the cycles and then
            # compute the extrema.
            if self.time.size != 12: v1,junk,junk,junk = self.annualCycle()
            if  var.time.size != 12: v2,junk,junk,junk = var .annualCycle()
            e1 = v1.timeOfExtrema(etype=method[:3])
            e2 = v2.timeOfExtrema(etype=method[:3])
        if e1.spatial:
            shift = e1.spatialDifference(e2)
        else:
            data  = e2.data      - e1.data
            mask  = e1.data.mask + e2.data.mask
            shift = Variable(data=data,unit=e1.unit,ndata=e1.ndata,lat=e1.lat,lon=e1.lon)
        shift.name = "phase_shift_of_%s" % e1.name
        shift.data += (shift.data < -0.5*365.)*365.
        shift.data -= (shift.data > +0.5*365.)*365.
        return shift
    
    def correlation(self,var,ctype,region=None):
        """Compute the correlation between two variables.

        Parameters
        ----------
        var : ILAMB.Variable.Variable
            The variable with which we will compute a correlation
        ctype : str
            The correlation type, one of {"spatial","temporal","spatiotemporal"}
        region : str, optional
            The region over which to perform a spatial correlation

        Notes
        -----
        Need to better think about what correlation means when data
        are masked. The sums ignore the data but then the number of
        items `n' is not constant and should be reduced for masked
        values.

        """
        def _correlation(x,y,axes=None):
            if axes is None: axes = range(x.ndim)
            if type(axes) == int: axes = (int(axes),)
            axes = tuple(axes)
            n    = 1
            for ax in axes: n *= x.shape[ax]
            xbar = x.sum(axis=axes)/n # because np.mean() doesn't take axes which are tuples
            ybar = y.sum(axis=axes)/n
            xy   = (x*y).sum(axis=axes)
            x2   = (x*x).sum(axis=axes)
            y2   = (y*y).sum(axis=axes)
            r    = (xy-n*xbar*ybar)/(np.sqrt(x2-n*xbar*xbar)*np.sqrt(y2-n*ybar*ybar))
            return r
        
        # checks on data consistency
        assert region is None
        assert self.data.shape == var.data.shape
        assert ctype in ["spatial","temporal","spatiotemporal"]

        # determine arguments for functions
        axes      = None
        out_time  = None
        out_lat   = None
        out_lon   = None
        out_area  = None
        out_ndata = None
        if ctype == "temporal":
            axes = 0
            if self.spatial:
                out_lat   = self.lat
                out_lon   = self.lon
                out_area  = self.area
            elif self.ndata:
                out_ndata = self.ndata
        elif ctype == "spatial":
            if self.spatial:  axes     = range(self.data.ndim)[-2:]
            if self.ndata:    axes     = self.data.ndim-1
            if self.temporal: out_time = self.time
        r = _correlation(self.data,var.data,axes=axes)
        return Variable(data=r,unit="-",
                        name="%s_correlation_of_%s" % (ctype,self.name),
                        time=out_time,ndata=out_ndata,
                        lat=out_lat,lon=out_lon,area=out_area)
    
    def bias(self,var):
        """UNTESTED

        if not temporal data is passed in, assume that these are the
        means and simply return the difference.

        """
        # If not a temporal variable, then we assume that the user is
        # passing in mean data and return the difference.
        if not self.temporal:
            assert self.temporal == var.temporal
            bias = self.spatialDifference(var)
            bias.name = "bias_of_%s" % self.name
            return bias
        if self.spatial:
            # If the data is spatial, then we interpolate it on a
            # common grid and take the difference.
            lat,lon  = ComposeSpatialGrids(self,var)
            self_int = self.interpolate(lat=lat,lon=lon)
            var_int  = var .interpolate(lat=lat,lon=lon)
            data     = var_int.data-self_int.data
            mask     = var_int.data.mask+self_int.data.mask
        elif (self.ndata or self.time.size == self.data.size):
            # If the data are at sites, then take the difference
            data = var.data.data-self.data.data
            mask = var.data.mask+self.data.mask
        else:
            raise il.NotSpatialVariable("Cannot take bias of scalars")
        # Finally we return the temporal mean of the difference
        bias = Variable(data=np.ma.masked_array(data,mask=mask),
                        name="bias_of_%s" % self.name,time=self.time,
                        unit=self.unit,ndata=self.ndata,
                        lat=self.lat,lon=self.lon,area=self.area).integrateInTime(mean=True)
        bias.name = bias.name.replace("_integrated_over_time_and_divided_by_time_period","")
        return bias
    
    def rmse(self,var):
        """
        UNTESTED
        """
        if not self.temporal or not var.temporal: raise il.NotTemporalVariable
        assert self.ndata == var.ndata
        if self.spatial:
            # If the data is spatial, then we interpolate it on a
            # common grid and take the difference.
            lat,lon  = ComposeSpatialGrids(self,var)
            self_int = self.interpolate(lat=lat,lon=lon)
            var_int  = var .interpolate(lat=lat,lon=lon)
            np.seterr(over='ignore',under='ignore')
            data     = (var_int.data-self_int.data)**2
            np.seterr(over='raise',under='raise')
            mask     = var_int.data.mask+self_int.data.mask
        elif self.ndata:
            # If the data are at sites, then take the difference
            np.seterr(over='ignore',under='ignore')
            data = (var.data.data-self.data.data)**2
            np.seterr(over='raise',under='raise')
            mask = var.data.mask+self.data.mask
            lat,lon = var.lat,var.lon
        else:
            raise il.NotSpatialVariable("Cannot take rmse of scalars")
        # Finally we return the temporal mean of the difference squared
        rmse = Variable(data=np.ma.masked_array(data,mask=mask),
                        name="rmse_of_%s" % self.name,time=self.time,
                        unit=self.unit,ndata=self.ndata,
                        lat=lat,lon=lon).integrateInTime(mean=True)
        rmse.name = rmse.name.replace("_integrated_over_time_and_divided_by_time_period","")
        rmse.data = np.sqrt(rmse.data)
        return rmse
    
    def interannualVariability(self):
        """
        UNTESTED
        """
        if not self.temporal: raise il.NotTemporalVariable
        np.seterr(over='ignore',under='ignore')
        data = self.data.std(axis=0)
        np.seterr(over='raise',under='raise')
        return Variable(data=data,
                        name="iav_of_%s" % self.name,
                        unit=self.unit,ndata=self.ndata,
                        lat=self.lat,lon=self.lon,area=self.area)

    def spatialDistribution(self,var,region="global"):
        """
        UNTESTED
        """
        assert self.temporal == var.temporal == False

        lats,lons = ILAMBregions[region]
        
        # First compute the observational spatial/site standard deviation
        rem_mask0  = np.copy(self.data.mask)
        if self.spatial:
            self.data.mask += (np.outer((self.lat>lats[0])*(self.lat<lats[1]),
                                        (self.lon>lons[0])*(self.lon<lons[1]))==0)
        else:
            self.data.mask += (        ((self.lat>lats[0])*(self.lat<lats[1])*
                                        (self.lon>lons[0])*(self.lon<lons[1]))==0)
            
        np.seterr(over='ignore',under='ignore')
        std0 = self.data.std()
        np.seterr(over='raise',under='raise')

        # Next compute the model spatial/site standard deviation
        rem_mask  = np.copy(var.data.mask)
        if var.spatial:
            var.data.mask += (np.outer((var.lat>lats[0])*(var.lat<lats[1]),
                                       (var.lon>lons[0])*(var.lon<lons[1]))==0)
        else:
            var.data.mask += (        ((var.lat>lats[0])*(var.lat<lats[1])*
                                       (var.lon>lons[0])*(var.lon<lons[1]))==0)
        np.seterr(over='ignore',under='ignore')
        std = var.data.std()
        np.seterr(over='raise',under='raise')

        # Interpolate to new grid for correlation
        if self.spatial:
            lat,lon  = ComposeSpatialGrids(self,var)
            lat      = lat[(lat>=lats[0])*(lat<=lats[1])]
            lon      = lon[(lon>=lons[0])*(lon<=lons[1])]
            self_int = self.interpolate(lat=lat,lon=lon)
            var_int  = var .interpolate(lat=lat,lon=lon)
        else:
            self_int = self
            var_int  = var
        R   = self_int.correlation(var_int,ctype="spatial") # add regions

        # Restore masks
        self.data.mask = rem_mask0
        var.data.mask  = rem_mask

        # Put together scores, we clip the standard deviation of both
        # variables at the same small amount, meant to avoid division
        # by zero errors.
        R0    = 1.0
        std0  = std0.clip(1e-12)
        std   = std .clip(1e-12)
        std  /= std0
        score = 4.0*(1.0+R.data)/((std+1.0/std)**2 *(1.0+R0))
        std   = Variable(data=std  ,name="normalized_spatial_std_of_%s_over_%s" % (self.name,region),unit="-")
        score = Variable(data=score,name="spatial_distribution_score_of_%s_over_%s" % (self.name,region),unit="-")
        return std,R,score
        
def Score(var,normalizer):
    """
    UNTESTED
    """
    score      = deepcopy(var)
    np.seterr(over='ignore',under='ignore')
    score.data = np.exp(-np.abs(score.data/normalizer.data))
    np.seterr(over='raise',under='raise')
    score.name = score.name.replace("bias","bias_score")
    score.name = score.name.replace("rmse","rmse_score")
    score.name = score.name.replace("iav" ,"iav_score")
    score.unit = "-"
    return score

def ComposeSpatialGrids(var1,var2):
    """Creates a grid which conforms the boundaries of both variables.
    
    This routine takes the union of the latitude and longitude
    cell boundaries of both grids and returns a new set of
    latitudes and longitudes which represent cell centers of the
    new grid.
    
    Parameters
    ----------
    var1,var2 : ILAMB.Variable.Variable
        The two variables for which we wish to find a common grid
    
    Returns
    -------
    lat : numpy.ndarray
        a 1D array of latitudes of cell centroids
    lon : numpy.ndarray
        a 1D array of longitudes of cell centroids
    """
    if not var1.spatial: il.NotSpatialVariable()
    if not var2.spatial: il.NotSpatialVariable()
    def _make_bnds(x):
        bnds       = np.zeros(x.size+1)
        bnds[1:-1] = 0.5*(x[1:]+x[:-1])
        bnds[ 0]   = max(x[ 0]-0.5*(x[ 1]-x[ 0]),-180)
        bnds[-1]   = min(x[-1]+0.5*(x[-1]-x[-2]),+180)
        return bnds
    lat1_bnd = _make_bnds(var1.lat)
    lon1_bnd = _make_bnds(var1.lon)
    lat2_bnd = _make_bnds(var2.lat)
    lon2_bnd = _make_bnds(var2.lon)
    lat_bnd  = np.hstack((lat1_bnd,lat2_bnd)); lat_bnd.sort(); lat_bnd = np.unique(lat_bnd)
    lon_bnd  = np.hstack((lon1_bnd,lon2_bnd)); lon_bnd.sort(); lon_bnd = np.unique(lon_bnd)
    lat      = 0.5*(lat_bnd[1:]+lat_bnd[:-1])
    lon      = 0.5*(lon_bnd[1:]+lon_bnd[:-1])
    return lat,lon

def ScoreSeasonalCycle(phase_shift):
    """
    UNTESTED
    """
    return Variable(data  = (1+np.cos(np.abs(phase_shift.data)/365*2*np.pi))*0.5,
                    unit  = "-",
                    name  = phase_shift.name.replace("phase_shift","phase_shift_score"),
                    ndata = phase_shift.ndata,
                    lat   = phase_shift.lat,
                    lon   = phase_shift.lon,
                    area  = phase_shift.area)

def AnalysisFluxrate(obs,mod,regions=['global'],dataset=None,benchmark_dataset=None,space_mean=True,table_unit=None,plot_unit=None):
    """
    UNTESTED
    """
    assert Units(obs.unit) == Units(mod.unit)
    spatial = obs.spatial
    
    # Integrate in time and divide through by the time period. We need
    # these maps/sites for plotting.
    obs_timeint = obs.integrateInTime(mean=True)
    mod_timeint = mod.integrateInTime(mean=True)
        
    # Compute maps of the bias and rmse. We will use these variables
    # later in the regional analysis to obtain means over individual
    # regions. Note that since we have already taken a temporal
    # average of the variables, the bias() function can reuse this
    # data and avoid extra averaging. We also compute maps of the
    # scores, each normalized in their respective manner.
    bias_map = obs_timeint.bias(mod_timeint)
    rmse_map = obs        .rmse(mod)
    if spatial:
        period_mean      = obs_timeint.integrateInSpace(mean=True)
        bias_score_map   = Score(bias_map,period_mean)
        rmse_mean        = rmse_map.integrateInSpace(mean=True)
        rmse_score_map   = Score(rmse_map,rmse_mean)
    else:
        period_mean,junk = obs_timeint.siteStats()
        bias_score_map   = Score(bias_map,period_mean)
        rmse_mean,junk   = rmse_map.siteStats()
        rmse_score_map   = Score(rmse_map,rmse_mean)

    # Compute maps of the phase shift. First we compute the mean
    # annual cycle over space/sites and then find the time where the
    # maxmimum occurs.
    obs_cycle,junk,junk,junk = obs.annualCycle()
    mod_cycle,junk,junk,junk = mod.annualCycle()
    obs_maxt_map    = obs_cycle.timeOfExtrema(etype="max")
    mod_maxt_map    = mod_cycle.timeOfExtrema(etype="max")
    shift_map       = obs_maxt_map.phaseShift(mod_maxt_map)
    shift_score_map = ScoreSeasonalCycle(shift_map)

    # Compute a map of interannual variability score.
    obs_iav_map   = obs.interannualVariability()
    mod_iav_map   = mod.interannualVariability()
    iav_score_map = obs_iav_map.spatialDifference(mod_iav_map)
    iav_score_map.name = obs_iav_map.name
    if spatial:
        obs_iav_mean      = obs_iav_map.integrateInSpace(mean=True)
    else:
        obs_iav_mean,junk = obs_iav_map.siteStats()
    iav_score_map = Score(iav_score_map,obs_iav_mean)
    
    # Perform analysis over regions. We will store these in
    # dictionaries of variables where the keys are the region names.
    obs_period_mean = {}
    obs_mean_cycle  = {}
    obs_spaceint    = {}
    mod_period_mean = {}
    mod_mean_cycle  = {}
    mod_spaceint    = {}
    bias  = {}; bias_score  = {}
    rmse  = {}; rmse_score  = {}
    shift = {}; shift_score = {}
    iav_score = {};
    std = {}; R = {}; sd_score = {}
    for region in regions:
        
        if spatial:

            # Compute the scalar integral over the specified region.
            obs_period_mean[region] = obs_timeint    .integrateInSpace(region=region,mean=space_mean) ##
            obs_mean_cycle [region] = obs_cycle      .integrateInSpace(region=region,mean=True)
            obs_spaceint   [region] = obs            .integrateInSpace(region=region,mean=True)
            mod_period_mean[region] = mod_timeint    .integrateInSpace(region=region,mean=space_mean) ##
            
            # Compute the scalar means over the specified region.
            bias           [region] = bias_map       .integrateInSpace(region=region,mean=space_mean) ##
            rmse           [region] = rmse_map       .integrateInSpace(region=region,mean=space_mean) ##
            bias_score     [region] = bias_score_map .integrateInSpace(region=region,mean=True)
            rmse_score     [region] = rmse_score_map .integrateInSpace(region=region,mean=True)
            shift          [region] = shift_map      .integrateInSpace(region=region,mean=True)
            shift_score    [region] = shift_score_map.integrateInSpace(region=region,mean=True)
            iav_score      [region] = iav_score_map  .integrateInSpace(region=region,mean=True)
            mod_mean_cycle [region] = mod_cycle      .integrateInSpace(region=region,mean=True)
            mod_spaceint   [region] = mod            .integrateInSpace(region=region,mean=True)
            
        else:

            # We need to check if there are datasites in this
            # region. If not, we will just skip the region.
            lats,lons = ILAMBregions[region]
            if ((obs.lat>lats[0])*(obs.lat<lats[1])*(obs.lon>lons[0])*(obs.lon<lons[1])).sum() == 0: continue
            
            # Compute the scalar period mean over sites in the specified region.
            obs_period_mean[region],junk = obs_timeint    .siteStats(region=region)
            obs_mean_cycle [region],junk = obs_cycle      .siteStats(region=region)
            obs_spaceint   [region],junk = obs            .siteStats(region=region)
            mod_period_mean[region],junk = mod_timeint    .siteStats(region=region)
            bias           [region],junk = bias_map       .siteStats(region=region)
            rmse           [region],junk = rmse_map       .siteStats(region=region)
            bias_score     [region],junk = bias_score_map .siteStats(region=region)
            rmse_score     [region],junk = rmse_score_map .siteStats(region=region)
            shift          [region],junk = shift_map      .siteStats(region=region)
            shift_score    [region],junk = shift_score_map.siteStats(region=region)
            iav_score      [region],junk = iav_score_map  .siteStats(region=region)
            mod_mean_cycle [region],junk = mod_cycle      .siteStats(region=region)
            mod_spaceint   [region],junk = mod            .siteStats(region=region)

        # Compute the spatial variability.
        std[region],R[region],sd_score[region] = obs_timeint.spatialDistribution(mod_timeint,region=region)
        
        # Change variable names to make things easier to parse later.
        obs_period_mean[region].name = "period_mean_of_%s_over_%s" % (obs.name,region)
        obs_mean_cycle [region].name = "cycle_of_%s_over_%s"       % (obs.name,region)
        obs_spaceint   [region].name = "spaceint_of_%s_over_%s"    % (obs.name,region)
        mod_period_mean[region].name = "period_mean_of_%s_over_%s" % (obs.name,region)
        bias           [region].name = "bias_of_%s_over_%s"        % (obs.name,region)
        rmse           [region].name = "rmse_of_%s_over_%s"        % (obs.name,region)
        shift          [region].name = "shift_of_%s_over_%s"       % (obs.name,region)
        bias_score     [region].name = "bias_score_of_%s_over_%s"  % (obs.name,region)
        rmse_score     [region].name = "rmse_score_of_%s_over_%s"  % (obs.name,region)
        shift_score    [region].name = "shift_score_of_%s_over_%s" % (obs.name,region)
        iav_score      [region].name = "iav_score_of_%s_over_%s"   % (obs.name,region)
        sd_score       [region].name = "sd_score_of_%s_over_%s"    % (obs.name,region)
        mod_mean_cycle [region].name = "cycle_of_%s_over_%s"       % (obs.name,region)
        mod_spaceint   [region].name = "spaceint_of_%s_over_%s"    % (obs.name,region)
        std            [region].name = "std_of_%s_over_%s"         % (obs.name,region)
        R              [region].name = "corr_of_%s_over_%s"        % (obs.name,region)
        
    # More variable name changes
    obs_timeint.name  = "timeint_of_%s"   % obs.name
    mod_timeint.name  = "timeint_of_%s"   % obs.name
    bias_map.name     = "bias_map_of_%s"  % obs.name
    obs_maxt_map.name = "phase_map_of_%s" % obs.name
    mod_maxt_map.name = "phase_map_of_%s" % obs.name
    shift_map.name    = "shift_map_of_%s" % obs.name

    # Unit conversions
    if table_unit is not None:
        for var in [obs_period_mean,mod_period_mean,bias,rmse]:
            if type(var) == type({}):
                for key in var.keys(): var[key].convert(table_unit)
            else:
                var.convert(plot_unit)
    if plot_unit is not None:
        for var in [mod_timeint,obs_timeint,bias_map,mod_mean_cycle,mod_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].convert(plot_unit)
            else:
                var.convert(plot_unit)

    # Optionally dump results to a NetCDF file
    if dataset is not None:
        for var in [mod_period_mean,bias,rmse,shift,bias_score,rmse_score,shift_score,iav_score,sd_score,
                    mod_timeint,bias_map,mod_maxt_map,shift_map,std,R,
                    mod_mean_cycle,mod_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(dataset)
            else:
                var.toNetCDF4(dataset)
    if benchmark_dataset is not None:
        for var in [obs_period_mean,obs_timeint,obs_maxt_map,obs_mean_cycle,obs_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(benchmark_dataset)
            else:
                var.toNetCDF4(benchmark_dataset)


def AnalysisRelationship(dep_var,ind_var,dataset,rname,**keywords):
    r"""
    
    
    """    
    def _extractMaxTemporalOverlap(v1,v2):  # should move to ilamblib?
        t0 = max(v1.time.min(),v2.time.min())
        tf = min(v1.time.max(),v2.time.max())
        for v in [v1,v2]:
            begin = np.argmin(np.abs(v.time-t0))
            end   = np.argmin(np.abs(v.time-tf))+1
            v.time = v.time[begin:end]
            v.data = v.data[begin:end,...]
        mask = v1.data.mask + v2.data.mask
        v1 = v1.data[mask==0].flatten()
        v2 = v2.data[mask==0].flatten()
        return v1,v2
    
    # convert to plot units
    dep_plot_unit = keywords.get("dep_plot_unit",dep_var.unit)
    ind_plot_unit = keywords.get("ind_plot_unit",ind_var.unit)    
    if dep_plot_unit is not None: dep_var.convert(dep_plot_unit)
    if ind_plot_unit is not None: ind_var.convert(ind_plot_unit)

    # if the variables are temporal, we need to get period means
    if dep_var.temporal: dep_var = dep_var.integrateInTime(mean=True)
    if ind_var.temporal: ind_var = ind_var.integrateInTime(mean=True)

    mask = dep_var.data.mask + ind_var.data.mask
    x    = ind_var.data[mask==0].flatten()
    y    = dep_var.data[mask==0].flatten()

    # Scott's rule (doi:10.1093/biomet/66.3.605) assumes that the
    # data is normally distributed
    #dx = 3.5*x.std()*np.power(x.size,-1./3.)
    #dy = 3.5*y.std()*np.power(y.size,-1./3.)

    # Compute 2D histogram, normalized by number of datapoints
    #Nx = int(round((x.max()-x.min())/dx,0))
    #Ny = int(round((y.max()-y.min())/dy,0))
    Nx = 50
    Ny = 50
    counts,xedges,yedges = np.histogram2d(x,y,[Nx,Ny])
    counts = np.ma.masked_values(counts,0)/float(x.size)

    # Compute mean relationship function
    nudge = 1e-15
    xedges[0] -= nudge; xedges[-1] += nudge
    xbins = np.digitize(x,xedges)-1
    xmean = []
    ymean = []
    ystd  = []
    for i in range(xedges.size-1):
        ind = (xbins==i)
        if ind.sum() < max(x.size*1e-4,10): continue
        xtmp = x[ind]
        ytmp = y[ind]
        xmean.append(xtmp.mean())
        ymean.append(ytmp.mean())
        try:        
            ystd.append(ytmp. std())
        except:
            ystd.append(np.sqrt((((ytmp-ytmp.mean())**2).sum())/float(ytmp.size-1)))
    xmean = np.asarray(xmean)
    ymean = np.asarray(ymean)
    ystd  = np.asarray(ystd )

    # Write histogram to the dataset
    grp = dataset.createGroup("relationship_%s" % rname)
    grp.createDimension("nv",size=2)
    for d_bnd,dname in zip([xedges,yedges],["ind","dep"]):
        d = 0.5*(d_bnd[:-1]+d_bnd[1:])
        dbname = dname + "_bnd"
        grp.createDimension(dname,size=d.size)
        D = grp.createVariable(dname,"double",(dname))
        D.setncattr("standard_name",dname)
        D.setncattr("bounds",dbname)
        D[...] = d
        B = grp.createVariable(dbname,"double",(dname,"nv"))
        B.setncattr("standard_name",dbname)
        B[:,0] = d_bnd[:-1]
        B[:,1] = d_bnd[+1:]
    H = grp.createVariable("histogram","double",("ind","dep"))
    H.setncattr("standard_name","histogram")
    H[...] = counts
        
    # Write relationship to the dataset
    grp.createDimension("ndata",size=xmean.size)
    X = grp.createVariable("ind_mean","double",("ndata"))
    X.setncattr("unit",ind_plot_unit)
    M = grp.createVariable("dep_mean","double",("ndata"))
    M.setncattr("unit",dep_plot_unit)
    S = grp.createVariable("dep_std" ,"double",("ndata"))
    X[...] = xmean
    M[...] = ymean
    S[...] = ystd
