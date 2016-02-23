from constants import spd,dpy,mid_months,regions as ILAMBregions
from pylab import get_cmap
from cfunits import Units
import ilamblib as il
import Post as post
import numpy as np

class Variable:
    r"""A class for managing variables and their analysis.

    There are two ways to create a Variable object. Because python
    does not support multiple constructors, we will use keyword
    arguments so that the users intent may be captured. The first way
    to specify a Variable is by loading a netCDF4 file. You can
    achieve this by specifying the 'filename' and 'variable_name'
    keywords. The second way is to use the remaining keyword arguments
    to specify data arrays directly. If you use the second way, you
    must specify the keywords 'data' and 'unit'. The rest are truly
    optional and depend on the nature of your data.

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
    time_bnds : numpy.ndarray, optional
        a 2D array of time bounds in days since 1850-01-01 00:00:00
    lat : numpy.ndarray, optional
        a 1D array of latitudes of cell centroids
    lon : numpy.ndarray, optional
        a 1D array of longitudes of cell centroids
    area : numpy.ndarray, optional
        a 2D array of the cell areas
    ndata : int, optional
        number of data sites this data represents
    alternate_vars : list of str, optional
        a list of alternate acceptable variable names

    Examples
    --------

    You can initiate a Variable by specifying the data directly.

    >>> lat = np.linspace(- 90, 90, 91)
    >>> lon = np.linspace(-180,180,181)
    >>> data = np.random.rand(91,181)
    >>> v = Variable(name="some_variable",unit="some_unit",lat=lat,lon=lon,data=data)

    Or you can initiate a variable by extracting a specific field from a netCDF4 file.

    >>> v = Variable(filename="some_netcdf_file.nc",variable_name="name_of_var_to_extract")

    """
    def __init__(self,**keywords):
        r"""Constructor for the variable class by specifying the data arrays.
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
            time_bnds = keywords.get("time_bnds" ,None)
            lat   = keywords.get("lat"  ,None)
            lon   = keywords.get("lon"  ,None)
            ndata = keywords.get("ndata",None)
            assert data is not None
            assert unit is not None
        else:
            assert variable_name is not None
            data,unit,name,time,time_bnds,lat,lon,ndata = il.FromNetCDF4(filename,variable_name,alternate_vars)
                            
        if not np.ma.isMaskedArray(data): data = np.ma.masked_array(data)
        self.data  = data 
        self.ndata = ndata
        self.unit  = unit
        self.name  = name
        
        # Handle time data
        self.time      = time      # time data
        self.time_bnds = time_bnds # bounds on time
        self.temporal  = False     # flag for temporal data
        self.dt        = 0.        # mean temporal spacing
        self.monthly   = False     # flag for monthly means
        if time is not None: 
            self.temporal = True
            if self.time_bnds is None:
                self.time_bnds = np.zeros((2,time.size))
                self.time_bnds[0,+1:] = 0.5*(time[:-1]+time[+1:])
                self.time_bnds[1,:-1] = 0.5*(time[:-1]+time[+1:])
                if time.size > 1:
                    self.time_bnds[0,  0] = time[ 0] - 0.5*(time[ 1]-time[ 0])
                    self.time_bnds[1, -1] = time[-1] + 0.5*(time[-1]-time[-2])
            self.dt = (self.time_bnds[1,:]-self.time_bnds[0,:]).mean()
            if np.allclose(self.dt,30,atol=3): self.monthly = True
            assert (2*self.time.size) == (self.time_bnds.size)
            
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
        r"""Integrates the variable over time period.

        Uses nodal integration to integrate to approximate 

        .. math:: \int_{t_0}^{t_f} v(t,\dots)\ dt

        The arguments of the integrand reflect that while it must be
        at least defined in time, the remaining arguments are
        flexible. If :math:`t_0` or :math:`t_f` are not specified, the
        variable will be integrated over the extent of its time
        domain. If the mean function value over time is desired, this
        routine will approximate

        .. math:: \frac{1}{t_f-t_0} \int_{t_0}^{t_f} v(t,\dots)\ dt
        
        again by nodal integration. The amount of time which we divide
        by is the non-masked amount of time. This means that if a
        function has some values masked or marked as invalid, we do
        not penalize the average value by including this as a time at
        which data is expected.

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
        t0   = keywords.get("t0",self.time_bnds[0,:].min())
        tf   = keywords.get("tf",self.time_bnds[1,:].max())
        mean = keywords.get("mean",False)
        
        # find which time bounds are included even partially in the interval [t0,tf]
        time_bnds = np.copy(self.time_bnds)
        ind       = np.where((t0<time_bnds[1,:])*(tf>time_bnds[0,:]))
        time_bnds[0,(t0>time_bnds[0,:])*(t0<time_bnds[1,:])] = t0
        time_bnds[1,(tf>time_bnds[0,:])*(tf<time_bnds[1,:])] = tf
        time_bnds = time_bnds[:,ind]
        dt        = (time_bnds[1,:]-time_bnds[0,:])[0,:]

        # now expand this dt to the other dimensions of the data array (i.e. space or datasites)
        for i in range(self.data.ndim-1): dt = np.expand_dims(dt,axis=-1)

        # approximate the integral by nodal integration (rectangle rule)
        integral = (self.data[ind]*dt).sum(axis=0)
        
        # the integrated array should be masked where *all* data in time was previously masked
        mask = False
        if self.data.ndim > 1 and self.data.mask.size > 1:
            mask = np.apply_along_axis(np.all,0,self.data.mask[ind])
        integral = np.ma.masked_array(integral,mask=mask,copy=False)
        
        # handle units
        unit = Units(self.unit)
        name = self.name + "_integrated_over_time"
        
        if mean:
            
            # divide thru by the non-masked amount of time, the units
            # can remain as input because we integrate over time and
            # then divide by the time interval in the same units
            name     += "_and_divided_by_time_period"
            if self.data.mask.size > 1:
                dt = (dt*(self.data.mask[ind]==0)).sum(axis=0)
            else:
                dt = dt.sum(axis=0)   
            np.seterr(over='ignore',under='ignore')
            integral /= dt
            np.seterr(over='raise' ,under='raise' )
            
        else:

            # if not a mean, we need to potentially handle unit conversions
            unit0    = Units("d")*unit
            unit     = Units(unit0.formatted().split()[-1])
            integral = Units.conform(integral,unit0,unit)
        
        return Variable(data  = integral,
                        unit  = unit.units,
                        name  = name,
                        lat   = self.lat,
                        lon   = self.lon,
                        area  = self.area,
                        ndata = self.ndata)

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
            if mean: integral /= np.ma.masked_array(self.area,mask=self.data.mask).sum()
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

        # handle the unit
        unit = Units(self.unit)
        if mean:

            # we have already divided thru by the non-masked area in
            # units of m^2, which are the same units of the integrand.
            name += "_and_divided_by_area"
            
        else:

            # if not a mean, we need to potentially handle unit conversions
            unit0    = Units("m2")*unit
            unit     = Units(unit0.formatted().split()[-1])
            integral = Units.conform(integral,unit0,unit)
            
        return Variable(data      = np.ma.masked_array(integral),
                        unit      = unit.units,
                        time      = self.time,
                        time_bnds = self.time_bnds,
                        name      = name)

    def siteStats(self,region=None):
        """Computes the mean and standard deviation of the variable over all data sites.

        Parameters
        ----------
        region : str, optional
            name of the region overwhich you wish to include stats.
        
        Returns
        -------
        mean : ILAMB.Variable.Variable
            a Variable instace with the mean values
        std : ILAMB.Variable.Variable
            a Variable instace with the standard deviation values

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
        mean = Variable(data      = mean,
                        unit      = self.unit,
                        time      = self.time,
                        time_bnds = self.time_bnds,
                        name      = "mean_%s%s" % (self.name,rname))
        std  = Variable(data      = std ,
                        unit      = self.unit,
                        time      = self.time,
                        time_bnds = self.time_bnds,
                        name      = "std_%s%s" % (self.name,rname))
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
        """Extracts a variable at a set of latitude and longitude.

        Parameters
        ----------
        lat : numpy.ndarray
            an array with the latitude values, must be same size as the longitude values
        lon : numpy.ndarray
            an array with the longitude values, must be same size as the latitude values

        Returns
        -------
        extracted : ILAMB.Variable.Variable
            The extracted variables
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
        return Variable(data      = data,
                        unit      = self.unit,
                        name      = self.name,
                        lat       = lat,
                        lon       = lon,
                        ndata     = ndata,
                        time      = time,
                        time_bnds = self.time_bnds)
        
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

    def convert(self,unit,density=998.2):
        """Convert the variable to a given unit.

        Parameter
        ---------
        unit : str
            the desired converted unit
        density : float, optional
            the mass density in [kg m-3] to use when converting linear
            rates to area density rates
        
        Return
        ------
        self : ILAMB.Variable.Variable
            this object with its unit converted

        """
        src_unit  = Units(self.unit)
        tar_unit  = Units(     unit)
        mask      = self.data.mask

        # standard units convert
        try:
            self.data = Units.conform(self.data,src_unit,tar_unit)
            self.data = np.ma.masked_array(self.data,mask=mask)
            self.unit = unit
            return self
        except:
            pass
        
        # assuming substance is water, try to convert (L / T) * (M / L^3) == (M / L^2 / T)
        try:
            linear_rate       = Units("m s-1")
            area_density_rate = Units("kg m-2 s-1")
            mass_density      = Units("kg m-3")
            if not (src_unit.equivalent(linear_rate) and
                    tar_unit.equivalent(area_density_rate)): raise
            np.seterr(over='ignore',under='ignore')
            self.data *= density
            np.seterr(over='raise',under='raise')
            self.data  = Units.conform(self.data,src_unit*mass_density,tar_unit)
            self.data  = np.ma.masked_array(self.data,mask=mask)
            self.unit  = unit
            return self
        except:
            pass

        # assuming substance is water, try to convert (M / L^2 / T) / (M / L^3) == (L / T) 
        try:
            linear_rate       = Units("m s-1")
            area_density_rate = Units("kg m-2 s-1")
            mass_density      = Units("kg m-3")
            if not (src_unit.equivalent(area_density_rate) and
                    tar_unit.equivalent(linear_rate)): raise
            np.seterr(over='ignore',under='ignore')
            self.data /= density
            np.seterr(over='raise',under='raise')
            self.data  = Units.conform(self.data,src_unit/mass_density,tar_unit)
            self.data  = np.ma.masked_array(self.data,mask=mask)
            self.unit  = unit
            return self
        except:
            pass
        
        # if no conversions pass, then raise this exception
        print "var_name = %s, var_unit = %s, convert_unit = %s " % (self.name,self.unit,unit)
        raise il.UnitConversionError()

    
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
            clmp = get_cmap(cmap)
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
        out_time_bnds = None
        if out_time is not None: out_time_bnds = self.time_bnds
        r = _correlation(self.data,var.data,axes=axes)
        return Variable(data=r,unit="-",
                        name="%s_correlation_of_%s" % (ctype,self.name),
                        time=out_time,time_bnds=out_time_bnds,ndata=out_ndata,
                        lat=out_lat,lon=out_lon,area=out_area)
    
    def bias(self,var):
        """rethink this

        if not temporal data is passed in, assume that these are the
        means and simply return the difference.

        """
        # If not a temporal variable, then we assume that the user is
        # passing in mean data and return the difference.
        lat,lon,area = self.lat,self.lon,self.area
        if not self.temporal:
            assert self.temporal == var.temporal
            bias = self.spatialDifference(var)
            bias.name = "bias_of_%s" % self.name
            return bias
        if self.spatial:
            # If the data is spatial, then we interpolate it on a
            # common grid and take the difference.

            same_grid = False
            try:
                same_grid = np.allclose(self.lat,var.lat)*np.allclose(self.lon,var.lon)
            except:
                pass
            if not same_grid:
                lat,lon  = il.ComposeSpatialGrids(self,var)
                area     = None
                self_int = self.interpolate(lat=lat,lon=lon)
                var_int  = var .interpolate(lat=lat,lon=lon)
                data     = var_int.data-self_int.data
                mask     = var_int.data.mask+self_int.data.mask
            else:
                data     = var.data     -self.data
                mask     = var.data.mask+self.data.mask

        elif (self.ndata or self.time.size == self.data.size):
            # If the data are at sites, then take the difference
            data = var.data.data-self.data.data
            mask = var.data.mask+self.data.mask
        else:
            raise il.NotSpatialVariable("Cannot take bias of scalars")
        # Finally we return the temporal mean of the difference
        bias = Variable(data=np.ma.masked_array(data,mask=mask),
                        name="bias_of_%s" % self.name,time=self.time,time_bnds=self.time_bnds,
                        unit=self.unit,ndata=self.ndata,
                        lat=lat,lon=lon,area=area).integrateInTime(mean=True)
        bias.name = bias.name.replace("_integrated_over_time_and_divided_by_time_period","")
        return bias
    
    def rmse(self,var):
        """
        UNTESTED
        """
        # If not a temporal variable, then we assume that the user is
        # passing in mean data and return the difference.
        lat,lon,area = self.lat,self.lon,self.area
        if not self.temporal:
            assert self.temporal == var.temporal
            rmse = self.spatialDifference(var)
            rmse.name = "rmse_of_%s" % self.name
            return rmse
        if self.spatial:
            # If the data is spatial, then we interpolate it on a
            # common grid and take the difference.
            same_grid = False
            try:
                same_grid = np.allclose(self.lat,var.lat)*np.allclose(self.lon,var.lon)
            except:
                pass
            if not same_grid:
                lat,lon  = il.ComposeSpatialGrids(self,var)
                area     = None
                self_int = self.interpolate(lat=lat,lon=lon)
                var_int  = var .interpolate(lat=lat,lon=lon)
                data     = var_int.data-self_int.data
                mask     = var_int.data.mask+self_int.data.mask
            else:
                data     = var.data     -self.data
                mask     = var.data.mask+self.data.mask
        elif (self.ndata or self.time.size == self.data.size):
            # If the data are at sites, then take the difference
            data = var.data.data-self.data.data
            mask = var.data.mask+self.data.mask
        else:
            raise il.NotSpatialVariable("Cannot take rmse of scalars")
        # Finally we return the temporal mean of the difference
        np.seterr(over='ignore',under='ignore')
        data *= data
        np.seterr(over='raise',under='raise')
        rmse = Variable(data=np.ma.masked_array(data,mask=mask),
                        name="rmse_of_%s" % self.name,time=self.time,time_bnds=self.time_bnds,
                        unit=self.unit,ndata=self.ndata,
                        lat=lat,lon=lon,area=area).integrateInTime(mean=True)
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
            lat,lon  = il.ComposeSpatialGrids(self,var)
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

    def coarsenInTime(self,intervals,window=0.):
        """
        """
        if not self.temporal: raise il.NotTemporalVariable
        assert intervals.ndim == 2
        n    = intervals.shape[1]
        shp  = (n,)+self.data.shape[1:]
        time = np.zeros(n)
        data = np.ma.zeros(shp)
        for i in range(n):
            t0          = intervals[0,i]-window
            tf          = intervals[1,i]+window
            time[i]     = 0.5*(t0+tf)
            mean        = self.integrateInTime(mean=True,t0=t0,tf=tf).convert(self.unit)
            data[i,...] = mean.data
        return Variable(name      = "coarsened_%s" % self.name,
                        unit      = self.unit,
                        time      = time,
                        time_bnds = intervals,
                        data      = data,
                        lat       = self.lat,
                        lon       = self.lon,
                        area      = self.area)
        

        
