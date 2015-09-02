from netCDF4 import Dataset
import numpy as np
import ilamblib as il
from constants import spd,dpy,mid_months,convert,regions as ILAMBregions
import Post as post

def FromNetCDF4(filename,variable_name):
    """Extracts data from a netCDF4 datafile for use in a Variable object.
    
    Intended to be used inside of the Variable constructor. Some of
    the return arguments will be None depending on the contents of the
    netCDF4 file.

    Parameters
    ----------
    filename : str, optional
        Name of the netCDF4 file from which to extract a variable
    variable_name : str, optional
        Name of the variable to extract from the netCDF4 file

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
    f         = Dataset(filename,mode="r")
    var       = f.variables[variable_name]
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
        for key in f.dimensions.keys():
            if "lat" in key: lat_name = key
            if "lon" in key: lon_name = key
        if lat_name is not None: lat = f.variables[lat_name][...]
        if lon_name is not None: lon = f.variables[lon_name][...]
        if lat.size != data: lat = None
        if lon.size != data: lon = None
    return np.ma.masked_array(var[...]),var.units,variable_name,t,lat,lon,data

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
        """
        # See if the user specified a netCDF4 file and variable
        filename      = keywords.get("filename"     ,None)
        variable_name = keywords.get("variable_name",None)
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
            data,unit,name,time,lat,lon,ndata = FromNetCDF4(filename,variable_name)
        area = keywords.get("area",None)

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
        self.area    = area
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
            if not mean: unit += " d"
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
            integral /= dt
            unit     += " d-1"
            name     += "_and_divided_by_time_period"

        # special handling of Joules per day --> Watts (again, need to improve)
        if "J" in unit and "d-1" in unit:
            integral /= spd
            unit      = unit.replace("J","W").replace("d-1","")

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
        mean = self.data.mean(axis=-1)
        std  = self.data.std (axis=-1)
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
        mean  = v.mean(axis=0)
        std   = v.std (axis=0)
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
        assert var.unit == self.unit
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
        """Incomplete attempt to handle unit conversions.

        The following is an incomplete attempt to handle unit
        conversions. It was thought out to handle conversions of
        "powers" (e.g. Pg to g) or time conversions (e.g. seconds to
        days). This depends on units being specified in the following
        way:

        kg m-2 s-1

        where no division or parenthesis is used.

        Parameter
        ---------
        unit : str
            the desired unit to convert to
        
        Return
        ------
        self : ILAMB.Variable.Variable
            this object with its unit converted
        """
        def _parseToken(t):
            power = 1.
            denom = False
            if "-" in t:
                t     = t.split("-")
                power = float(t[-1])
                denom = True
                t     = t[0]
            return t,denom,power
        stoken = self.unit.split(" ")
        ttoken =      unit.split(" ")
        fct    = 1.0
        for s in stoken:
            s,sdenom,spower = _parseToken(s)
            found = False
            for t in ttoken:
                t,tdenom,tpower = _parseToken(t)
                if convert[s].has_key(t):
                    found = True
                    if sdenom: 
                        fct /= convert[s][t]**spower
                    else:
                        fct *= convert[s][t]**spower
                if found: break
            assert found==True
        self.data *= fct
        self.unit  = unit
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
            
        V = dataset.createVariable(self.name,"double",dim)
        V.setncattr("units",self.unit)
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
            t = self.time/365.+1850
            ax.plot(t,self.data,'-',
                    color=color,lw=lw,alpha=alpha,label=label)
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
            size = norm*30+20
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
            return (xy-n*xbar*ybar)/(np.sqrt(x2-n*xbar*xbar)*np.sqrt(y2-n*ybar*ybar))

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
        if not self.temporal or not var.temporal: raise il.NotTemporalVariable
        assert self.ndata == var.ndata
        if self.ndata:
            data = var.data.data-self.data.data
            mask = var.data.mask+self.data.mask
            data = np.ma.masked_array(data,mask=mask)
            data = data.mean(axis=-1)
            mask = np.all(mask,axis=-1)
            data = np.ma.masked_array(data,mask=mask)
        return Variable(data=data,
                        name="bias_of_%s" % self.name,unit=self.unit,
                        ndata=self.ndata,lat=self.lat,lon=self.lon,area=self.area)
    
    def rmse(self,var):
        """
        UNTESTED
        """
        if not self.temporal or not var.temporal: raise il.NotTemporalVariable
        assert self.ndata == var.ndata


    def spatialDistribution(self,var):
        """
        """
        pass

    def seasonalCycle(self,var):
        """
        """
        pass

    def interannualVariability(self,var):
        """
        """
        pass

def BiasScore(bias,period_mean):
    from copy import deepcopy
    score = deepcopy(bias)
    score.data = np.exp(-np.abs(score.data/period_mean.data))
    score.name = score.name.replace("bias","bias_score")
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

    """
    return Variable(data  = (1+np.cos(np.abs(phase_shift.data)/365*2*np.pi))*0.5,
                    unit  = "-",
                    name  = phase_shift.name.replace("phase_shift","phase_shift_score"),
                    ndata = phase_shift.ndata,
                    lat   = phase_shift.lat,
                    lon   = phase_shift.lon,
                    area  = phase_shift.area)

def AnalysisFluxrate(obs,mod,regions=['global.large'],dataset=None):
    """
    
    """
    # Check the data types. If the variables are datasites, make sure
    # they have the same number of datasites. If the observational
    # data are datasites and the model is globally gridded data,
    # extract the datasites. We will also capture whether or not the
    # data is spatially defined.
    if obs.ndata is not None:
        if mod.ndata is not None:
            assert obs.ndata == mod.ndata
        else:
            mod = mod.extractDatasites(obs.lat,obs.lon)
    spatial = obs.spatial
    
    # Check time intervals. The data from the observations and models
    # might not be on the same timeframe. So we compute the maximum
    # overlap and ensure that we have the right data shape and that
    # the data represent monthly means.
    t0 = max(obs.time.min(),mod.time.min())
    tf = min(obs.time.max(),mod.time.max())
    for var in [obs,mod]:
        begin = np.argmin(np.abs(var.time-t0))
        end   = np.argmin(np.abs(var.time-tf))+1
        var.time = var.time[begin:end]
        var.data = var.data[begin:end,...]
    assert obs.time.shape == mod.time.shape
    assert np.allclose(obs.time,mod.time,atol=14)

    # Integrate in time and divide through by the time period. We need
    # these maps/sites for plotting. We also need the period mean
    # across all space/sites for computing scores in other metrics.
    obs_timeint = obs.integrateInTime(mean=True)
    mod_timeint = mod.integrateInTime(mean=True)
    if spatial:
        period_mean      = obs_timeint.integrateInSpace(mean=True)
    else:
        period_mean,junk = obs_timeint.siteStats()
        
    # Compute maps of the bias and rmse. We will use these variables
    # later in the regional analysis to obtain means over individual
    # regions. Note that since we have already taken a temporal
    # average of the variables, the bias() function can reuse this
    # data and avoid extra averaging.
    bias = obs_timeint.bias(mod_timeint)
    rmse = obs        .rmse(mod)
    
    # regional analysis dumps variables into dictionaries
    for region in regions:

        if spatial:

            # 
            obs_periodmean[region] = obs_timeint.integrateInSpace(region=region)
            mod_periodmean[region] = mod_timeint.integrateInSpace(region=region)

            # scalar mean bias
            meanbias[region]       = bias.integrateInSpace(region=region,mean=True)
            
            
        else:
            
            # scalar period mean
            obs_periodmean[region],obs_sitestd[region] = obs_timeint.siteStats(region=region)
            mod_periodmean[region],mod_sitestd[region] = mod_timeint.siteStats(region=region)

            meanbias[region],junk = bias.siteStats(region=region)


            
            
    # optionally dump results to a NetCDF file
    if dataset is not None:
        for var in []:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(dataset)
            else:
                var.toNetCDF4(dataset)
                
if __name__ == "__main__":
    import os

    obs = Variable(filename = os.environ["ILAMB_ROOT"]+"/DATA/gpp/FLUXNET-MTE/derived/gpp.nc",
                   variable_name = "gpp")
    mod = Variable(filename = os.environ["ILAMB_ROOT"]+"/MODELS/CMIP5/inmcm4/gpp/gpp_Lmon_inmcm4_historical_r1i1p1_185001-200512.nc",
                   variable_name = "gpp")
    dataset = Dataset("gpp.nc",mode="w")
    AnalysisFluxrate(obs,mod,regions=['global.large','amazon'],dataset=dataset)

    obs = Variable(filename = os.environ["ILAMB_ROOT"]+"/DATA/le/FLUXNET/derived/le.nc",
                   variable_name = "le"); obs.name = "hfls"
    mod = Variable(filename = os.environ["ILAMB_ROOT"]+"/MODELS/CMIP5/inmcm4/hfls/hfls_Amon_inmcm4_historical_r1i1p1_185001-200512.nc",
                   variable_name = "hfls")
    dataset = Dataset("hfls.nc",mode="w")
    AnalysisFluxrate(obs,mod,regions=['global.large','amazon'],dataset=dataset)
    
