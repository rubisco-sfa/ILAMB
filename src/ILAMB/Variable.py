import ilamblib as il
import Post as post
from constants import spd,spy,convert,regions as ILAMBregions
import numpy as np
import pylab as plt

class Variable:
    """A class for managing variables defined in time and the globe.

    The Variable object requires that you specify at a minimum the
    data and its unit in the constructor. However, optionally you may
    choose to add time and space information as well. The behavior of
    the member routines will then change based on what has been
    defined. For example, the plot routine will render a xy-scatter
    plot if the variable is a function of time and a global
    pseudocolor plot if the variable is spatial.
    """

    def __init__(self,data,unit,name="unnamed",time=None,lat=None,lon=None,area=None):
        """Constructor for the variable class

        Parameter
        ---------
        data : numpy.ndarray
            The array which contains the data which constitutes the
            variable. If no time or space information is included, the
            data is expected to be a scalar. We assume a prescendence of
            dimensions, ('time','lat','lon').
        unit : str 
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
        """
        assert type(data) is type(np.ma.masked_array())
        self.data = data # [monthly means (for now)]
        self.unit = unit
        self.name = name
        self.attributes = {}
        self.time = time # [days since 1/1/1850]
        self.lat  = lat
        self.lon  = lon
        self.area = area # [m2]
        self.temporal = False
        self.spatial  = False
        self.dt       = 0. 
        if time is not None: 
            self.temporal = True
            self.dt = (time[1:]-time[:-1]).mean()
        if lat  is not None: self.spatial  = True
        if lon  is not None:
            self.lon = (self.lon<=180)*self.lon+(self.lon>180)*(self.lon-360)
        if self.spatial:
            assert lat is not None
            assert lon is not None
            if self.area is None: self.area = il.CellAreas(lat,lon)
        self.monthly = False
        if self.temporal:
            # dt = mean temporal spacing, tells us what kind of
            # analysis is appropriate (annual cycle, diurnal cycle,
            # etc)
            dt = (time[1:]-time[:-1]).mean()
            if np.allclose(dt,30,atol=3): self.monthly = True
        
    def toNetCDF4(self,dataset,attributes={}):
        """Adds the variable to the specified netCDF4 dataset

        Parameters
        ----------
        dataset : netCDF4.Dataset
            a dataset into which you wish to save this variable
        attributes : dictionary
            a dictionary of additional information to be saved along
            with the variable
        """
        def _checkTime(t,dataset):
            """A local function for ensuring the time dimension is saved in the dataset."""
            if "time" in dataset.dimensions.keys():
                assert t.shape == dataset.variables["time"][...].shape
                assert np.allclose(t,dataset.variables["time"][...],atol=0.5*self.dt)
            else:
                dataset.createDimension("time")
                T = dataset.createVariable("time","double",("time"))
                T.setncattr("units","days since 1850-01-01 00:00:00")
                T.setncattr("calendar","noleap")
                T.setncattr("axis","T")
                T.setncattr("long_name","time")
                T.setncattr("standard_name","time")
                T[...] = t
        def _checkSpace(lat,lon,dataset):
            """A local function for ensuring space dimensions are saved in the dataset."""
            if "lat" in dataset.dimensions.keys():
                assert lat.shape == dataset.variables["lat"][...].shape
                assert np.allclose(lat,dataset.variables["lat"][...])
                assert lon.shape == dataset.variables["lon"][...].shape
                assert np.allclose(lon,dataset.variables["lon"][...])
            else:
                dataset.createDimension("lon",size=lon.size)
                X = dataset.createVariable("lon","double",("lon"))
                X.setncattr("units","degrees_east")
                X.setncattr("axis","X")
                X.setncattr("long_name","longitude")
                X.setncattr("standard_name","longitude")
                X[...] = lon
                dataset.createDimension("lat",size=lat.size)
                Y = dataset.createVariable("lat","double",("lat"))
                Y.setncattr("units","degrees_north")
                Y.setncattr("axis","Y")
                Y.setncattr("long_name","latitude")
                Y.setncattr("standard_name","latitude")
                Y[...] = lat

        if self.temporal: _checkTime(self.time,dataset)
        if self.spatial:  _checkSpace(self.lat,self.lon,dataset)

        dim = []
        if self.temporal: 
            dim.append("time")
        if self.spatial: 
            dim.append("lat")
            dim.append("lon")

        V = dataset.createVariable(self.name,"double",dim)
        V.setncattr("units",self.unit)
        for attr in attributes.keys(): V.setncattr(attr,attributes[attr])
        V[...] = self.data

    def integrateInSpace(self,region=None):
        """Integrates the variable over space

        Uses nodal integration to integrate the variable over the
        specified region. If no region is specified, then perform the
        integration over the extent of the dataset.
        
        Parameters
        ----------
        region : str
            name of the region overwhich you wish to integrate
        
        Returns
        -------
        integral : ILAMB.Variable.Variable
            a Variable instace with the integrated value along with the
            appropriate name and unit change.
        """
        if not self.spatial: raise il.NotSpatialVariable()
        if region is None:
            integral = il.SpatiallyIntegratedTimeSeries(self.data,self.area)
            name = self.name + "_integrated_over_space"
        else:
            rem_mask  = np.copy(self.data.mask)
            lats,lons = ILAMBregions[region]
            mask      = (np.outer((self.lat>lats[0])*(self.lat<lats[1]),
                                  (self.lon>lons[0])*(self.lon<lons[1]))==0)
            self.data.mask += mask
            integral  = il.SpatiallyIntegratedTimeSeries(self.data,self.area)
            self.data.mask = rem_mask
            name = self.name + "_integrated_over_%s" % region
        unit = self.unit.replace(" m-2","")
        return Variable(integral,unit,time=self.time,name=name)

    def convert(self,unit):
        """Incomplete attempt to handle unit conversions

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

    def spatialDifference(self,var):
        """Computes the point-wise difference of two spatially defined variables
        
        This routine uses ILAMB.ilamblib.TrueError to compute a
        pointwise difference between the two variables. First the
        boundaries of the lat/lon array are found and a composite is
        made which contains the boundaries of both grids. Second we
        interpolate both variables at this new resolution by a
        neareast neighbor approach. Finally we return a new variable
        object which is the difference between the input variable and
        this variable.

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
            bnds[0]    = x[0] -0.5*(x[ 1]-x[ 0])
            bnds[-1]   = x[-1]+0.5*(x[-1]-x[-2])
            return bnds
        assert var.unit == self.unit
        assert self.temporal == False
        if not self.spatial: raise il.NotSpatialVariable("Must be a spatial variabel to compute the difference")
        lat_bnd1 = _make_bnds(self.lat)
        lon_bnd1 = _make_bnds(self.lon)
        lat_bnd2 = _make_bnds( var.lat)
        lon_bnd2 = _make_bnds( var.lon)
        lat_bnd,lon_bnd,lat,lon,error = il.TrueError(lat_bnd1,lon_bnd1,self.lat,self.lon,self.data,
                                                     lat_bnd2,lon_bnd2, var.lat, var.lon, var.data)
        diff = Variable(error,var.unit,lat=lat,lon=lon,name="%s_minus_%s" % (var.name,self.name))
        return diff

    def integrateInTime(self):
        """Integrates the variable over time

        Uses nodal integration to integrate the variable over the time domain.
        
        Parameters
        ----------
        region : str
            name of the region overwhich you wish to integrate
        
        Returns
        -------
        integral : ILAMB.Variable.Variable
            a Variable instace with the integrated value along with the
            appropriate name and unit change.
        """

        if not self.temporal: raise il.NotTemporalVariable()
        integral = il.TemporallyIntegratedTimeSeries(self.time,self.data)
        if " s-1" in self.unit: 
            integral *= spd
            unit      = self.unit.replace(" s-1","")
        if " d-1" in self.unit: 
            unit      = self.unit.replace(" d-1","")
        if " y-1" in self.unit: 
            integral *= spy
            unit      = self.unit.replace(" y-1","")
        name = self.name + "_integrated_over_time"
        return Variable(integral,unit,lat=self.lat,lon=self.lon,area=self.area,name=name)

    def _overlap(self,var):
        """A local function for determining indices of this variable and the
        input variable which represent the beginning and end of the
        variables' overlap in the time dimension
        """
        mint = max(var.time.min(),self.time.min())
        maxt = min(var.time.max(),self.time.max())
        b     = np.argmin(np.abs( var.time-mint)); e     = np.argmin(np.abs( var.time-maxt)) 
        ref_b = np.argmin(np.abs(self.time-mint)); ref_e = np.argmin(np.abs(self.time-maxt)) 
        comparable = (var.time[b:e].shape == self.time[ref_b:ref_e].shape)
        if comparable:
            comparable = np.allclose(var.time[b:e],self.time[ref_b:ref_e],atol=0.5*self.dt)
        return comparable,b,e,ref_b,ref_e

    def bias(self,var,normalize="none"):
        """Computes the bias
        
        Uses ILAMB.ilamblib.Bias to compuate the bias between these
        two variables relative to this variable

        Parameters
        ----------
        var : ILAMB.Variable.Variable
            The variable we wish to compare against this variable
        normalize : str
            The normalization type as defined in ILAMB.ilamblib.Bias
        
        Returns
        -------
        bias : float
            The bias of the two variables relative to this variable
        """
        comparable,b,e,vb,ve = self._overlap(var)
        if not comparable: raise il.VarsNotComparable()
        mw = il.MonthlyWeights(self.time[b:e])
        return il.Bias(self.data[b:e],var.data[vb:ve],normalize=normalize)

    def RMSE(self,var,normalize="none"):
        """Computes the RootMeanSquaredError
        
        Uses ILAMB.ilamblib.RootMeanSquaredError to compuate the
        RootMeanSquaredError between these two variables relative to
        this variable.

        Parameters
        ----------
        var : ILAMB.Variable.Variable
            The variable we wish to compare against this variable
        normalize : str
            The normalization type as defined in
            ILAMB.ilamblib.RootMeanSquaredError
        
        Returns
        -------
        rmse : float
            The rmse of the two variables relative to this variable

        """
        comparable,b,e,vb,ve = self._overlap(var)
        if not comparable: raise il.VarsNotComparable()
        mw = il.MonthlyWeights(self.time[b:e])
        return il.RootMeanSquaredError(self.data[b:e],var.data[vb:ve],normalize=normalize)

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
        vmin : float, optional
            The minimum plotted value
        vmax : float, optional
            The maximum plotted value
        region : str, optional
            The region on which to display a spatial variable
        cmap : str
            The colormap to be used in plotting the spatial variable
        """
        vmin   = keywords.get("vmin",self.data.min())
        vmax   = keywords.get("vmax",self.data.max())
        region = keywords.get("region","global")
        cmap   = keywords.get("cmap","rainbow")
        if self.temporal and not self.spatial:
            ax.plot(self.time,self.data,'-')
        if not self.temporal and self.spatial:
            ax     = post.GlobalPlot(self.lat,self.lon,self.data,ax,
                                     vmin   = vmin  , vmax = vmax,
                                     region = region, cmap = cmap)
        return ax

    def annualCycle(self):
        pass
