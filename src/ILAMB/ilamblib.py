from datetime import datetime
from netCDF4 import Dataset
import numpy as np

class VarNotInFile(Exception):
    pass

class VarNotMonthly(Exception):
    pass

class VarNotInModel(Exception):
    pass

class UnknownUnit(Exception):
    pass

class AreasNotInModel(Exception):
    pass

class MisplacedData(Exception):
    pass

def GenerateDistinctColors(N,saturation=0.67,value=0.67):
    r"""Generates N distinct colors.

    Computes N distinct colors using HSV color space. We hold the
    saturation and value levels constant and linearly vary the
    hue. Finally we convert to RGB color space for use with other
    functions.

    Parameters
    ----------
    N : int
        number of distinct colors to generate
    saturation : float, optional
        argument of HSV color space
    value : float, optional
        argument of HSV color space

    Returns
    -------
    RGB_tuples : list
       list of N distinct RGB tuples    
    """
    from colorsys import hsv_to_rgb
    HSV_tuples = [(x/float(N-1), saturation, value) for x in range(N)]
    RGB_tuples = map(lambda x: hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def ExtractPointTimeSeries(filename,variable,lat,lon,verbose=False):
    r"""Extracts the timeseries of a given variable at a given point from a
    netCDF file.

    Parameters
    ----------
    filename : string
        name of the NetCDF file to read
    variable : string
        name of the variable to extract
    lat : float
        latitude in degrees at which to extract field
    lon : float 
        longitude in degrees east of the international dateline at which to extract field

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ma.core.MaskedArray
        an array of the extracted variable
    unit : string
        a description of the extracted unit

    Notes
    -----
    Collapse this function with ExtractTimeSeries.
    """
    f = Dataset(filename)
    try:
        if verbose: print "Looking for %s in %s" % (variable,filename)
        vari = f.variables[variable]
    except:
        if verbose: print "%s is not a variable in this netCDF file" % variable
        raise VarNotInFile("%s is not a variable in this netCDF file" % variable)

    # determine time shift
    t    = f.variables['time']
    unit = t.units.split(" since ")
    assert unit[0] == "days"
    t0   = datetime(1850,1,1,0,0,0)
    tf   = datetime.strptime((unit[-1].split())[0],"%Y-%m-%d")
    dt   = (tf-t0).days

    # extract variable by finding the closest lat,lon if they exist
    try:
        lats = f.variables['lat']
        lons = f.variables['lon']
        ilat = np.argmin(np.abs(lats[...]-lat))
        ilon = np.argmin(np.abs(lons[...]-lon))
        var  = np.ma.masked_values(vari[...,ilon,ilat],vari._FillValue)
    except:
        var  = np.ma.masked_values(vari[...],vari._FillValue)
    return t[:]+dt,var,vari.units

def ExtractTimeSeries(filename,variable,verbose=False):
    r"""Extracts the timeseries of a given variable from a netCDF file.

    Parameters
    ----------
    filename : string
        name of the NetCDF file to read
    variable : string
        name of the variable to extract

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ma.core.MaskedArray
        an array of the extracted variable
    unit : string
        a description of the extracted unit

    Notes
    -----
    Collapse this function with ExtractPointTimeSeries.
    """
    f = Dataset(filename)
    try:
        if verbose: print "Looking for %s in %s" % (variable,filename)
        vari = f.variables[variable]
    except:
        if verbose: print "%s is not a variable in this netCDF file" % variable
        raise VarNotInFile("%s is not a variable in this netCDF file" % variable)
    t    = f.variables['time']
    unit = t.units.split(" since ")
    assert unit[0] == "days"
    t0   = datetime(1850,1,1,0,0,0)
    tf   = datetime.strptime((unit[-1].split())[0],"%Y-%m-%d")
    dt   = (tf-t0).days
    var  = np.ma.masked_values(vari[...],vari._FillValue)
    lat  = f.variables["lat"][...]
    lon  = f.variables["lon"][...]
    return t[:]+dt,var,vari.units,lat,lon

def RootMeanSquaredError(reference,prediction,normalize="none",weights=None):
    r"""
    Computes the root mean squared error (RMSE) of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of
    length :math:`n` the RMSE is

    .. math:: \sqrt{\sum_{i=1}^{n}\frac{\left(x_i-y_i\right)^2}{n}}

    where :math:`\mathbf{x}` is considered the reference vector. The
    RMSE can be normalized in one of several ways. The keyword
    "maxmin" will return the normalized RMSE by

    .. math:: \frac{\text{RMSE}}{\max(\mathbf{x})-\min(\mathbf{x})}

    Parameters
    ----------
    reference : numpy.ndarray
        1D array representing the first data series
    prediction : numpy.ndarray
        1D array representing the second data series
    normalize : string, optional
        use to specify the normalization technique
    weights : numpy.ndarray, optional
        specify to use a weighted mean

    Returns
    -------
    rmse : float
        the root mean squared error

    Example
    -------
    >>> x = np.asarray([1,2,3])
    >>> y = np.asarray([4,5,6])
    >>> RootMeanSquaredError(x,y)
    3.0
    >>> RootMeanSquaredError(x,y,normalize="maxmin")
    1.5  
    """
    assert reference.size == prediction.size
    rmse = np.sqrt(np.ma.average((prediction-reference)**2,weights=weights))
    if normalize == "maxmin": rmse /= (reference.max()-reference.min())
    if normalize == "score":
        rmse0 = np.sqrt(np.ma.average(reference**2,weights=weights))
        rmse  = (1-rmse/rmse0).clip(0)
    return rmse

def Bias(reference,prediction,normalize="none",weights=None):
    r"""
    Computes the bias of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of
    length :math:`n` the bias is

    .. math:: \text{bias} = \bar{\mathbf{y}}-\bar{\mathbf{x}}

    where :math:`\mathbf{x}` is considered the reference vector. The
    bar notation denotes a mean, or

    .. math:: \bar{\mathbf{x}} = \frac{\sum_{i=1}^{n}x_i}{n}

    The RMSE can be normalized in one of several ways. The keyword
    "maxmin" will return the normalized bias by

    .. math:: \text{normalized bias} = \frac{\text{bias}}{\max(\mathbf{x})-\min(\mathbf{x})}

    The "score" keyword will normlize the bias by

    .. math:: \text{normalized bias} = 1-\left|\frac{\text{bias}}{\bar{\mathbf{x}}}\right|

    where values less than zero will be clipped at zero.

    Parameters
    ----------
    reference : numpy.ndarray
        1D array representing the first data series
    prediction : numpy.ndarray
        1D array representing the second data series
    normalize : string, optional
        use to specify the normalization technique, one of "score","maxmin"
    weights : numpy.ndarray, optional
        specify to use a weighted mean

    Returns
    -------
    bias : float
        the bias

    Example
    -------
    >>> x = np.asarray([1,2,1,2])
    >>> y = np.asarray([2,1,2,1])
    >>> Bias(x,y)
    0.0
    """
    assert reference.size == prediction.size
    pmean = np.ma.average(prediction,weights=weights)
    rmean = np.ma.average(reference,weights=weights)
    bias  = pmean-rmean
    if normalize == "maxmin": bias /= (reference.max()-reference.min())
    if normalize == "score" : bias  = (1.-np.abs(bias/rmean)).clip(0,1)
    return bias

def AnnualMean(t,var):
    """
    Computes the annual mean of the input time series.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable
    
    Returns
    -------
    tmean : numpy.ndarray
        a 1D array of the mean annual times in days since 00:00:00 1/1/1850
    vmean : numpy.ndarray
        a 1D array of the mean annual values of the input array var
    
    Example
    -------
    >>> t = np.asarray([15.5,45.,74.5,105.,135.5,166.,196.5,227.5,258.,288.5,319.,349.5])
    >>> x = np.ma.array([1,1,2,3,1,2,1,1,1,3,2,3],mask=[1,1,0,0,1,0,1,1,1,0,0,0]) # mask the ones
    >>> tmean,xmean = AnnualMean(t,x)
    >>> np.allclose(xmean[0],2.5027322404371586)
    True
    """
    assert t.size >= 12
    begin = np.argmin(t[:11]%365)
    end   = begin+int(t[begin:].size/12.)*12
    from constants import dpm_noleap as dpm
    dpy   = dpm.sum()
    tmean = np.ma.average(  t[begin:end].reshape((-1,12)),axis=1,weights=dpm/dpy)
    vmean = np.ma.average(var[begin:end].reshape((-1,12)),axis=1,weights=dpm/dpy)
    return tmean,vmean

def AnnualMinMax(t,var):
    """
    Computes the annual minimum and maximum of the input time series.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable
    
    Returns
    -------
    vmin : numpy.ndarray
        a 1D array of the minimum annual values of the input array var
    vmax : numpy.ndarray
        a 1D array of the maximum annual values of the input array var
    
    Example
    -------
    >>> t = np.asarray([15.5,45.,74.5,105.,135.5,166.,196.5,227.5,258.,288.5,319.,349.5])
    >>> x = np.ma.array([1,1,2,3,1,2,1,1,1,3,2,3],mask=[1,1,0,0,1,0,1,1,1,0,0,0]) # mask the ones
    >>> tmean,xm = AnnualMean(t,x)
    >>> np.allclose(xmean[0],2.5027322404371586)
    True
    """
    assert t.size >= 12 
    begin = np.argmin(t[:11]%365)
    end   = begin+int(t[begin:].size/12.)*12
    vmax  = np.ma.max(var[begin:end].reshape((-1,12)),axis=1)
    vmin  = np.ma.min(var[begin:end].reshape((-1,12)),axis=1)
    return vmin,vmax

def DecadalAmplitude(t,var):
    r"""
    Computes the mean and standard deviation of the amplitude over
    decades of the input time series.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable
    
    Returns
    -------
    tmean : numpy.ndarray
        an array of times indicating the mean time in each decade
    Amean : numpy.ndarray
        an array of mean amplitudes over decades
    Astd : numpy.ndarray
        an array of the standard deviation of the amplitudes over decades

    Notes
    -----
    Fractions of a decade at the beginning and end of the dataset are
    discarded.
    """
    begin = np.argmin(t[:119]%(365*10))
    end   = begin+int(t[begin:].size/120.)*120
    tmean = np.apply_along_axis(np.mean,1,t[begin:end].reshape((-1,120)))
    v     = var[begin:end].reshape((-1,10,12))
    A     = np.ma.max(v,axis=2)-np.ma.min(v,axis=2)
    Amean = np.ma.mean(A,axis=1)
    Astd  = np.ma.std (A,axis=1)
    return tmean,Amean,Astd
    
def WindowedTrend(t,var,window=365.):
    r"""Compute a windowed trend.

    For each point in the time series, compute the slope of the
    best-fit line which passes through the data lying within the
    specified window of time.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable
    window : float
        the number of days to use in the trend computation

    Returns
    -------
    trend : numpy.ndarray
        a 1D array of the slope in units of var per year
    """
    trend = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        tleft  = t[i]-window*0.5
        tright = t[i]+window*0.5
        dl     = max(t[0]-tleft  ,0); tleft += dl; tright += dl
        dr     = max(tright-t[-1],0); tleft -= dr; tright -= dr
        condition = (t>=tleft)*(t<=tright)
        x = np.ma.masked_where(condition,t  ,copy=False)
        y = np.ma.masked_where(condition,var,copy=False)
        p = np.ma.polyfit(x/365.+1850,y,1)
        trend[i] = p[0]
    return trend

def AnnualCycleInformation(t,var):
    r"""Returns information regarding the annual cycle

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable    

    Returns
    -------
    tmax : numpy.ndarray
        a 1D array of the mean maximum times of the year in fractions
        of a year

    """
    begin = np.argmin(t[:11]%365)
    end   = begin+int(t[begin:].size/12.)*12
    ts    = np.arange(12)
    shp   = (-1,12) + var.shape[1:]
    v     = var[begin:end,...].reshape(shp)
    vmean = np.ma.mean(v,axis=0)
    vstd  = np.ma.std (v,axis=0)
    ts    = ts[np.ma.argmax(v,axis=1)]
    tmax  = np.ma.mean(ts,axis=0)
    tmstd = np.ma.std (ts,axis=0)
    return vmean,vstd,tmax,tmstd

def DecadalMaxTime(t,var):
    r""" 
    For each decade in the input dataset, compute the mean time of the
    year in which the maximum var is realized.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable    

    Returns
    -------
    tmax : numpy.ndarray
        a 1D array of the mean maximum times of the year in fractions
        of a year

    Notes
    -----
    Fractions of a decade at the beginning and end of the dataset are
    discarded.
    """
    begin = np.argmin(t[:119]%(365*10))
    end   = begin+int(t[begin:].size/120.)*120
    tt    = t[begin:end].reshape((-1,10,12))/365.+1850.
    ts    = tt[0,0,:]-tt[0,0,0]
    v     = var[begin:end].reshape((-1,10,12))
    tmax  = ts[np.ma.argmax(v,axis=2)]
    tmax  = np.ma.mean(tmax,axis=1)
    return tmax

def DecadalMinTime(t,var):
    r""" 
    For each decade in the input dataset, compute the mean time of the
    year in which the minimum var is realized.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable    

    Returns
    -------
    tmin : numpy.ndarray
        a 1D array of the mean minimum times of the year in fractions
        of a year

    Notes
    -----
    Fractions of a decade at the beginning and end of the dataset are
    discarded.
    """
    begin = np.argmin(t[:119]%(365*10))
    end   = begin+int(t[begin:].size/120.)*120
    tt    = t[begin:end].reshape((-1,10,12))/365.+1850.
    ts    = tt[0,0,:]-tt[0,0,0]
    v     = var[begin:end].reshape((-1,10,12))
    tmin  = ts[np.ma.argmin(v,axis=2)]
    tmin  = np.ma.mean(tmin,axis=1)
    return tmin

def MonthlyWeights(t):
    r"""For the given time series, return the number of days in time
    period.

    Each element of the time array is assumed to correspond to a
    month. The routine then returns the number of days in each month
    of the entire time series. These weights are ideal to be used in
    temporal evaluations.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850

    Returns
    -------
    w : numpy.ndarray
        a 1D array of weights, number of days per month
    """
    from constants import dpm_noleap
    dpy = dpm_noleap.sum()
    monthly_weights = dpm_noleap/dpy
    w  = monthly_weights[np.asarray((t % dpy)/dpy*12,dtype='int')]
    return w

def SpatiallyIntegratedTimeSeries(var,areas):
    r"""Integrate a variable over space.
    
    Given a variable :math:`f(\mathbf{x},t)`, the spatially averaged
    variable is then given as

    .. math:: \overline{f}(t) = \int_A f(\mathbf{x},t)\ dA

    where we approximate this integral by nodal integration.

    Parameters
    ----------
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable where
        there are at least two dimensions, (...,latitudes,longitudes)
    area : numpy.ndarray
        a two-dimensional array of areas of the form,
        (latitudes,longitudes)

    Returns
    -------
    vbar : numpy.array
        the spatially integrated variable
    """
    assert var.shape[-2:] == areas.shape
    vbar = (var*areas).sum(axis=-1).sum(axis=-1)
    return vbar
            
def TemporallyIntegratedTimeSeries(t,var):
    r"""Integrate a variable over time.
    
    Given a variable :math:`f(\mathbf{x},t)`, the temporally averaged
    variable is then given as

    .. math:: \hat{f}(\mathbf{x}) = \int_t f(\mathbf{x},t)\ dt

    where we approximate this integral by nodal integration.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array assumed to be a monthly average of a variable with
        the dimensions, (ntimes,latitudes,longitudes)

    Returns
    -------
    vhat : numpy.array
        the temporally integrated variable
    """
    wgt  = MonthlyWeights(t)*365*24*3600
    if var.ndim == 1:
        vhat = np.ma.sum(var*wgt) 
    else:
        vhat = np.ma.sum(var*wgt[:,np.newaxis,np.newaxis],axis=0) 
    return vhat

def CellAreas(lat,lon):
    """Given arrays of latitude and longitude, return cell areas in square meters.

    """
    from constants import earth_rad

    x = np.zeros(lon.size+1)
    x[1:-1] = 0.5*(lon[1:]+lon[:-1])
    x[ 0]   = lon[ 0]-0.5*(lon[ 1]-lon[ 0])
    x[-1]   = lon[-1]+0.5*(lon[-1]-lon[-2])
    if(x.max() > 181): x -= 180
    x *= np.pi/180.

    y = np.zeros(lat.size+1)
    y[1:-1] = 0.5*(lat[1:]+lat[:-1])
    y[ 0]   = lat[ 0]-0.5*(lat[ 1]-lat[ 0])
    y[-1]   = lat[-1]+0.5*(lat[-1]-lat[-2])
    y *= np.pi/180.

    dx    = earth_rad*(x[1:]-x[:-1])
    dy    = earth_rad*(np.sin(y[1:])-np.sin(y[:-1]))
    areas = np.outer(dx,dy).T

    return areas

def GlobalLatLonGrid(res):
    r"""Generates a latitude/longitude grid at a desired resolution
    
    Computes 1D arrays of latitude and longitude values which
    correspond to cell interfaces and centroids at a given resolution.

    Parameters
    ----------
    res : float
        the desired resolution of the grid in degrees

    Returns
    -------
    lat_bnd : numpy.ndarray
        a 1D array of latitudes which represent cell interfaces
    lon_bnd : numpy.ndarray
        a 1D array of longitudes which represent cell interfaces
    lat : numpy.ndarray
        a 1D array of latitudes which represent cell centroids
    lon : numpy.ndarray
        a 1D array of longitudes which represent cell centroids
    """
    nlon    = int(360./res)+1
    nlat    = int(180./res)+1
    lon_bnd = np.linspace(-180,180,nlon)
    lat_bnd = np.linspace(-90,90,nlat)
    lat     = 0.5*(lat_bnd[1:]+lat_bnd[:-1])
    lon     = 0.5*(lon_bnd[1:]+lon_bnd[:-1])
    return lat_bnd,lon_bnd,lat,lon

def NearestNeighborInterpolation(lat1,lon1,data1,lat2,lon2):
    r"""Interpolates globally grided data at another resolution

    Parameters
    ----------
    lat1 : numpy.ndarray
        a 1D array of latitudes of cell centroids corresponding to the 
        source data
    lon1 : numpy.ndarray
        a 1D array of longitudes of cell centroids corresponding to the 
        source data
    data1 : numpy.ndarray
        an array of data to be interpolated of shape = (lat1.size,lon1.size,...)
    lat2 : numpy.ndarray
        a 1D array of latitudes of cell centroids corresponding to the 
        target resolution
    lon2 : numpy.ndarray
        a 1D array of longitudes of cell centroids corresponding to the 
        target resolution

    Returns
    -------
    data2 : numpy.ndarray
        an array of interpolated data of shape = (lat2.size,lon2.size,...)
    """
    rows  = np.apply_along_axis(np.argmin,1,np.abs(lat2[:,np.newaxis]-lat1))
    cols  = np.apply_along_axis(np.argmin,1,np.abs(lon2[:,np.newaxis]-lon1))
    data2 = data1[np.ix_(rows,cols)]
    return data2
    
def TrueError(lat1_bnd,lon1_bnd,lat1,lon1,data1,lat2_bnd,lon2_bnd,lat2,lon2,data2):
    r"""Computes the pointwise difference between two sets of gridded data

    To obtain the pointwise error we populate a list of common cell
    interfaces and then interpolate both input arrays to the composite
    grid resolution using nearest-neighbor interpolation.

    Parameters
    ----------
    lat1_bnd, lon1_bnd, lat1, lon1 : numpy.ndarray
        1D arrays corresponding to the latitude/longitudes of the cell 
        interfaces/centroids
    data1 : numpy.ndarray
        an array of data to be interpolated of shape = (lat1.size,lon1.size,...)
    lat2_bnd, lon2_bnd, lat2, lon2 : numpy.ndarray
        1D arrays corresponding to the latitude/longitudes of the cell 
        interfaces/centroids
    data2 : numpy.ndarray
        an array of data to be interpolated of shape = (lat2.size,lon2.size,...)

    Returns
    -------
    lat_bnd, lon_bnd, lat, lon : numpy.ndarray
        1D arrays corresponding to the latitude/longitudes of the cell 
        interfaces/centroids of the resulting error
    error : numpy array
        an array of the pointwise error of shape = (lat.size,lon.size,...)
    """
    # combine limits, sort and remove duplicates
    lat_bnd = np.hstack((lat1_bnd,lat2_bnd)); lat_bnd.sort(); lat_bnd = np.unique(lat_bnd)
    lon_bnd = np.hstack((lon1_bnd,lon2_bnd)); lon_bnd.sort(); lon_bnd = np.unique(lon_bnd)

    # need centroids of new grid for nearest-neighbor interpolation
    lat = 0.5*(lat_bnd[1:]+lat_bnd[:-1])
    lon = 0.5*(lon_bnd[1:]+lon_bnd[:-1])
    
    # interpolate datasets at new grid
    d1 = NearestNeighborInterpolation(lat1,lon1,data1,lat,lon)
    d2 = NearestNeighborInterpolation(lat2,lon2,data2,lat,lon)
    
    # relative to the first grid/data
    error = d2-d1
    return lat_bnd,lon_bnd,lat,lon,error
