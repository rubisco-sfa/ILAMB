from netCDF4 import Dataset
import numpy as np
from datetime import datetime

DAYS_PER_MONTH = np.asarray([31,28,31,30,31,30,31,31,30,31,30,31],dtype='float')

def ExtractPointTimeSeries(filename,variable,lat,lon,navg=1):
    r"""
    Extracts the timeseries of a given variable at a given point from a
    netcdf file.

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
    navg : int, optional
        number of non-fill variable layers to include in average

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ndarray
        an array of the extracted variable

    Raises
    ------
    NotImplementedError
            If the variable is in an unexpected format
    """
    f    = Dataset(filename)
    try:
        vari = f.variables[variable]
    except:
        raise ValueError("%s is not a variable in this netCDF file" % variable)
    t    = f.variables['time']
    unit = t.units.split(" since ")
    assert unit[0] == "days"
    t0   = datetime(1850,1,1,0,0,0)
    tf   = datetime.strptime((unit[-1].split())[0],"%Y-%m-%d")
    dt   = (tf-t0).days
    lats = f.variables['lat']
    lons = f.variables['lon']
    ilat = lats[...].searchsorted(lat)
    ilon = lons[...].searchsorted(lon)
    if vari.ndim == 4:
        var   = vari[:,:,ilon,ilat]
        if var is not np.ma.masked:
            var   = var[:,0]
        else:
            first = np.apply_along_axis(np.sum,1,var.mask)
            var   = var[np.ix_(range(t.shape[0])),first][0,:]
    else:
        raise NotImplementedError("Unexpected data format for given variable.")
    return t[:]-dt,var

def RootMeanSquaredError(reference,prediction,normalize="none"):
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
    normalize : string
        use to specify the normalization technique

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
    rmse  = np.sqrt(((prediction-reference)**2).mean())
    if normalize == "maxmin": rmse /= (reference.max()-reference.min())
    return rmse

def Bias(reference,prediction,normalize="none"):
    r"""
    Computes the bias of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of length :math:`n` the bias is

    .. math:: \sum_{i=1}^{n}\frac{\left(x_i-y_i\right)}{n}

    where :math:`\mathbf{x}` is considered the reference vector. The
    RMSE can be normalized in one of several ways. The keyword
    "maxmin" will return the normalized bias by

    .. math:: \frac{\text{bias}}{\max(\mathbf{x})-\min(\mathbf{x})}

    Parameters
    ----------
    reference : numpy.ndarray
        1D array representing the first data series
    prediction : numpy.ndarray
        1D array representing the second data series
    normalize : string
        use to specify the normalization technique

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
    bias = (prediction-reference).mean()
    if normalize == "maxmin": rmse /= (reference.max()-reference.min())
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
    dpm   = DAYS_PER_MONTH # needs to be read from a constants file
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
    r"""
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

