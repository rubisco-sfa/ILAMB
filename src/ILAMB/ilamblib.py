from netCDF4 import Dataset
import numpy as np

DAYS_PER_MONTH = np.asarray([31,28,31,30,31,30,31,31,30,31,30,31],dtype='float')

def ExtractPointTimeSeries(filename,variable,lat,lon,navg=1):
    """
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
        a 1D array of times in days since 00:00:00 1/1/1850
    var : numpy.ndarray
        an array of the extracted variable

    Raises
    ------
    NotImplementedError
            If the variable is in an unexpected format
    """
    f     = Dataset(filename)
    t     = f.variables['time']
    lats  = f.variables['lat']
    lons  = f.variables['lon']
    ilat  = lats[...].searchsorted(lat)
    ilon  = lons[...].searchsorted(lon)
    var   = f.variables[variable]
    ndim  = len(var.shape)
    if ndim == 4:
        var   = var[:,:,ilon,ilat]
        first = var.mask[0,:].sum() # assumes same number of masked layers for all time series
        last  = first+navg
        var   = np.apply_along_axis(np.mean,1,var.data[:,first:last])
    else:
        raise NotImplementedError("Unexpected data format for given variable.")
    return t[:],var

def ComputeNormalizedRootMeanSquaredError(reference,prediction):
    r"""
    Computes the normalized root mean squared error (NRMSE) of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of length :math:`n` the NRMSE is

    .. math:: \frac{\sqrt{\sum_{i=1}^{n}\frac{\left(x_i-y_i\right)^2}{n}}}{\max(\mathbf{x})-\min(\mathbf{x})}

    where :math:`\mathbf{x}` is considered the reference vector.

    Parameters
    ----------
    reference : numpy.ndarray
        1D array representing the first data series
    prediction : numpy.ndarray
        1D array representing the second data series

    Returns
    -------
    nrmse : float
        the normalized root mean squared error

    Example
    -------
    >>> x = np.asarray([1,2,3])
    >>> y = np.asarray([4,5,6])
    >>> ComputeNormalizedRootMeanSquaredError(x,y)
    1.5
  
    """
    assert reference.size == prediction.size
    nrmse  = np.sqrt(((prediction-reference)**2).mean())
    #nrmse /= (reference.max()-reference.min())
    return nrmse

def ComputeNormalizedBias(reference,prediction):
    r"""
    Computes the normalized bias of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of length :math:`n` the normalized bias is

    .. math:: \sum_{i=1}^{n}\frac{\left(x_i-y_i\right)}{n}

    where :math:`\mathbf{x}` is considered the reference vector.

    Parameters
    ----------
    reference : numpy.ndarray
        1D array representing the first data series
    prediction : numpy.ndarray
        1D array representing the second data series

    Returns
    -------
    bias : float
        the normalized bias

    Example
    -------
    >>> x = np.asarray([1,2,1,2])
    >>> y = np.asarray([2,1,2,1])
    >>> ComputeNormalizedBias(x,y)
    0.0
    """
    assert reference.size == prediction.size
    bias = (prediction-reference).mean()
    return bias

def ComputeAnnualMean(t,var):
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
        a 1D array of times in days since 00:00:00 1/1/1850
    vmean : numpy.ndarray
        an array assumed to be a monthly average of a variable
    
    Example
    -------
    >>> t = np.asarray([15.5,45.,74.5,105.,135.5,166.,196.5,227.5,258.,288.5,319.,349.5])
    >>> x = np.ma.array([1,1,2,3,1,2,1,1,1,3,2,3],mask=[1,1,0,0,1,0,1,1,1,0,0,0]) # mask the ones
    >>> tmean,xmean = ComputeAnnualMean(t,x)
    >>> np.allclose(xmean[0],2.5027322404371586)
    True
    """
    assert t.size >= 12 # you need at least 12 months
    begin = np.argmin(t[:11]%365)
    end   = begin+int(t[begin:].size/12.)*12
    tmean = np.ma.average(  t[begin:end].reshape((-1,12)),axis=1,weights=DAYS_PER_MONTH/365.)
    vmean = np.ma.average(var[begin:end].reshape((-1,12)),axis=1,weights=DAYS_PER_MONTH/365.)
    vmax  = np.ma.max(var[begin:end].reshape((-1,12)),axis=1)
    vmin  = np.ma.min(var[begin:end].reshape((-1,12)),axis=1)
    return tmean,vmean,vmin,vmax

def ComputeDecadalAmplitude(t,var):
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
        a 1D array of times in days since 00:00:00 1/1/1850
    vmean : numpy.ndarray
        an array assumed to be a monthly average of a variable
    """
    tmean,vmean,vmin,vmax = ComputeAnnualMean(t,var)
    A     = vmax-vmin
    begin = np.argmin(tmean[:9]%365)
    end   = begin+int(tmean[begin:].size/10.)*10
    tmean = np.apply_along_axis(np.mean,1,tmean[begin:end].reshape((-1,10)))
    Amean = np.apply_along_axis(np.mean,1,A[begin:end].reshape((-1,10)))
    Astd  = np.apply_along_axis(np.std ,1,A[begin:end].reshape((-1,10)))
    return tmean,Amean,Astd

def ComputeTrend(t,var,window=10.):
    tt    = []
    trend = []
    for i in range(t.shape[0]):
        if (t[i] < t[0]+window*365./2 or t[i] > t[-1]-window*365./2): continue
        condition = (t>(t[i]-0.5*window*365.))*(t<(t[i]+0.5*window*365.))
        x = np.ma.masked_where(condition,t  ,copy=False)
        y = np.ma.masked_where(condition,var,copy=False)
        p = np.ma.polyfit(x/365.+1850,y,1)
        tt.append(t[i])
        trend.append(p[0])
    return np.asarray(tt),np.asarray(trend)
