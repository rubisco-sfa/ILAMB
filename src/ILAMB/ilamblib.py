from netCDF4 import Dataset
import numpy as np

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
    """
    Computes the normalized root mean squared error (NRMSE) of two vectors.

    Given two vectors :math:`\mathbf{x}` and :math:`\mathbf{y}` of length :math:`n` the NRMSE is

    .. math:: \\frac{\sqrt{\sum_{i=1}^{n}\\frac{\left(x_i-y_i\\right)^2}{n}}}{\max(\mathbf{x})-\min(\mathbf{x})}

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
    nrmse /= (reference.max()-reference.min())
    return nrmse
