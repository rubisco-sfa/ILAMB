from constants import dpy,mid_months,regions as ILAMBregions
from netCDF4 import Dataset,num2date,date2num
from sympy import sympify,postorder_traversal
from scipy.stats.mstats import mode
from datetime import datetime
from cfunits import Units
from copy import deepcopy
import numpy as np

class VarNotInFile(Exception):
    def __str__(self): return "VarNotInFile"
    
class VarNotMonthly(Exception):
    def __str__(self): return "VarNotMonthly"

class VarNotInModel(Exception):
    def __str__(self): return "VarNotInModel"

class VarsNotComparable(Exception):
    def __str__(self): return self.message

class VarNotOnTimeScale(Exception):
    def __str__(self): return "VarNotOnTimeScale"

class UnknownUnit(Exception):
    def __str__(self): return "UnknownUnit"

class AreasNotInModel(Exception):
    def __str__(self): return "AreasNotInModel"

class MisplacedData(Exception):
    def __str__(self): return "MisplacedData"

class NotTemporalVariable(Exception):
    def __str__(self): return "NotTemporalVariable"

class NotSpatialVariable(Exception):
    def __str__(self): return "NotSpatialVariable"

class UnitConversionError(Exception):
    def __str__(self): return "UnitConversionError"

class AnalysisError(Exception):
    def __str__(self): return "AnalysisError"

    
def GenerateDistinctColors(N,saturation=0.67,value=0.67):
    r"""Generates a series of distinct colors.

    Computes N distinct colors using HSV color space, holding the
    saturation and value levels constant and linearly vary the
    hue. Colors are returned as a RGB tuple.

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
    HSV_tuples = [(x/float(N), saturation, value) for x in range(N)]
    RGB_tuples = map(lambda x: hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def _convertCalendar(t,unit=None):
    r"""A local function used to convert calendar representations

    This routine converts the representation of time to the ILAMB
    default: days since 1850-1-1 00:00:00 on a 365-day calendar. This
    is so we can make comparisons with data from other models and
    benchmarks.

    Parameters
    ----------
    t : netCDF4 variable
        the netCDF4 variable which represents time

    Returns
    -------
    t : numpy.ndarray
        a numpy array of the converted times
    """
    cal  = "noleap"
    if unit is None:
        unit = t.units.replace("No. of ","")
    shp  = t[...].shape
    data = t[...].flatten()
    unit = unit.replace("0000","1850")
    
    # if the time array is masked, the novalue number can cause num2date problems
    if type(data) == type(np.ma.empty(0)): data.data[data.mask] = 0
    if "calendar" in t.ncattrs(): cal = t.calendar
    if "year" in unit: 
        data *= 365.
        unit  = "days since 0-1-1"
    t = num2date(data,unit,calendar=cal) # converts data to dates

    # FIX: find a better way, converts everything to noleap calendar
    tmp = []
    for i in range(t.size): tmp.append(float((t[i].year-1850.)*365. + mid_months[t[i].month-1]))
    t = np.asarray(tmp)
    
    # if time was masked, we need to remask it as date2num doesn't handle
    if type(data) == type(np.ma.empty(0)):
        t = np.ma.masked_array(t,mask=data.mask)

    t = t.reshape(shp)
    return t

def ExtractPointTimeSeries(filename,variable,lat,lon,verbose=False):
    r"""Extracts the timeseries of a given variable at a given lat,lon
    point from a netCDF file.

    In addition to extracting the variable, this routine converts the
    representation of time to the ILAMB default: days since 1850-1-1
    00:00:00 on a 365-day calendar. This is so we can make comparisons
    with data from other models and benchmarks. We also adjust the
    representation of longitudes to be on the interval [-180,180].

    Parameters
    ----------
    filename : string
        name of the NetCDF file to read
    variable : string
        name of the variable to extract
    lat : float
        latitude in degrees at which to extract field
    lon : float 
        longitude in degrees on interval [-180,180]
    verbose : bool, optional
        enable to print additional information

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ma.core.MaskedArray
        a masked array of the extracted variable
    unit : string
        a description of the extracted unit
    """
    f = Dataset(filename)
    try:
        if verbose: print "Looking for %s in %s" % (variable,filename)
        vari = f.variables[variable]
    except:
        if verbose: print "%s is not a variable in this netCDF file" % variable
        raise VarNotInFile("%s is not a variable in this netCDF file" % variable)
    time_name = None
    lat_name  = None
    lon_name  = None
    for key in vari.dimensions:
        if "time" in key: time_name = key
        if "lat"  in key: lat_name  = key
        if "lon"  in key: lon_name  = key
    t    = _convertCalendar(f.variables[time_name])
    if lat_name is None:
        v = np.ma.masked_values(vari[...],vari._FillValue)
    else:
        lats = f.variables[lat_name][...]
        lons = f.variables[lon_name][...]
        lons = (lons<=180)*lons+(lons>180)*(lons-360)
        ilat = np.apply_along_axis(np.argmin,1,np.abs(lat[:,np.newaxis]-lats))
        ilon = np.apply_along_axis(np.argmin,1,np.abs(lon[:,np.newaxis]-lons))
        data = vari[...] # unfortunate data copy because slicing problems in netCDF4
        v    = np.ma.masked_values(data[...,ilat,ilon],vari._FillValue)
    return t,v,vari.units

def ExtractTimeSeries(filename,variable,verbose=False):
    r"""Extracts the timeseries of a given variable at a given lat,lon
    point from a netCDF file.

    In addition to extracting the variable, this routine converts the
    representation of time to the ILAMB default: days since 1850-1-1
    00:00:00 on a 365-day calendar. This is so we can make comparisons
    with data from other models and benchmarks. We also adjust the
    representation of longitudes to be on the interval [-180,180].

    Parameters
    ----------
    filename : string
        name of the NetCDF file to read
    variable : string
        name of the variable to extract
    verbose : bool, optional
        enable to print additional information

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ma.core.MaskedArray
        a masked array of the extracted variable
    unit : string
        a description of the extracted unit
    lat : float
        latitude in degrees
    lon : float 
        longitude in degrees on interval [-180,180]
    """
    f = Dataset(filename)
    try:
        if verbose: print "Looking for %s in %s" % (variable,filename)
        vari = f.variables[variable]
    except:
        if verbose: print "%s is not a variable in this netCDF file" % variable
        raise VarNotInFile("%s is not a variable in this netCDF file" % variable)
    time_name = None
    lat_name  = None
    lon_name  = None
    for key in vari.dimensions:
        if "time" in key: time_name = key
        if "lat"  in key: lat_name  = key
        if "lon"  in key: lon_name  = key
    t    = _convertCalendar(f.variables[time_name])
    lat  = f.variables[lat_name][...]
    lon  = f.variables[lon_name][...]
    var  = vari[...]
    if type(var) is type(np.asarray([])):
        var = np.ma.masked_values(vari[...],vari._FillValue,copy=False)
    return t,var,vari.units,lat,lon

def MultiModelMean(M,variable,**keywords):
    r"""Given a list of models and a variable, compute the mean across models.
    
    First we create an output grid at the specified spatial resolution
    and covering the maximum amount of overlap in the temporal
    dimension. Then we loop over each model and interpolate in space
    and time using nearest neighbor interpolation.

    Parameter
    ---------
    M : list of ILAMB.ModelResults.ModelResults
        the models to be involved in the average
    variable : string
        the variable we desire to average
    output_unit : string, optional
        if specified, will try to convert the units of the variable 
        extract to these units given. (See convert in ILAMB.constants)
    spatial_resolution : float, optional
        latitude and longitude resolution in degrees
    res_lat : float, optional
        latitude resolution in degrees, overwrites spatial_resolution
    res_lon : float, optional
        longitude resolution in degrees, overwrites spatial_resolution

    Returns
    -------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    lat : numpy.ndarray
        latitude in degrees
    lon : numpy.ndarray
        longitude in degrees on interval [-180,180]
    mean_data : numpy.ma.core.MaskedArray
        a masked array of the averaged variable
    num_model : numpy.ndarray
        an integer array of how many models contribute at each 
        point in space and time 
    models : string
        a string containing all the model names involved in the
        average
    """
    # process keywords
    output_unit        = keywords.get("output_unit","")
    spatial_resolution = keywords.get("spatial_resolution",0.5)
    res_lat            = keywords.get("res_lat",spatial_resolution)
    res_lon            = keywords.get("res_lon",spatial_resolution)

    # grab data from all models
    data   = []
    t0,tf  = 1e20,1e-20
    models = ""
    for m in M:
        if m.land_areas is None: continue
        try:
            v   = m.extractTimeSeries(variable,output_unit=output_unit)
            t   = v.time
            var = v.data
        except VarNotInModel: continue
        except VarNotMonthly: continue
        data.append((m,t,var))
        t0 = min(t0,t.min())
        tf = max(tf,t.max())
        models += m.name + ","

    # setup time range
    t = np.arange(int(round(t0/365.)),int(round(tf/365.)))
    t = t[:,np.newaxis]*365. + mid_months
    t = t.flatten()

    # setup spatial range
    lat_bnd,lon_bnd,lat,lon = GlobalLatLonGrid(spatial_resolution,
                                               from_zero = True,
                                               res_lat   = res_lat,
                                               res_lon   = res_lon)

    # sum up via interpolation
    mean_data = np.zeros((t.size,lat.size,lon.size))
    num_model = np.zeros((t.size,lat.size,lon.size),dtype=int)
    for tpl in data:
        m,tm,vm    = tpl
        rows       = np.apply_along_axis(np.argmin,1,np.abs(lat[:,np.newaxis]-m.lat))
        cols       = np.apply_along_axis(np.argmin,1,np.abs(lon[:,np.newaxis]-m.lon))
        times      = np.apply_along_axis(np.argmin,1,np.abs(t  [:,np.newaxis]-tm))
        data       = vm.data[np.ix_(times,rows,cols)]
        mask       = vm.mask[np.ix_(times,rows,cols)]
        space_mask = (m.land_areas[np.ix_(rows,cols)]<1e-12)        
        time_mask  = (t<tm.min()-2)+(t>tm.max()+2)
        mask      += time_mask[:,np.newaxis,np.newaxis]+mask
        mean_data += (mask==0)*data
        num_model += (mask==0)

    # now take averages
    mean_data = np.ma.masked_where(num_model==0,mean_data,copy=False)
    mean_data /= num_model.clip(1)
    np.ma.set_fill_value(mean_data,1e20)

    return t,lat,lon,mean_data,num_model,models[:-1]

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
        rmse  = np.exp(1-rmse/rmse0)/np.exp(1)
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
    if normalize == "score" : bias  = np.exp((1.-np.abs(bias/rmean)))/np.exp(1)
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
    ts    = np.copy(mid_months)
    shp   = (-1,12) + var.shape[1:]
    v     = var[begin:end,...].reshape(shp)
    vmean = np.ma.mean(v,axis=0)
    vstd  = np.ma.std (v,axis=0)
    
    # if the maximum is likely to be around the new year, then we need to shift times
    imax,junk = np.histogram(np.ma.argmax(v,axis=1),np.linspace(-0.5,11.5,13))
    if imax[[0,1,2,9,10,11]].sum() > imax[[3,4,5,6,7,8]].sum(): ts[:6] += 365.

    ts    = ts[np.ma.argmax(v,axis=1)]
    tmax  = np.ma.mean(ts,axis=0)
    if tmax > 365: tmax -= 365.
    tmstd = np.ma.std (ts,axis=0)
    return vmean,vstd,tmax,tmstd

def AnnualMaxTime(t,var):
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
    ts    = np.copy(mid_months)
    shp   = (-1,12) + var.shape[1:]
    v     = var[begin:end,...].reshape(shp)
    if v.shape == 3:
        imax  = np.ma.argmax(v,axis=1) # for each year and each lat/lon, which month is the maximum?
        tmax  = np.zeros(var[0,...].shape)
        for i in range(imax.shape[1]):
            for j in range(imax.shape[2]):
                if var.mask[0,i,j]: continue
                hist,bins = np.histogram(imax[:,i,j],np.linspace(-0.5,11.5,13))
                tmax[i,j] = ts[np.argmax(hist)]
        mask = var[0,...].mask
    else:
        imax  = np.ma.argmax(v,axis=1) # for each year and each lat/lon, which month is the maximum?
        tmax  = np.zeros(var[0,...].shape)
        for i in range(imax.shape[1]):
            if var.mask[0,i]: continue
            hist,bins = np.histogram(imax[:,i],np.linspace(-0.5,11.5,13))
            tmax[i] = ts[np.argmax(hist)]
        mask = False
    tmax  = np.ma.masked_array(tmax,mask=mask)
    tmstd = np.ma.masked_array(np.zeros(tmax.shape))
    return tmax,tmstd

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
    np.seterr(over='ignore',under='ignore')
    vbar = (var*areas).sum(axis=-1).sum(axis=-1)
    np.seterr(over='raise',under='raise')
    return vbar

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

def GlobalLatLonGrid(res,**keywords):
    r"""Generates a latitude/longitude grid at a desired resolution
    
    Computes 1D arrays of latitude and longitude values which
    correspond to cell interfaces and centroids at a given resolution.

    Parameters
    ----------
    res : float
        the desired resolution of the grid in degrees
    from_zero : boolean
        sets longitude convention { True:(0,360), False:(-180,180) }

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
    from_zero = keywords.get("from_zero",False)
    res_lat   = keywords.get("res_lat",res)
    res_lon   = keywords.get("res_lon",res)
    nlon    = int(360./res_lon)+1
    nlat    = int(180./res_lat)+1
    lon_bnd = np.linspace(-180,180,nlon)
    if from_zero: lon_bnd += 180
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

def MaxMonthMode(t,var):
    r"""For each cell/site, return the month where the maximum of var is most frequently realized.

    Parameters
    ----------
    t : numpy.ndarray
        a 1D array of times in days since 1850-01-01 00:00:00
    var : numpy.ma.core.MaskedArray
        a masked array representing a global gridded data of shape (time,lat,lon) 
        or site data of shape (time,site) or even a scalar time series of shape (time).
    
    Returns
    -------
    mode : numpy.ma.core.MaskedArray
        a masked array of the Julian day of the year representing the middle of the
        month where the input variable is most often at its maximum. Masked values 
        reflect where no data is given at all for a specific cell/site.

    Notes
    -----
    This routine can be generalized to extract any kind of cycle
    information. Perhaps an option for the user to specify a cycle
    time and we compute the rest from that. Also, it appears to be
    quite slow. It could be the mode function from scipy.mstats
    """
    assert t.ndim == 1
    assert t.size == var.shape[0]
    assert np.allclose((t[1:]-t[:-1]).mean(),30,atol=3) # only for monthly means
    begin = np.argmin(t[:11]%365)
    end   = begin+int(t[begin:].size/12.)*12
    ts    = mid_months
    shp   = (-1,12) + var.shape[1:]
    v     = var[begin:end,...].reshape(shp)
    mask  = v.mask
    if mask.size > 1: mask = np.apply_along_axis(np.all,1,v.mask) # masks years where no data exists
    maxmonth = np.ma.masked_array(np.argmax(v,axis=1),mask=mask,dtype=int)
    print maxmonth.shape
    imode,nmode = mode(maxmonth)
    if mask.size > 1: mask = np.apply_along_axis(np.all,0,mask) # masks cells where no data exists
    return np.ma.masked_array(ts[np.asarray(imode,dtype='int')],mask=mask)

def SympifyWithArgsUnits(expression,args,units):
    """
    
    """
    expression = sympify(expression)
    
    # We need to do what sympify does but also with unit
    # conversions. So we traverse the expression tree in post order
    # and take actions based on the kind of operation being performed.
    for expr in postorder_traversal(expression):

        if expr.is_Atom: continue        
        ekey = str(expr) # expression key
        
        if expr.is_Add:

            # Addition will require that all args should be the same
            # unit. As a convention, we will try to conform all units
            # to the first variable's units. 
            key0 = None
            for arg in expr.args:
                key = str(arg)
                if not args.has_key(key): continue
                if key0 is None:
                    key0 = key
                else:
                    # Conform these units to the units of the first arg
                    Units.conform(args[key],
                                  Units(units[key]),
                                  Units(units[key0]),
                                  inplace=True)
                    units[key] = units[key0]

            # Now add the result of the addition the the disctionary
            # of arguments.
            args [ekey] = sympify(str(expr),locals=args)
            units[ekey] = units[key0]

        elif expr.is_Pow:

            assert len(expr.args) == 2 # check on an assumption
            power = float(expr.args[1])
            args [ekey] = args[str(expr.args[0])]**power
            units[ekey] = Units(units[str(expr.args[0])])
            units[ekey] = units[ekey]**power
        
        elif expr.is_Mul:

            unit = Units("1")
            for arg in expr.args:
                key   = str(arg)
                if units.has_key(key): unit *= Units(units[key])
        
            args [ekey] = sympify(str(expr),locals=args)
            units[ekey] = Units(unit).formatted()

    return args[ekey],units[ekey]


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
    time_bnds : numpy.ndarray
        A 1D array of time bounds in days since 1850-01-01 00:00:00
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
        while alternate_vars.count(None) > 0: alternate_vars.pop(alternate_vars.index(None))
        for var_name in alternate_vars:
            if var_name in f.variables.keys():
                found = True
                var   = f.variables[var_name]
    if found == False:
        alternate_vars.insert(0,variable_name)
        raise RuntimeError("Unable to find [%s] in the file: %s" % (",".join(alternate_vars),filename))
    time_name     = None
    time_bnd_name = None
    lat_name      = None
    lon_name      = None
    data_name     = None
           
    for key in var.dimensions:
        if "time" in key:
            time_name = key
            t = f.variables[key]
            if "bounds" in t.ncattrs():
                time_bnd_name = t.getncattr("bounds")
        if "lat"  in key: lat_name  = key
        if "lon"  in key: lon_name  = key
        if "data" in key: data_name = key
    if time_name is None:
        t = None
    else:
        t = _convertCalendar(f.variables[time_name])
    if time_bnd_name is None:
        t_bnd = None
    else:
        t_bnd = _convertCalendar(f.variables[time_bnd_name],unit=f.variables[time_name].units).T
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
    
    return v,units,variable_name,t,t_bnd,lat,lon,data

        
def Score(var,normalizer,eps=1e-12,theta=0.5,error=1.0):
    """Remaps a normalized variable to the interval [0,1].

    Parameters
    ----------
    var : ILAMB.Variable.Variable
        The variable to normalize, usually represents an error of some sort
    normalizer : ILAMB.Variable.Variable
        The variable by which we normalize 
    eps : float, optional
        A small parameter used to avoid division by zero, defaults to 1e-12
    theta,error : float, optional
        Parameters which control the mapping, see the notes.
    """
    score = deepcopy(var)
    np.seterr(over='ignore',under='ignore')
    # Compute the absolute relative error
    score.data = np.abs(score.data/(np.abs(normalizer.data)+eps))
    # Remap the error to [0,1]
    score.data = np.exp(np.log(theta)/error*score.data)
    np.seterr(over='raise',under='raise')
    score.name = score.name.replace("bias","bias_score")
    score.name = score.name.replace("rmse","rmse_score")
    score.name = score.name.replace("iav" ,"iav_score")
    score.unit = "1"
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
    from Variable import Variable
    return Variable(data  = (1+np.cos(np.abs(phase_shift.data)/365*2*np.pi))*0.5,
                    unit  = "1",
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
            obs_spaceint   [region] = obs            .integrateInSpace(region=region,mean=True)
            mod_period_mean[region] = mod_timeint    .integrateInSpace(region=region,mean=space_mean) ##
            
            # Compute the scalar means over the specified region.
            bias           [region] = bias_map       .integrateInSpace(region=region,mean=space_mean) ##
            rmse           [region] = rmse_map       .integrateInSpace(region=region,mean=space_mean) ##
            bias_score     [region] = bias_score_map .integrateInSpace(region=region,mean=True)
            rmse_score     [region] = rmse_score_map .integrateInSpace(region=region,mean=True)
            mod_spaceint   [region] = mod            .integrateInSpace(region=region,mean=True)
            
        else:

            # We need to check if there are datasites in this
            # region. If not, we will just skip the region.
            lats,lons = ILAMBregions[region]
            if ((obs.lat>lats[0])*(obs.lat<lats[1])*(obs.lon>lons[0])*(obs.lon<lons[1])).sum() == 0: continue
            
            # Compute the scalar period mean over sites in the specified region.
            obs_period_mean[region],junk = obs_timeint    .siteStats(region=region)
            obs_spaceint   [region],junk = obs            .siteStats(region=region)
            mod_period_mean[region],junk = mod_timeint    .siteStats(region=region)
            bias           [region],junk = bias_map       .siteStats(region=region)
            rmse           [region],junk = rmse_map       .siteStats(region=region)
            bias_score     [region],junk = bias_score_map .siteStats(region=region)
            rmse_score     [region],junk = rmse_score_map .siteStats(region=region)
            mod_spaceint   [region],junk = mod            .siteStats(region=region)

        # Compute the spatial variability.
        std[region],R[region],sd_score[region] = obs_timeint.spatialDistribution(mod_timeint,region=region)
        
        # Change variable names to make things easier to parse later.
        obs_period_mean[region].name = "period_mean_of_%s_over_%s" % (obs.name,region)
        obs_spaceint   [region].name = "spaceint_of_%s_over_%s"    % (obs.name,region)
        mod_period_mean[region].name = "period_mean_of_%s_over_%s" % (obs.name,region)
        bias           [region].name = "bias_of_%s_over_%s"        % (obs.name,region)
        rmse           [region].name = "rmse_of_%s_over_%s"        % (obs.name,region)
        bias_score     [region].name = "bias_score_of_%s_over_%s"  % (obs.name,region)
        rmse_score     [region].name = "rmse_score_of_%s_over_%s"  % (obs.name,region)
        sd_score       [region].name = "sd_score_of_%s_over_%s"    % (obs.name,region)
        mod_spaceint   [region].name = "spaceint_of_%s_over_%s"    % (obs.name,region)
        std            [region].name = "std_of_%s_over_%s"         % (obs.name,region)
        R              [region].name = "corr_of_%s_over_%s"        % (obs.name,region)
        
    # More variable name changes
    obs_timeint.name  = "timeint_of_%s"   % obs.name
    mod_timeint.name  = "timeint_of_%s"   % obs.name
    bias_map.name     = "bias_map_of_%s"  % obs.name

    # Unit conversions
    if table_unit is not None:
        for var in [obs_period_mean,mod_period_mean,bias,rmse]:
            if type(var) == type({}):
                for key in var.keys(): var[key].convert(table_unit)
            else:
                var.convert(plot_unit)
    if plot_unit is not None:
        for var in [mod_timeint,obs_timeint,bias_map,mod_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].convert(plot_unit)
            else:
                var.convert(plot_unit)

    # Optionally dump results to a NetCDF file
    if dataset is not None:
        for var in [mod_period_mean,bias,rmse,bias_score,rmse_score,sd_score,
                    mod_timeint,bias_map,std,R,mod_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(dataset)
            else:
                var.toNetCDF4(dataset)
    if benchmark_dataset is not None:
        for var in [obs_period_mean,obs_timeint,obs_spaceint]:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(benchmark_dataset)
            else:
                var.toNetCDF4(benchmark_dataset)

    # The next analysis bit requires we are dealing with monthly mean data
    if not obs.monthly: return
        
    # Compute of the phase shift. First we compute the mean
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
    obs_mean_cycle  = {}
    mod_mean_cycle  = {}
    shift = {}; shift_score = {}
    iav_score = {};
    for region in regions:
        
        if spatial:

            # Compute the scalar integral over the specified region.
            obs_mean_cycle [region] = obs_cycle      .integrateInSpace(region=region,mean=True)
            
            # Compute the scalar means over the specified region.
            shift          [region] = shift_map      .integrateInSpace(region=region,mean=True)
            shift_score    [region] = shift_score_map.integrateInSpace(region=region,mean=True)
            iav_score      [region] = iav_score_map  .integrateInSpace(region=region,mean=True)
            mod_mean_cycle [region] = mod_cycle      .integrateInSpace(region=region,mean=True)
            
        else:

            # We need to check if there are datasites in this
            # region. If not, we will just skip the region.
            lats,lons = ILAMBregions[region]
            if ((obs.lat>lats[0])*(obs.lat<lats[1])*(obs.lon>lons[0])*(obs.lon<lons[1])).sum() == 0: continue
            
            # Compute the scalar period mean over sites in the specified region.
            obs_mean_cycle [region],junk = obs_cycle      .siteStats(region=region)
            shift          [region],junk = shift_map      .siteStats(region=region)
            shift_score    [region],junk = shift_score_map.siteStats(region=region)
            iav_score      [region],junk = iav_score_map  .siteStats(region=region)
            mod_mean_cycle [region],junk = mod_cycle      .siteStats(region=region)
        
        # Change variable names to make things easier to parse later.
        obs_mean_cycle [region].name = "cycle_of_%s_over_%s"       % (obs.name,region)
        shift          [region].name = "shift_of_%s_over_%s"       % (obs.name,region)
        shift_score    [region].name = "shift_score_of_%s_over_%s" % (obs.name,region)
        iav_score      [region].name = "iav_score_of_%s_over_%s"   % (obs.name,region)
        mod_mean_cycle [region].name = "cycle_of_%s_over_%s"       % (obs.name,region)
        
    # More variable name changes
    obs_maxt_map.name = "phase_map_of_%s" % obs.name
    mod_maxt_map.name = "phase_map_of_%s" % obs.name
    shift_map.name    = "shift_map_of_%s" % obs.name

    # Unit conversions
    if plot_unit is not None:
        for var in [mod_mean_cycle]:
            if type(var) == type({}):
                for key in var.keys(): var[key].convert(plot_unit)
            else:
                var.convert(plot_unit)

    # Optionally dump results to a NetCDF file
    if dataset is not None:
        for var in [shift,shift_score,iav_score,
                    mod_maxt_map,shift_map,
                    mod_mean_cycle]:
            if type(var) == type({}):
                for key in var.keys(): var[key].toNetCDF4(dataset)
            else:
                var.toNetCDF4(dataset)
    if benchmark_dataset is not None:
        for var in [obs_maxt_map,obs_mean_cycle]:
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

    # grab regions
    regions = keywords.get("regions",["global"])
    
    # convert to plot units
    dep_plot_unit = keywords.get("dep_plot_unit",dep_var.unit)
    ind_plot_unit = keywords.get("ind_plot_unit",ind_var.unit)    
    if dep_plot_unit is not None: dep_var.convert(dep_plot_unit)
    if ind_plot_unit is not None: ind_var.convert(ind_plot_unit)

    # if the variables are temporal, we need to get period means
    if dep_var.temporal: dep_var = dep_var.integrateInTime(mean=True)
    if ind_var.temporal: ind_var = ind_var.integrateInTime(mean=True)
    mask = dep_var.data.mask + ind_var.data.mask

    # analysis over regions
    for region in regions:

        lats,lons = ILAMBregions[region]
        rmask     = (np.outer((dep_var.lat>lats[0])*(dep_var.lat<lats[1]),
                              (dep_var.lon>lons[0])*(dep_var.lon<lons[1]))==0)
        rmask    += mask
        x    = ind_var.data[rmask==0].flatten()
        y    = dep_var.data[rmask==0].flatten()

        # Compute 2D histogram, normalized by number of datapoints
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
        grp = dataset.createGroup("%s_relationship_%s" % (region,rname))
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

def ClipTime(v,t0,tf):
    r"""
    """
    begin = np.argmin(np.abs(v.time_bnds[0,:]-t0))
    end   = np.argmin(np.abs(v.time_bnds[1,:]-tf))
    while v.time_bnds[0,begin] > t0:
        begin    -= 1
        if begin <= 0:
            begin = 0
            break
    while v.time_bnds[1,  end] < tf:
        end      += 1
        if end   >= v.time.size-1:
            end   = v.time.size-1
            break
    v.time      = v.time     [  begin:(end+1)    ]
    v.time_bnds = v.time_bnds[:,begin:(end+1)    ]
    v.data      = v.data     [  begin:(end+1),...]
    return v
    
def MakeComparable(ref,com,**keywords):
    r"""Make two variables comparable.

    Given a reference variable and a comparison variable, make the two
    variables comparable or raise an exception explaining why they are
    not.

    Parameters
    ----------
    ref : ILAMB.Variable.Variable
        the reference variable object
    com : ILAMB.Variable.Variable
        the comparison variable object
    clip_ref : bool, optional
        enable in order to clip the reference variable time using the
        limits of the comparison variable (defult is False)
    mask_ref : bool, optional
        enable in order to mask the reference variable using an
        interpolation of the comparison variable (defult is False)
    eps : float, optional
        used to determine how close you can be to a specific time
        (expressed in days since 1-1-1850) and still be considered the
        same time (default is 30 minutes)
    window : float, optional
        specify to extend the averaging intervals (in days since
        1-1-1850) when a variable must be coarsened (default is 0)

    Returns
    -------
    ref : ILAMB.Variable.Variable
        the modified reference variable object
    com : ILAMB.Variable.Variable
        the modified comparison variable object

    """
    # Process keywords
    clip_ref = keywords.get("clip_ref",False)
    mask_ref = keywords.get("mask_ref",False)
    eps      = keywords.get("eps"     ,30./60./24.)
    window   = keywords.get("window"  ,0.)
    
    # If one variable is temporal, then they both must be
    if ref.temporal != com.temporal:
        msg  = "\n  The datasets are not both uniformly temporal:\n"
        msg += "    reference = %s, comparison = %s\n" % (ref.temporal,com.temporal)
        raise VarsNotComparable(msg)

    # If the reference is spatial, the comparison must be
    if ref.spatial and not com.spatial:
        msg = "\n  The reference data is spatial but the comparison data is not\n"
        raise VarsNotComparable(msg)

    # If the reference represents observation sites, extract them from
    # the comparison
    if ref.ndata is not None and com.spatial: com = com.extractDatasites(ref.lat,ref.lon)

    # If both variables represent observations sites, make sure you
    # have the same number of sites and that they represent the same
    # location. Note this is after the above extraction so at this
    # point the ndata field of both variables should be equal.
    if ref.ndata != com.ndata:
        msg  = "\n  One or both datasets are understood as site data but differ in number of sites.\n"
        msg += "    number of sites of reference = %d, comparison = %d\n" % (ref.ndata,com.ndata)
        raise VarsNotComparable(msg)
    if ref.ndata is not None:
        if not (np.allclose(ref.lat,com.lat) or np.allclose(ref.lon,com.lon)):
            msg  = "\n  Both datasets represent sites, but the locations are different:"
            msg += "    Maximum difference lat = %.f, lon = %.f" % (np.abs((ref.lat-com.lat)).max(),
                                                                    np.abs((ref.lon-com.lon)).max())
            raise VarsNotComparable(msg)
        
    if ref.temporal:

        # If the reference time scale is significantly larger than the
        # comparison, coarsen the comparison
        if np.log10(ref.dt/com.dt) > 0.5:
            com = com.coarsenInTime(ref.time_bnds,window=window)
        
        # Time bounds of the reference dataset
        t0  = ref.time_bnds[0, 0]
        tf  = ref.time_bnds[1,-1]

        # Find the comparison time range which fully encompasses the reference
        com = ClipTime(com,t0,tf)
        
        if clip_ref:

            # We will clip the reference dataset too
            t0  = max(t0,com.time_bnds[0, 0])
            tf  = min(tf,com.time_bnds[1,-1])
            ref = ClipTime(ref,t0,tf)

        else:
            
            # The comparison dataset needs to fully cover the reference in time
            if (com.time_bnds[0, 0] > (t0+eps) or
                com.time_bnds[1,-1] < (tf-eps)):
                msg  = "\n  Comparison dataset does not cover the time frame of the reference:\n"
                msg += "    t0: %.16e <= %.16e (%s)\n" % (com.time_bnds[0, 0],t0+eps,com.time_bnds[0, 0] <= (t0+eps))
                msg += "    tf: %.16e >= %.16e (%s)\n" % (com.time_bnds[1,-1],tf-eps,com.time_bnds[1,-1] >= (tf-eps))
                raise VarsNotComparable(msg)

        # Check that we now are on the same time intervals
        if ref.time.size != com.time.size:
            msg  = "\n  Datasets have differing numbers of time intervals:\n"
            msd += "    reference = %d, comparison = %d\n" % (ref.time.size,com.time.size)
            raise VarsNotComparable(msg)
        if not np.allclose(ref.time_bnds,com.time_bnds,atol=0.1*ref.dt):
            msg  = "\n  Datasets are defined on different time intervals"
            raise VarsNotComparable(msg)
                                   
    # Apply the reference mask to the comparison dataset and
    # optionally vice-versa
    mask = ref.interpolate(time=com.time,lat=com.lat,lon=com.lon)
    com.data.mask = mask.data.mask
    if mask_ref:
        mask = com.interpolate(time=ref.time,lat=ref.lat,lon=ref.lon)
        ref.data.mask = mask.data.mask
            
    return ref,com
