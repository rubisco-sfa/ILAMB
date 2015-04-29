import ilamblib as il
import Post as post
from constants import spd,spy,convert,regions as ILAMBregions
import numpy as np
import pylab as plt

class Variable:

    def __init__(self,data,unit,name="unnamed",time=None,lat=None,lon=None,area=None):
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
            dt = (time[1:]-time[:-1]).mean()
            if np.allclose(dt,30,atol=3): self.monthly = True
        
    def toNetCDF4(self,dataset):

        def _checkTime(t,dataset):
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
            if "lat" in dataset.dimensions.keys():
                assert lat.shape == dataset.variables["lat"][...].shape
                assert np.allclose(lat,dataset.variables["lat"][...])
                assert lon.shape == dataset.variables["lon"][...].shape
                assert np.allclose(lon,dataset.variables["lon"][...])
            else:
                dataset.createDimension("lon")
                X = dataset.createVariable("lon","double",("lon"))
                X.setncattr("units","degrees_east")
                X.setncattr("axis","X")
                X.setncattr("long_name","longitude")
                X.setncattr("standard_name","longitude")
                X[...] = lon
                dataset.createDimension("lat")
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
        for attr in self.attributes: V.setncattr(attr,self.attributes[attr])
        V[...] = self.data

    
    def integrateInSpace(self,region=None):
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

    def annualCycle(self,):
        pass

    def convert(self,unit):
        r"""Lame attempt to handle unit conversions"""
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
        def _make_bnds(x):
            bnds       = np.zeros(x.size+1)
            bnds[1:-1] = 0.5*(x[1:]+x[:-1])
            bnds[0]    = x[0] -0.5*(x[ 1]-x[ 0])
            bnds[-1]   = x[-1]+0.5*(x[-1]-x[-2])
            return bnds
        assert var.unit == self.unit
        lat_bnd1 = _make_bnds(self.lat)
        lon_bnd1 = _make_bnds(self.lon)
        lat_bnd2 = _make_bnds( var.lat)
        lon_bnd2 = _make_bnds( var.lon)
        lat_bnd,lon_bnd,lat,lon,error = il.TrueError(lat_bnd1,lon_bnd1,self.lat,self.lon,self.data,
                                                     lat_bnd2,lon_bnd2, var.lat, var.lon, var.data)
        return Variable(error,var.unit,lat=lat,lon=lon,name="%s_minus_%s" % (var.name,self.name))

    def integrateInTime(self):
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
        mint = max(var.time.min(),self.time.min())
        maxt = min(var.time.max(),self.time.max())
        b     = np.argmin(np.abs( var.time-mint)); e     = np.argmin(np.abs( var.time-maxt)) 
        ref_b = np.argmin(np.abs(self.time-mint)); ref_e = np.argmin(np.abs(self.time-maxt)) 
        comparable = (var.time[b:e].shape == self.time[ref_b:ref_e].shape)
        if comparable:
            comparable = np.allclose(var.time[b:e],self.time[ref_b:ref_e],atol=0.5*self.dt)
        return comparable,b,e,ref_b,ref_e

    def bias(self,var,normalize="none"):
        comparable,b,e,vb,ve = self._overlap(var)
        if not comparable: raise il.VarsNotComparable()
        mw = il.MonthlyWeights(self.time[b:e])
        return il.Bias(self.data[b:e],var.data[vb:ve],normalize=normalize)

    def RMSE(self,var,normalize="none"):
        comparable,b,e,vb,ve = self._overlap(var)
        if not comparable: raise il.VarsNotComparable()
        mw = il.MonthlyWeights(self.time[b:e])
        return il.RootMeanSquaredError(self.data[b:e],var.data[vb:ve],normalize=normalize)

    def plot(self,ax,**keywords):
        if self.temporal and not self.spatial:
            ax.plot(self.time,self.data,'-')
        if not self.temporal and self.spatial:
            vmin   = keywords.get("vmin",self.data.min())
            vmax   = keywords.get("vmax",self.data.max())
            region = keywords.get("region","global")
            cmap   = keywords.get("cmap","rainbow")
            ax     = post.GlobalPlot(self.lat,self.lon,self.data,ax,
                                     vmin   = vmin  , vmax = vmax,
                                     region = region, cmap = cmap)
        return ax
