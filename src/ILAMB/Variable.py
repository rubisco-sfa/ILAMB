import ilamblib as il
from constants import spd,spy,regions as ILAMBregions
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
        if time is not None: self.temporal = True
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
                assert np.allclose(t,dataset.variables["time"][...],atol=15)
            else:
                dataset.createDimension("time")
                T = dataset.createVariable("time","double",("time"))
                T.setncattr("units","days since 1850-01-01 00:00:00")
                T.setncattr("calendar","noleap")
                T.setncattr("axis","T")
                T.setncattr("long_name","time")
                T.setncattr("standard_name","time")
                T[...] = t

        if self.temporal: _checkTime(self.time,dataset)
        
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

    def convert(self,unit):
        pass

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
        return Variable(integral,unit,lat=self.lat,lon=self.lon,area=self.area)

    def bias(self,var):
        pass

    def RMSE(self,var):
        pass

    def plot(self,ax):
        if self.temporal and not self.spatial:
            ax.plot(self.time,self.data,'-')
        return ax

if __name__ == "__main__":
    from netCDF4 import Dataset
    from os import environ
    d   = Dataset("%s/DATA/gpp/FLUXNET-MTE/derived/gpp.nc" % environ["ILAMB_ROOT"],mode="r")
    t   = d.variables["time"]
    lat = d.variables["lat"]
    lon = d.variables["lon"]
    gpp = d.variables["gpp"]
    v   = Variable(gpp[...],gpp.units,time=t[...],lon=lon[...],lat=lat[...])
    va  = v.integrateInSpace(region="amazon")
    vt  = va.integrateInTime()

    #va.plot()
