from ILAMB.Confrontation import Confrontation
from ILAMB.Confrontation import getVariableList
from ILAMB.constants import earth_rad,mid_months,lbl_months,bnd_months
from ILAMB.Variable import Variable
from ILAMB.Regions import Regions
import ILAMB.ilamblib as il
import ILAMB.Post as post
from netCDF4 import Dataset
from copy import deepcopy
import pylab as plt
import numpy as np
import os,glob

def VariableReduce(var,region="global",time=None,depth=None,lat=None,lon=None):
    ILAMBregions = Regions()
    out = deepcopy(var)
    out.data.mask += ILAMBregions.getMask(region,out)
    if time  is not None:
        out = out.integrateInTime (t0=time[0] ,tf=time[1] ,mean=True)
    if depth is not None and var.layered:
        out = out.integrateInDepth(z0=depth[0],zf=depth[1],mean=True)
    if lat   is not None:
        lat0        = np.argmin(np.abs(var.lat-lat[0]))
        latf        = np.argmin(np.abs(var.lat-lat[1]))+1
        wgt         = earth_rad*(np.sin(var.lat_bnds[:,1])-np.sin(var.lat_bnds[:,0]))[lat0:latf]
        np.seterr(over='ignore',under='ignore')
        out.data    = np.ma.average(out.data[...,lat0:latf,:],axis=-2,weights=wgt/wgt.sum())
        np.seterr(over='raise',under='raise')
        out.lat     = None
        out.lat_bnd = None
        out.spatial = False
    if lon   is not None:
        lon0        = np.argmin(np.abs(var.lon-lon[0]))
        lonf        = np.argmin(np.abs(var.lon-lon[1]))+1
        wgt         = earth_rad*(var.lon_bnds[:,1]-var.lon_bnds[:,0])[lon0:lonf]
        np.seterr(over='ignore',under='ignore')
        out.data    = np.ma.average(out.data[...,lon0:lonf],axis=-1,weights=wgt/wgt.sum())
        np.seterr(over='raise',under='raise')
        out.lon     = None
        out.lon_bnd = None
        out.spatial = False

    return out

class ConfIOMB(Confrontation):

    def __init__(self,**keywords):

        # Calls the regular constructor
        super(ConfIOMB,self).__init__(**keywords)

        # Setup a html layout for generating web views of the results
        pages = []

        # Mean State page
        pages.append(post.HtmlPage("MeanState","Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Period mean at surface",
                               "Mean regional depth profiles"])
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation","Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        with Dataset(self.source) as dset:
            for attr in dset.ncattrs():
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (attr,dset.getncattr(attr).encode('ascii','ignore'))
        self.layout = post.HtmlLayout(pages,self.longname)

    def stageData(self,m):

        mem_slab = self.keywords.get("mem_slab",100.) # Mb
        
        # peak at the obs dataset without reading much into memory,
        # this assumes that the reference dataset was encoded using
        # our datum
        with Dataset(self.source) as dset:
            climatology = True if "climatology" in dset.variables["time"].ncattrs() else False
            ot          = dset.variables["time"       ].size
            om          = dset.variables[self.variable].size *8e-6
            unit        = dset.variables[self.variable].units
            if climatology:
                t  = np.round(dset.variables[dset.variables["time"].climatology][...]/365.)*365.
                t0 = t[0,0]; tf = t[-1,1]
            else:
                if "bounds" in dset.variables["time"].ncattrs():
                    t = dset.variables[dset.variables["time"].bounds][...]
                    t0 = t[0,0]; tf = t[-1,1]
                else:
                    t = dset.variables["time"][...]
                    t0 = t[0]; tf = t[-1]
                    
        # peak at the model dataset without reading much into memory
        vname = ([v for v in [self.variable,] + self.alternate_vars if v in m.variables.keys()])[0]
        for fname in m.variables[vname]:
            mt = 0; mm = 0.; mt0 = t0; mtf = tf; shp = None
            with Dataset(fname) as dset:
                t   = dset.variables["time"]
                tb  = dset.variables[dset.variables["time"].bounds] if "bounds" in dset.variables["time"].ncattrs() else None
                if tb:
                    t,tb = il.ConvertCalendar(t,tb)
                    t = tb
                else:
                    t    = il.ConvertCalendar(t)
                i   = (t >= t0)*(t <= tf)
                mt0 = max(mt0,t[i].min())
                mtf = min(mtf,t[i].max())
                nt  = i.sum()
                v   = dset.variables[vname]
                shp = v.shape 
                mt += nt
                mm += (v.size / v.shape[0] * nt) * 8e-6
                
        # if obs is a climatology, then build a climatology in slabs
        if climatology:            
            mt0  = int(mt0/365)*365 + bnd_months[mid_months.searchsorted(mt0 % 365)]
            mtf  = int(mtf/365)*365 + bnd_months[mid_months.searchsorted(mtf % 365)+1]
            data = np.ma.zeros((12,)+shp[1:])
            dnum = np.ma.zeros(data.shape,dtype=int)
            ns   = int(mm/mem_slab)+1
            dt   = (mtf-mt0)/ns
            nm   = 0
            for i in range(ns):
                
                # find slab begin/end to the nearest month
                st0 = mt0 +  i   *dt
                st0 = int(st0/365)*365 + bnd_months[mid_months.searchsorted(st0 % 365)]
                stf = mt0 + (i+1)*dt
                stf = int(stf/365)*365 + bnd_months[mid_months.searchsorted(stf % 365)]
                v   = m.extractTimeSeries(vname,initial_time=st0,final_time=stf).trim(t=[st0,stf]).convert(unit)               
                nm += v.time.size

                # accumulate the sum for the mean cycle
                mind = mid_months.searchsorted(v.time % 365.)
                data[mind,...] += v.data
                dnum[mind,...] += 1

            data /= dnum
            obs = Variable(filename       = self.source,
                           variable_name  = self.variable,
                           alternate_vars = self.alternate_vars)
            mod = Variable(name  = obs.name,
                           unit  = obs.unit,
                           data  = data,
                           time  = obs.time,
                           lat   = v.lat,
                           lon   = v.lon,
                           depth = v.depth,
                           time_bnds  = obs.time_bnds,
                           lat_bnds   = v.lat_bnds,
                           lon_bnds   = v.lon_bnds,
                           depth_bnds = v.depth_bnds)
            yield obs,mod

            
        # if obs is historical, then we yield slabs of both
        else:
            mt0  = int(mt0/365)*365 + bnd_months[mid_months.searchsorted(mt0 % 365)]
            mtf  = int(mtf/365)*365 + bnd_months[mid_months.searchsorted(mtf % 365)+1]
            om  *= (mtf-mt0)/(tf-t0)
            ns   = int(max(om,mm)/mem_slab)+1
            dt   = (mtf-mt0)/ns
            
            for i in range(ns):
                
                # find slab begin/end to the nearest month
                st0 = mt0 +  i   *dt
                st0 = int(st0/365)*365 + bnd_months[mid_months.searchsorted(st0 % 365)]
                stf = mt0 + (i+1)*dt
                stf = int(stf/365)*365 + bnd_months[mid_months.searchsorted(stf % 365)]
                obs = Variable(filename       = self.source,
                               variable_name  = self.variable,
                               alternate_vars = self.alternate_vars,
                               t0             = st0,
                               tf             = stf).trim(t=[st0,stf])
                mod = m.extractTimeSeries(vname,initial_time=st0,final_time=stf).trim(t=[st0,stf]).convert(obs.unit)                     
                yield obs,mod
              
    def confront(self,m):
        
        for obs,mod in self.stageData(m):
            print obs
            print mod
            
