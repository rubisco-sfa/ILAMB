from ILAMB.Confrontation import Confrontation
from mpl_toolkits.basemap import Basemap
from ILAMB.Variable import Variable
from netCDF4 import Dataset
import ILAMB.ilamblib as il
import numpy as np
import os

class ConfEvapFraction(Confrontation):

    def stageData(self,m):

        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars)
        if obs.time is None: raise il.NotTemporalVariable()
        self.pruneRegions(obs)
        
        sh = m.extractTimeSeries("hfss",
                                 initial_time = obs.time_bnds[ 0,0],
                                 final_time   = obs.time_bnds[-1,1],
                                 lats         = None if obs.spatial else obs.lat,
                                 lons         = None if obs.spatial else obs.lon)
        le = m.extractTimeSeries("hfls",
                                 initial_time = obs.time_bnds[ 0,0],
                                 final_time   = obs.time_bnds[-1,1],
                                 lats         = None if obs.spatial else obs.lat,
                                 lons         = None if obs.spatial else obs.lon)
        sh.data = np.abs(sh.data)
        le.data = np.abs(le.data)
        mod     = Variable(name      = self.variable,
                           unit      = "1",
                           data      = le.data/(le.data+sh.data),
                           lat       = sh.lat,
                           lat_bnds  = sh.lat_bnds,
                           lon       = sh.lon,
                           lon_bnds  = sh.lon_bnds,
                           time      = sh.time,
                           time_bnds = sh.time_bnds)
        
        obs,mod = il.MakeComparable(obs,mod,
                                    mask_ref  = True,
                                    clip_ref  = True,
                                    logstring = "[%s][%s]" % (self.longname,m.name))
        
        return obs,mod
    

