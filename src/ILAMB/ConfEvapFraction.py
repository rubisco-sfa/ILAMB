from .Confrontation import Confrontation
from .Variable import Variable
from netCDF4 import Dataset
from . import ilamblib as il
import numpy as np
import os
from mpi4py import MPI

import logging
logger = logging.getLogger("%i" % MPI.COMM_WORLD.rank)

def _evapfrac(sh,le,vname,energy_threshold):
    mask = ((le.data<0)+
            (sh.data<0)+
            ((le.data+sh.data)<energy_threshold))
    sh.data = np.ma.masked_array(sh.data,mask=mask)
    le.data = np.ma.masked_array(le.data,mask=mask)
    np.seterr(over='ignore',under='ignore')
    ef      = np.ma.masked_array(le.data/(le.data+sh.data),mask=mask)
    np.seterr(over='warn',under='warn')
    ef      = Variable(name      = vname,
                       unit      = "1",
                       data      = ef,
                       lat       = sh.lat,
                       lat_bnds  = sh.lat_bnds,
                       lon       = sh.lon,
                       lon_bnds  = sh.lon_bnds,
                       time      = sh.time,
                       time_bnds = sh.time_bnds)
    return sh,le,ef

class ConfEvapFraction(Confrontation):

    def __init__(self,**keywords):
        if not ('hfss_source' in keywords and 'hfls_source' in keywords):
            msg = "This confrontation requires two data sources 'hfss_source' and 'hfls_source'"
            raise il.MisplacedData(msg)
        for key in ['hfss_source','hfls_source']:
            keywords[key] = os.path.join(os.environ["ILAMB_ROOT"],keywords[key])
        keywords['source'] = keywords['hfss_source']
        self.source = keywords['source']
        super(ConfEvapFraction,self).__init__(**keywords)
        self.derived = "hfss + hfls" # just for ilamb-doctor to detect required symbols
        
    def stageData(self,m):

        energy_threshold = float(self.keywords.get("energy_threshold",20.)) # W m-2
        
        # Handle obs data
        sh_obs = Variable(filename = self.keywords["hfss_source"],
                          variable_name = "hfss",alternate_vars=['sh'],
                          t0 = None if len(self.study_limits) != 2 else self.study_limits[0],
                          tf = None if len(self.study_limits) != 2 else self.study_limits[1]).convert("W m-2")
        le_obs = Variable(filename = self.keywords["hfls_source"],
                          variable_name = "hfls",alternate_vars=['le'],
                          t0 = None if len(self.study_limits) != 2 else self.study_limits[0],
                          tf = None if len(self.study_limits) != 2 else self.study_limits[1]).convert("W m-2")
        sh_obs,le_obs,obs = _evapfrac(sh_obs,le_obs,self.variable,energy_threshold)

        # Prune out uncovered regions
        if obs.time is None: raise il.NotTemporalVariable()
        self.pruneRegions(obs)

        # Handle model data
        sh_mod = m.extractTimeSeries("hfss",
                                     alt_vars = ["FSH"],
                                     initial_time = obs.time_bnds[ 0,0],
                                     final_time   = obs.time_bnds[-1,1],
                                     lats         = None if obs.spatial else obs.lat,
                                     lons         = None if obs.spatial else obs.lon)
        le_mod = m.extractTimeSeries("hfls",
                                     alt_vars = ["EFLX_LH_TOT"],
                                     initial_time = obs.time_bnds[ 0,0],
                                     final_time   = obs.time_bnds[-1,1],
                                     lats         = None if obs.spatial else obs.lat,
                                     lons         = None if obs.spatial else obs.lon)
        sh_mod,le_mod,mod = _evapfrac(sh_mod,le_mod,self.variable,energy_threshold)

        # Make variables comparable
        obs,mod = il.MakeComparable(obs,mod,
                                    mask_ref  = True,
                                    clip_ref  = True,
                                    logstring = "[%s][%s]" % (self.longname,m.name))
        sh_obs,sh_mod = il.MakeComparable(sh_obs,sh_mod,
                                          mask_ref  = True,
                                          clip_ref  = True,
                                          logstring = "[%s][%s]" % (self.longname,m.name))
        le_obs,le_mod = il.MakeComparable(le_obs,le_mod,
                                          mask_ref  = True,
                                          clip_ref  = True,
                                          logstring = "[%s][%s]" % (self.longname,m.name))

        # Compute the mean ef
        sh_obs = sh_obs.integrateInTime(mean=True)
        le_obs = le_obs.integrateInTime(mean=True)
        np.seterr(over='ignore',under='ignore')
        obs_timeint = np.ma.masked_array(le_obs.data/(le_obs.data+sh_obs.data),mask=(sh_obs.data.mask+le_obs.data.mask))
        np.seterr(over='warn',under='warn')
        obs_timeint = Variable(name      = self.variable,
                               unit      = "1",
                               data      = obs_timeint,
                               lat       = sh_obs.lat,
                               lat_bnds  = sh_obs.lat_bnds,
                               lon       = sh_obs.lon,
                               lon_bnds  = sh_obs.lon_bnds)
        sh_mod = sh_mod.integrateInTime(mean=True)
        le_mod = le_mod.integrateInTime(mean=True)
        np.seterr(over='ignore',under='ignore')
        mod_timeint = np.ma.masked_array(le_mod.data/(le_mod.data+sh_mod.data),mask=(sh_mod.data.mask+le_mod.data.mask))
        np.seterr(over='warn',under='warn')
        mod_timeint = Variable(name      = self.variable,
                               unit      = "1",
                               data      = mod_timeint,
                               lat       = sh_mod.lat,
                               lat_bnds  = sh_mod.lat_bnds,
                               lon       = sh_mod.lon,
                               lon_bnds  = sh_mod.lon_bnds)

        return obs,mod,obs_timeint,mod_timeint

    def requires(self):
        return ['hfss','hfls'],[]

    def confront(self,m):
        r"""Confronts the input model with the observational data.

        This routine is exactly the same as Confrontation except that
        user-provided period means are passed as options to the analysis.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        """
        # Grab the data
        obs,mod,obs_timeint,mod_timeint = self.stageData(m)

        mod_file = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        obs_file = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name,      ))
        with il.FileContextManager(self.master,mod_file,obs_file) as fcm:

            # Encode some names and colors
            fcm.mod_dset.setncatts({"name" :m.name,
                                    "color":m.color,
                                    "weight":self.cweight,                                    
                                    "complete":0})
            if self.master:
                fcm.obs_dset.setncatts({"name" :"Benchmark",
                                        "color":np.asarray([0.5,0.5,0.5]),
                                        "weight":self.cweight,
                                        "complete":0})

            # Read in some options and run the mean state analysis
            mass_weighting = self.keywords.get("mass_weighting",False)
            skip_rmse      = self.keywords.get("skip_rmse"     ,False)
            skip_iav       = self.keywords.get("skip_iav"      ,True )
            skip_cycle     = self.keywords.get("skip_cycle"    ,False)
            if obs.spatial:
                il.AnalysisMeanStateSpace(obs,mod,dataset   = fcm.mod_dset,
                                          regions           = self.regions,
                                          benchmark_dataset = fcm.obs_dset,
                                          table_unit        = self.table_unit,
                                          plot_unit         = self.plot_unit,
                                          space_mean        = self.space_mean,
                                          skip_rmse         = skip_rmse,
                                          skip_iav          = skip_iav,
                                          skip_cycle        = skip_cycle,
                                          mass_weighting    = mass_weighting,
                                          ref_timeint       = obs_timeint,
                                          com_timeint       = mod_timeint)
            else:
                il.AnalysisMeanStateSites(obs,mod,dataset   = fcm.mod_dset,
                                          regions           = self.regions,
                                          benchmark_dataset = fcm.obs_dset,
                                          table_unit        = self.table_unit,
                                          plot_unit         = self.plot_unit,
                                          space_mean        = self.space_mean,
                                          skip_rmse         = skip_rmse,
                                          skip_iav          = skip_iav,
                                          skip_cycle        = skip_cycle,
                                          mass_weighting    = mass_weighting)
            fcm.mod_dset.setncattr("complete",1)
            if self.master: fcm.obs_dset.setncattr("complete",1)
        logger.info("[%s][%s] Success" % (self.longname,m.name))
