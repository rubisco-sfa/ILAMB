from Confrontation import Confrontation
from Variable import Variable
from netCDF4 import Dataset
import ilamblib as il
import Post as post
import numpy as np
import os

class ConfNBP(Confrontation):
    """A confrontation for examining the global net ecosystem carbon balance.

    This class is derived from the base Confrontation class and
    implements a specialized version of the routines stageData() and
    confront(). As the other routines are not implemented here, the
    versions from the base class will be used.

    Parameters
    ----------
    name : str
        a name for the confrontation
    srcdata : str
        full path to the observational dataset
    variable : str
        name of the variable to extract from the source dataset
    
    Other Parameters
    ----------------
    output_path : str, optional
        path into which all output from this confrontation will be generated
    alternate_vars : list of str, optional
        other accepted variable names when extracting from models
    derived : str, optional
        an algebraic expression which captures how the confrontation variable may be generated
    regions : list of str, optional
        a list of regions over which the spatial analysis will be performed (default is global)
    table_unit : str, optional
        the unit to use in the output HTML table
    plot_unit : str, optional
        the unit to use in the output images
    space_mean : bool, optional
        enable to take spatial means (as opposed to spatial integrals) in the analysis (enabled by default)
    relationships : list of ILAMB.Confrontation.Confrontation, optional
        a list of confrontations with whose data we use to study relationships
    cmap : str, optional
        the colormap to use in rendering plots (default is 'jet')
    land : str, bool
        enable to force the masking of areas with no land (default is False)

    """
            
    def stageData(self,m):
        r"""Extracts model data and integrates it over the globe to match the confrontation dataset.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model result context

        Returns
        -------
        obs : ILAMB.Variable.Variable
            the variable context associated with the observational dataset
        mod : ILAMB.Variable.Variable
            the variable context associated with the model result

        """
        # get the observational data
        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars)

        # the model data needs integrated over the globe
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars = self.alternate_vars)
        mod = mod.integrateInSpace().convert(obs.unit)
        
        obs,mod = il.MakeComparable(obs,mod,clip_ref=True)

        # sign convention is backwards
        obs.data *= -1.
        mod.data *= -1.
        
        return obs,mod

    def confront(self,m):
        r"""Confronts the input model with the observational data.

        This analysis deviates from the standard
        ILAMB.ilamblib.AnalysisMeanState. We will compare the bias and
        RMSE as before, but many of the other comparisons are not
        appropriate here.
        
        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        """
        # Grab the data
        obs,mod = self.stageData(m)
        
        obs_sum  = obs.accumulateInTime().convert("Pg")
        mod_sum  = mod.accumulateInTime().convert("Pg")
        
        obs_mean   = obs.integrateInTime(mean=True)
        mod_mean   = mod.integrateInTime(mean=True)
        bias       = obs.bias(mod)
        rmse       = obs.rmse(mod)
        bias_score = il.Score(bias,obs_mean)
        rmse_score = il.Score(rmse,obs_mean)
        
        # change names to make things easier to parse later
        obs       .name = "spaceint_of_nbp_over_global"
        mod       .name = "spaceint_of_nbp_over_global"
        obs_sum   .name = "accumulate_of_nbp_over_global"
        mod_sum   .name = "accumulate_of_nbp_over_global"
        obs_mean  .name = "Period Mean global"
        mod_mean  .name = "Period Mean global"
        bias      .name = "Bias global"       
        rmse      .name = "RMSE global"       
        bias_score.name = "Bias Score global" 
        rmse_score.name = "RMSE Score global" 

        # Dump to files
        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        mod       .toNetCDF4(results)
        mod_sum   .toNetCDF4(results)
        mod_mean  .toNetCDF4(results)
        bias      .toNetCDF4(results)
        rmse      .toNetCDF4(results)
        bias_score.toNetCDF4(results)
        rmse_score.toNetCDF4(results)
        results.close()
        
        if self.master:
            results = Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w")
            results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
            obs     .toNetCDF4(results)
            obs_sum .toNetCDF4(results)
            obs_mean.toNetCDF4(results)
            results.close()
            
        
