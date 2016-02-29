from Confrontation import Confrontation
from Variable import Variable
from netCDF4 import Dataset
import ilamblib as il
import Post as post
import numpy as np
import os

class ConfNBP(Confrontation):
    """A generic class for confronting model results with observational data.

    This class is meant to provide the user with a simple way to
    specify observational datasets and compare them to model
    results. A generic analysis routine is called which checks mean
    states of the variables, afterwhich the results are tabulated and
    plotted automatically. A HTML page is built dynamically as plots
    are created based on available information and successful
    analysis.

    Parameters
    ----------
    name : str
        a name for the confrontation
    srcdata : str
        full path to the observational dataset
    variable_name : str
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
    def __init__(self,name,srcdata,variable_name,**keywords):
        
        # Initialize
        self.master         = True
        self.name           = name
        self.srcdata        = srcdata
        self.variable_name  = variable_name
        self.output_path    = keywords.get("output_path","_build/%s/" % self.name)
        self.alternate_vars = keywords.get("alternate_vars",[])
        self.derived        = keywords.get("derived",None)
        self.regions        = ["global"]
        self.data           = None
        self.cmap           = keywords.get("cmap","jet")
        self.land           = keywords.get("land",False)
        self.limits         = None
        self.longname       = self.output_path
        self.longname       = self.longname.replace("//","/").replace("./","").replace("_build/","")
        if self.longname[-1] == "/": self.longname = self.longname[:-1]
        self.longname       = "/".join(self.longname.split("/")[1:])
        self.table_unit     = keywords.get("table_unit",None)
        self.plot_unit      = keywords.get("plot_unit",None)
        self.space_mean     = keywords.get("space_mean",True)        
        self.relationships  = keywords.get("relationships",None)

        # Make sure the source data exists
        try:
            os.stat(self.srcdata)
        except:
            msg  = "\n\nI am looking for data for the %s confrontation here\n\n" % self.name
            msg += "%s\n\nbut I cannot find it. " % self.srcdata
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?\n"
            raise il.MisplacedData(msg)

        # Setup a html layout for generating web views of the results
        self.layout = post.HtmlLayout(self,regions=self.regions)
        self.layout.setHeader("CNAME / RNAME / MNAME")
        self.layout.setSections(["Temporally integrated period mean",
                                 "Spatially integrated regional mean",
                                 "Period Mean Relationships"])

        # Define relative weights of each score in the overall score
        # (FIX: need some way for the user to modify this)
        self.weight = {"bias_score" :1.,
                       "rmse_score" :2.,
                       "shift_score":1.,
                       "iav_score"  :1.,
                       "sd_score"   :1.}
            
    def stageData(self,m):
        r"""Extracts model data which matches the observational dataset defined along with this confrontation.

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
        obs = Variable(filename       = self.srcdata,
                       variable_name  = self.variable_name,
                       alternate_vars = self.alternate_vars)

        # the model data needs integrated over the globe
        mod = m.extractTimeSeries(self.variable_name,
                                  alt_vars = self.alternate_vars)
        mod = mod.integrateInSpace().convert(obs.unit)
        
        obs,mod = il.MakeComparable(obs,mod,clip_ref=True)

        # sign convention is backwards
        obs.data *= -1.
        mod.data *= -1.
        
        return obs,mod

    def confront(self,m):
        r"""Confronts the input model with the observational data.

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
        obs_mean  .name = "period_mean_of_nbp_over_global"
        mod_mean  .name = "period_mean_of_nbp_over_global"
        bias      .name = "bias_of_nbp_over_global"       
        rmse      .name = "rmse_of_nbp_over_global"       
        bias_score.name = "bias_score_of_nbp_over_global" 
        rmse_score.name = "rmse_score_of_nbp_over_global" 

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
            
        
