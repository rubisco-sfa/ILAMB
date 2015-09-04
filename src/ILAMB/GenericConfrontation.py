import ilamblib as il
from Variable import *
from constants import four_code_regions
import os,glob
from netCDF4 import Dataset
import Post as post
import pylab as plt

class GenericConfrontation:
    def __init__(self,name,srcdata,variable_name,**keywords):

        # Initialize 
        self.name           = name
        self.srcdata        = srcdata
        self.variable_name  = variable_name
        self.output_path    = keywords.get("output_path","_build/%s/" % self.name)
        self.alternate_vars = keywords.get("alternate_vars",[])
        self.regions        = keywords.get("regions",four_code_regions)

        # Make sure the source data exists
        try:
            os.stat(self.srcdata)
        except:
            msg  = "\n\nI am looking for data for the %s confrontation here\n\n" % self.name
            msg += "%s\n\nbut I cannot find it. " % self.srcdata
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?\n"
            raise il.MisplacedData(msg)
        
        # Build the output directory (fix for parallel somehow,
        # perhaps a keyword to make this the master?)
        dirs = self.output_path.split("/")
        for i,d in enumerate(dirs):
            dname = "/".join(dirs[:(i+1)])
            if not os.path.isdir(dname): os.mkdir(dname)

        # Setup a html layout for generating web views of the results
        self.layout = post.HtmlLayout(self,regions=self.regions)
        self.layout.setHeader("CNAME / RNAME / MNAME")

        # Define relative weights of each score in the overall score
        # (FIX: need some way for the user to modify this)
        self.weight = {"Bias Score"                    :1.,
                       "RMSE Score"                    :2.,
                       "Seasonal Cycle Score"          :1.,
                       "Interannual Variability Score" :1.,
                       "Spatial Distribution Score"    :1.}
        
    def confront(self,m):
        r"""Confronts the input model with the observational data.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results
        """
        # Read in the data, and perform consistency checks depending
        # on the data types found
        obs = Variable(filename=self.srcdata,variable_name=self.variable_name,alternate_vars=self.alternate_vars)
        if obs.spatial:
            mod = m.extractTimeSeries(self.variable_name,
                                      initial_time = obs.time[ 0],
                                      final_time   = obs.time[-1])
        else:
            mod = m.extractTimeSeries(self.variable_name,
                                      lats         = obs.lat,
                                      lons         = obs.lon,
                                      initial_time = obs.time[ 0],
                                      final_time   = obs.time[-1])
        t0 = max(obs.time[ 0],mod.time[ 0])
        tf = min(obs.time[-1],mod.time[-1])
        for var in [obs,mod]:
            begin = np.argmin(np.abs(var.time-t0))
            end   = np.argmin(np.abs(var.time-tf))+1
            var.time = var.time[begin:end]
            var.data = var.data[begin:end,...]
        assert obs.time.shape == mod.time.shape       # same number of times
        assert np.allclose(obs.time,mod.time,atol=14) # same times +- two weeks
        assert obs.ndata == mod.ndata                 # same number of datasites

        # Open a dataset for recording the results of this confrontation
        results = Dataset("%s/%s.nc" % (self.output_path,m.name),mode="w")
        AnalysisFluxrate(obs,mod,dataset=results,regions=self.regions)

    def postProcessFromFiles(self):
        """
        """
        # if LEFT is in the file, then store a metric called RIGHT
        name_map = {"period_mean":"Period Mean",
                    "bias_of"    :"Bias",
                    "rmse_of"    :"RMSE",
                    "shift_of"   :"Phase Difference",
                    "bias_score" :"Bias Score",
                    "rmse_score" :"RMSE Score",
                    "shift_score":"Seasonal Cycle Score",
                    "iav_score"  :"Interannual Variability Score",
                    "sd_score"   :"Spatial Distribution Score"}
        metrics = {}
        plots   = {}
        
        # Loop over all result files from all models
        for fname in glob.glob("%s/*.nc" % self.output_path):

            # Extract the model name from the filename
            mname     = (fname.split("/")[-1])[:-3]

            # Grab a list of variables which are part of this result file
            f         = Dataset(fname)
            variables = [v for v in f.variables.keys() if v not in f.dimensions.keys()]
            f.close()

            # Loop over all variables. If a scalar is found, add it to
            # the metrics dictionary for placement in the HTML Google
            # table we will build. If temporal/spatial/site data is
            # found, store references to these variables and
            # compute/store plotting limits.
            metrics[mname] = {}
            for vname in variables:
                var = Variable(filename=fname,variable_name=vname)

                if var.data.size == 1:
                    # This is a scalar variable and might be something
                    # we need to put in the metrics dictionary
                    name = "_".join(var.name.split("_")[:2])
                    for region in self.regions:
                        if region not in metrics[mname].keys(): metrics[mname][region] = {}
                        if region in var.name and var.data.size ==1 and name in name_map.keys():
                            metrics[mname][region][name_map[name]] = var
                else:
                    # This is not a scalar and thus we should make a
                    # plot. But we need to track the limits across all
                    # models.
                    if not plots.has_key(var.name):
                        plots[var.name] = {}
                        plots[var.name]["min"] =  1e20
                        plots[var.name]["max"] = -1e20
                        plots[var.name]["colorbar"] = True
                    plots[var.name]["min"] = min(plots[var.name]["min"],var.data.min())
                    plots[var.name]["max"] = max(plots[var.name]["max"],var.data.max())
                    plots[var.name][mname] = var

        # Walk through the metrics dictionary computing the weighted overall scores
        for model in metrics.keys():
            for region in metrics[model].keys():
                overall_score  = 0.
                sum_of_weights = 0.
                scores = [s for s in metrics[model][region].keys() if s in self.weight.keys()]
                for score in scores:
                    overall_score  += self.weight[score]*metrics[model][region][score].data
                    sum_of_weights += self.weight[score]
                overall_score /= max(sum_of_weights,1e-12)
                metrics[model][region]["Overall Score"] = Variable(data=overall_score,name="overall_score",unit="-")
                
        space_opts = {
            "timeint"  : { "cmap":"Greens" , "sym":False },
            "bias"     : { "cmap":"seismic", "sym":True  },
            "phase"    : { "cmap":"jet"    , "sym":False },  
            "shift"    : { "cmap":"PRGn"   , "sym":True  },
        }
        
        # 
        for pname in plots.keys():
            plot   = plots[pname]
            models = plots[pname].keys();
            models.remove("max"); models.remove("min"); models.remove("colorbar")
            for model in models:
                var  = plot[model]
                name = var.name.split("_")[0]
                if (var.spatial or var.ndata is not None) and name in space_opts.keys():
                    vmin = plot["min"]
                    vmax = plot["max"]
                    if space_opts[name]["sym"]:
                        vabs =  max(abs(plot["min"]),abs(plot["max"]))
                        vmin = -vabs
                        vmax =  vabs
                    for region in self.regions:
                        fig = plt.figure(figsize=(6.8,2.8))
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        var.plot(ax,region=region,vmin=vmin,vmax=vmax,cmap=space_opts[name]["cmap"])
                        fig.savefig("%s/%s_%s_%s.png" % (self.output_path,model,region,name))
                        plt.close()
                
        # Write the html page
        f = file("%s/%s.html" % (self.output_path,self.name),"w")
        self.layout.setMetrics(metrics)
        f.write("%s" % self.layout)
        f.close()

        
if __name__ == "__main__":
    import os
    from ModelResult import ModelResult
    m   = ModelResult(os.environ["ILAMB_ROOT"]+"/MODELS/CMIP5/inmcm4",modelname="inmcm4")

    gpp = GenericConfrontation("GPPFluxnetGlobalMTE",
                               os.environ["ILAMB_ROOT"]+"/DATA/gpp/FLUXNET-MTE/derived/gpp.nc",
                               "gpp")
    gpp.confront(m)
    gpp.postProcessFromFiles()
    
    hfls = GenericConfrontation("LEFluxnetSites",os.environ["ILAMB_ROOT"]+"/DATA/le/FLUXNET/derived/le.nc",
                                "hfls",
                                alternate_vars=["le"])
    hfls.confront(m)
    hfls.postProcessFromFiles()
