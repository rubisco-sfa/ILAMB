import ilamblib as il
from Variable import *
from constants import four_code_regions,space_opts,time_opts
import os,glob,re
from netCDF4 import Dataset
import Post as post
import pylab as plt

class GenericConfrontation:
    
    def __init__(self,name,srcdata,variable_name,**keywords):

        # Initialize
        self.master         = False
        self.name           = name
        self.srcdata        = srcdata
        self.variable_name  = variable_name
        self.output_path    = keywords.get("output_path","_build/%s/" % self.name)
        self.alternate_vars = keywords.get("alternate_vars",[])
        self.regions        = keywords.get("regions",four_code_regions)
        self.data           = None
        self.cmap           = keywords.get("cmap","jet")
        self.land           = keywords.get("land",False)
        self.limits         = None
        self.longname       = self.output_path
        self.longname       = self.longname.replace("//","/").replace("./","").replace("_build/","")
        if self.longname[-1] == "/": self.longname = self.longname[:-1]
        
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
        #dirs = self.output_path.split("/")
        #for i,d in enumerate(dirs):
            #dname = "/".join(dirs[:(i+1)])
            #if not os.path.isdir(dname): os.mkdir(dname)

        # Setup a html layout for generating web views of the results
        self.layout = post.HtmlLayout(self,regions=self.regions)
        self.layout.setHeader("CNAME / RNAME / MNAME")
        self.layout.setSections(["Temporally integrated period mean",
                                 "Spatially integrated regional mean"])
        
        # Define relative weights of each score in the overall score
        # (FIX: need some way for the user to modify this)
        self.weight = {"bias_score" :1.,
                       "rmse_score" :2.,
                       "shift_score":1.,
                       "iav_score"  :1.,
                       "sd_score"   :1.}

    def stageData(self,m):
        """
        """
        # Read in the data, and perform consistency checks depending
        # on the data types found
        if self.data is None:
            obs = Variable(filename=self.srcdata,variable_name=self.variable_name,alternate_vars=self.alternate_vars)
            self.data = obs
        else:
            obs = self.data
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
        if self.land and mod.spatial:
            mod.data = np.ma.masked_array(mod.data,
                                          mask=mod.data.mask+(mod.area<1e-2)[np.newaxis,:,:],
                                          copy=False)
        mod.convert(obs.unit)
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

        # Open a dataset for recording the results of this confrontation
        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        AnalysisFluxrate(obs,mod,dataset=results,regions=self.regions)
        
    def determinePlotLimits(self):
        """
        This is essentially the reduction via datafile
        """
        limits = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            dataset   = Dataset(fname)
            variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
            for vname in variables:
                var   = dataset.variables[vname]
                pname = "_".join(vname.split("_")[:2])
                if var[...].size <= 1: continue
                if not limits.has_key(vname):
                    limits[vname] = {}
                    limits[vname]["min"] = +1e20
                    limits[vname]["max"] = -1e20
                limits[vname]["min"] = min(limits[vname]["min"],var.getncattr("min"))
                limits[vname]["max"] = max(limits[vname]["max"],var.getncattr("max"))
            dataset.close()
        self.limits = limits

    def computeOverallScore(self,m):
        """
        """
        fname     = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        dataset   = Dataset(fname,mode="r+")
        variables = [v for v in dataset.variables.keys() if "score" in v and "overall" not in v]
        scores    = []
        for v in variables:
            s = "_".join(v.split("_")[:2])
            if s not in scores: scores.append(s)
        for region in self.regions:
            for v in variables:
                if region not in v: continue
                overall_score  = 0.
                sum_of_weights = 0.    
                for score in scores:
                    overall_score  += self.weight[score]*dataset.variables[v][...]
                    sum_of_weights += self.weight[score]
                overall_score /= max(sum_of_weights,1e-12)
            name = "overall_score_over_%s" % region
            if name in dataset.variables.keys():
                dataset.variables[name][0] = overall_score
            else:
                Variable(data=overall_score,name=name,unit="-").toNetCDF4(dataset)
        dataset.close()
        
    def postProcessFromFiles(self,m):
        """
        """
        fname     = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        dataset   = Dataset(fname)
        variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
        color     = dataset.getncattr("color")
        for vname in variables:
            
            # is this a variable we need to plot?
            pname = vname.split("_")[0]
            if vname not in self.limits.keys(): continue
            var = Variable(filename=fname,variable_name=vname)
            
            if var.spatial and not var.temporal:
            
                # determine plot limits and colormap
                vmin = self.limits[vname]["min"]
                vmax = self.limits[vname]["max"]
                cmap = space_opts[pname]["cmap"]
                if space_opts[pname]["sym"]:
                    vabs =  max(abs(vmin),abs(vmax))
                    vmin = -vabs
                    vmax =  vabs
                if cmap == "choose": cmap = self.cmap
                
                # plot variable
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,region=region,vmin=vmin,vmax=vmax,cmap=cmap)
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

            if not var.spatial and var.temporal:

                # plot variable
                for region in self.regions:
                    fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                    var.plot(ax,lw=2,color=color,label=m.name,
                             ticks     =time_opts[pname]["ticks"],
                             ticklabels=time_opts[pname]["ticklabels"])
                    ylbl = time_opts[pname]["ylabel"]
                    if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                    ax.set_ylabel(ylbl)
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

            
        """
        # Loop over all result files from all models
        for fname in glob.glob("%s/*.nc" % self.output_path):
            f         = Dataset(fname)
            variables = [v for v in f.variables.keys() if v not in f.dimensions.keys()]
            for vname in variables:
                var = Variable(filename=fname,variable_name=vname)
                
            
            # Grab a list of variables which are part of this result file
            f         = Dataset(fname)
            variables = [v for v in f.variables.keys() if v not in f.dimensions.keys()]
            mname         = f.getncattr("name")
            colors[mname] = f.getncattr("color")
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
                        plots[var.name]["legend"]   = False
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
                # save overall score and possibly rewrite it
                
        # Generate plots and html page
        for pname in plots.keys():
            plot   = plots[pname]
            models = plots[pname].keys()
            models.remove("max"); models.remove("min"); models.remove("colorbar"); models.remove("legend")
            for model in models:
                var  = plot[model]
                name = var.name.split("_")[0]

                # spatial plotting
                if (var.spatial or var.ndata is not None) and name in space_opts.keys():
                    vmin = plot["min"]
                    vmax = plot["max"]
                    opts = space_opts[name]
                    cmap = self.cmap
                    if opts["cmap"] != "choose": cmap = opts["cmap"]
                    self.layout.addFigure(opts["section"],name,opts["pattern"],
                                          side=opts["sidelbl"],legend=opts["haslegend"])
                    if opts["sym"]:
                        vabs =  max(abs(plot["min"]),abs(plot["max"]))
                        vmin = -vabs
                        vmax =  vabs
                    if not plot["legend"] and opts["haslegend"]:
                        plot["legend"] = True
                        fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
                        label = opts["label"]
                        if label == "unit": label =_UnitStringToMatplotlib(var.unit)
                        post.ColorBar(ax,vmin=vmin,vmax=vmax,cmap=cmap,ticks=opts["ticks"],
                                      ticklabels=opts["ticklabels"],label=label)
                        fig.savefig("%s/legend_%s.png" % (self.output_path,name))
                        plt.close()
                    for region in self.regions:
                        fig = plt.figure(figsize=(6.8,2.8))
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        var.plot(ax,region=region,vmin=vmin,vmax=vmax,cmap=cmap)
                        fig.savefig("%s/%s_%s_%s.png" % (self.output_path,model,region,name))
                        plt.close()

                # temporal plotting
                if var.temporal and name in time_opts.keys():
                    opts = time_opts[name]
                    self.layout.addFigure(opts["section"],name,opts["pattern"],
                                          side=opts["sidelbl"],legend=opts["haslegend"])
                    for region in self.regions:
                        fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                        var.plot(ax,lw=2,color=colors[model],label=model,ticks=opts["ticks"],
                                 ticklabels=opts["ticklabels"])
                        ylbl = opts["ylabel"]
                        if ylbl == "unit": ylbl = _UnitStringToMatplotlib(var.unit)
                        ax.set_ylabel(ylbl)
                        fig.savefig("%s/%s_%s_%s.png" % (self.output_path,model,region,name))
                        plt.close()
                    
                  
        # Write the html page
        f = file("%s/%s.html" % (self.output_path,self.name),"w")
        self.layout.setMetrics(metrics)
        f.write("%s" % self.layout)
        f.close()
        """
        
if __name__ == "__main__":
    import os
    from ModelResult import ModelResult
    m   = ModelResult(os.environ["ILAMB_ROOT"]+"/MODELS/CMIP5/inmcm4",modelname="inmcm4")

    gpp = GenericConfrontation("GPPFluxnetGlobalMTE",
                               os.environ["ILAMB_ROOT"]+"/DATA/gpp/FLUXNET-MTE/derived/gpp.nc",
                               "gpp",
                               regions = ['global.large','amazon'])
    gpp.confront(m)
    gpp.postProcessFromFiles()
    
    hfls = GenericConfrontation("LEFluxnetSites",os.environ["ILAMB_ROOT"]+"/DATA/le/FLUXNET/derived/le.nc",
                                "hfls",
                                alternate_vars = ["le"],
                                regions = ['global.large','amazon'])
    hfls.confront(m)
    hfls.postProcessFromFiles()
