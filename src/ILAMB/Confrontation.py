import ilamblib as il
from Variable import *
from constants import four_code_regions,space_opts,time_opts,mid_months,bnd_months
import os,glob,re
from netCDF4 import Dataset
import Post as post
import pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Confrontation:

    def __init__(self,name,srcdata,variable_name,**keywords):

        # Initialize
        self.master         = True
        self.name           = name
        self.srcdata        = srcdata
        self.variable_name  = variable_name
        self.output_path    = keywords.get("output_path","_build/%s/" % self.name)
        self.alternate_vars = keywords.get("alternate_vars",[])
        self.derived        = keywords.get("derived",None)
        self.regions        = keywords.get("regions",four_code_regions)
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
        self.correlation    = keywords.get("correlation",None)
        
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
        """
        """
        # Read in the data, and perform consistency checks depending
        # on the data types found
        if self.data is None:
            obs = Variable(filename=self.srcdata,variable_name=self.variable_name,alternate_vars=self.alternate_vars)
            self.data = obs
        else:
            obs = self.data
        if obs.time is None: raise il.NotTemporalVariable()
        t0 = obs.time.min()
        tf = obs.time.max()

        if obs.spatial:
            try:
                mod = m.extractTimeSeries(self.variable_name,
                                          alt_vars     = self.alternate_vars,
                                          initial_time = t0,
                                          final_time   = tf)
            except:
                mod = m.derivedVariable(self.variable_name,self.derived,
                                        initial_time = t0,
                                        final_time   = tf)
        else:
            try:
                mod = m.extractTimeSeries(self.variable_name,
                                          alt_vars     = self.alternate_vars,
                                          lats         = obs.lat,
                                          lons         = obs.lon,
                                          initial_time = t0,
                                          final_time   = tf)
            except:
                mod = m.derivedVariable(self.variable_name,self.derived,
                                        lats         = obs.lat,
                                        lons         = obs.lon,
                                        initial_time = t0,
                                        final_time   = tf)
        
        if obs.time.shape != mod.time.shape:
            t0 = max(obs.time.min(),mod.time.min())
            tf = min(obs.time.max(),mod.time.max())
            for var in [obs,mod]:
                begin = np.argmin(np.abs(var.time-t0))
                end   = np.argmin(np.abs(var.time-tf))+1
                var.time = var.time[begin:end]
                var.data = var.data[begin:end,...]

        if obs.time.shape != mod.time.shape: raise il.VarNotOnTimeScale()
        if not np.allclose(obs.time,mod.time,atol=20): raise il.VarsNotComparable()
        if self.land and mod.spatial:
            mod.data = np.ma.masked_array(mod.data,
                                          mask=mod.data.mask+(mod.area<1e-2)[np.newaxis,:,:],
                                          copy=False)
        mod.convert(obs.unit)
        return obs,mod

    def confront(self,m,clist=None):
        r"""Confronts the input model with the observational data.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results
        """
        # Grab the data
        obs,mod = self.stageData(m)

        # Open a dataset for recording the results of this
        # confrontation, record for the benchmark if we are the master
        # process.
        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        benchmark_results = None
        fname = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        if self.master:
            benchmark_results = Dataset(fname,mode="w")
            benchmark_results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})

        # Perform the standard fluxrate analysis
        try:
            AnalysisFluxrate(obs,mod,dataset=results,regions=self.regions,benchmark_dataset=benchmark_results,
                             table_unit=self.table_unit,plot_unit=self.plot_unit,space_mean=self.space_mean)
        except:
            results.close()
            os.system("rm -f %s/%s_%s.nc" % (self.output_path,self.name,m.name))
            raise il.AnalysisError()
        
        # Perform relationship analysis
        obs_dep,mod_dep = obs,mod
        dep_name        = self.longname.split("/")[0]
        dep_plot_unit   = self.plot_unit
        if (dep_plot_unit is None): dep_plot_unit = obs_dep.unit
        
        if clist is not None:
            for c in clist:
                obs_ind,mod_ind = c.stageData(m) # independent variable
                ind_name = c.longname.split("/")[0]            
                ind_plot_unit = c.plot_unit
                if (ind_plot_unit is None): ind_plot_unit = obs_ind.unit
                if self.master:
                    AnalysisRelationship(obs_dep,obs_ind,benchmark_results,ind_name,
                                         dep_plot_unit=dep_plot_unit,ind_plot_unit=ind_plot_unit)
                AnalysisRelationship(mod_dep,mod_ind,results,ind_name,
                                     dep_plot_unit=dep_plot_unit,ind_plot_unit=ind_plot_unit)
                
        # close files
        results.close()
        if self.master: benchmark_results.close()
                
    def determinePlotLimits(self):
        """
        """
        # Determine the min/max of variables over all models
        limits = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            try:
                dataset = Dataset(fname)
            except:
                continue
            variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
            for vname in variables:
                var   = dataset.variables[vname]
                pname = vname.split("_")[0]
                if var[...].size <= 1: continue
                if not space_opts.has_key(pname): continue
                if not limits.has_key(pname):
                    limits[pname] = {}
                    limits[pname]["min"]  = +1e20
                    limits[pname]["max"]  = -1e20
                    limits[pname]["unit"] = post.UnitStringToMatplotlib(var.getncattr("units"))
                limits[pname]["min"] = min(limits[pname]["min"],var.getncattr("min"))
                limits[pname]["max"] = max(limits[pname]["max"],var.getncattr("max"))
            dataset.close()
        
        # Second pass to plot legends
        for pname in limits.keys():
            opts = space_opts[pname]

            # Determine plot limits and colormap
            if opts["sym"]:
                vabs =  max(abs(limits[pname]["min"]),abs(limits[pname]["min"]))
                limits[pname]["min"] = -vabs
                limits[pname]["max"] =  vabs
            limits[pname]["cmap"] = opts["cmap"]
            if limits[pname]["cmap"] == "choose": limits[pname]["cmap"] = self.cmap

            # Plot a legend for each key
            if opts["haslegend"]:
                fig,ax = plt.subplots(figsize=(6.8,1.0),tight_layout=True)
                label  = opts["label"]
                if label == "unit": label = limits[pname]["unit"]
                post.ColorBar(ax,
                              vmin = limits[pname]["min"],
                              vmax = limits[pname]["max"],
                              cmap = limits[pname]["cmap"],
                              ticks = opts["ticks"],
                              ticklabels = opts["ticklabels"],
                              label = label)
                fig.savefig("%s/legend_%s.png" % (self.output_path,pname))
                plt.close()

        self.limits = limits

    def computeOverallScore(self,m):
        """
        Done outside analysis such that weights can be changed and analysis need not be rerun
        """
        fname = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        try:
            dataset = Dataset(fname,mode="r+")
        except:
            return
        variables = [v for v in dataset.variables.keys() if "score" in v and "overall" not in v]
        scores    = []
        for v in variables:
            s = "_".join(v.split("_")[:2])
            if s not in scores: scores.append(s)
        overall_score = 0.
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

    def compositePlots(self):
        """
        """
        if not self.master: return
        models = []
        colors = []
        corr   = {}
        std    = {}
        cycle  = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            dataset = Dataset(fname)
            models.append(dataset.getncattr("name"))
            colors.append(dataset.getncattr("color"))
            for region in self.regions:
                if not std.  has_key(region): std  [region] = []
                if not corr. has_key(region): corr [region] = []
                if not cycle.has_key(region): cycle[region] = []
                key = [v for v in dataset.variables.keys() if ("corr_" in v and region in v)]
                if len(key)>0: corr [region].append(Variable(filename=fname,variable_name=key[0]).data.data)
                key = [v for v in dataset.variables.keys() if ("std_"  in v and region in v)]
                if len(key)>0: std  [region].append(Variable(filename=fname,variable_name=key[0]).data.data)
                key = [v for v in dataset.variables.keys() if ("cycle_"  in v and region in v)]
                if len(key)>0: cycle[region].append(Variable(filename=fname,variable_name=key[0]))
                
        # composite annual cycle plot
        self.layout.addFigure("Spatially integrated regional mean",
                              "compcycle",
                              "RNAME_compcycle.png",
                              side   = "CYCLES",
                              legend = True)
        for region in self.regions:
            if not cycle.has_key(region): continue
            fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
            for name,color,var in zip(models,colors,cycle[region]):
                var.plot(ax,lw=2,color=color,label=name,
                         ticks      = time_opts["cycle"]["ticks"],
                         ticklabels = time_opts["cycle"]["ticklabels"])
                ylbl = time_opts["cycle"]["ylabel"]
                if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                ax.set_ylabel(ylbl)
            fig.savefig("%s/%s_compcycle.png" % (self.output_path,region))
            plt.close()

        # plot legends with model colors (sorted with Benchmark data on top)
        def _alphabeticalBenchmarkFirst(key):
            key = key[0].upper()
            if key == "BENCHMARK": return 0
            return key
        tmp = sorted(zip(models,colors),key=_alphabeticalBenchmarkFirst)
        fig,ax = plt.subplots()
        for model,color in tmp:
            ax.plot(0,0,'o',mew=0,ms=8,color=color,label=model)
        handles,labels = ax.get_legend_handles_labels()
        plt.close()
        fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
        ax.legend(handles,labels,loc="upper left",ncol=3,fontsize=10,numpoints=1)
        ax.axis('off')
        fig.savefig("%s/legend_compcycle.png" % self.output_path)
        fig.savefig("%s/legend_spatial_variance.png" % self.output_path)
        plt.close()
        
        # spatial distribution Taylor plot
        self.layout.addFigure("Temporally integrated period mean",
                              "spatial_variance",
                              "RNAME_spatial_variance.png",
                              side   = "SPATIAL DISTRIBUTION",
                              legend = True)       
        if "Benchmark" in models: colors.pop(models.index("Benchmark"))
        for region in self.regions:
            if not (std.has_key(region) and corr.has_key(region)): continue
            if len(std[region]) != len(corr[region]): continue
            if len(std[region]) == 0: continue
            fig = plt.figure(figsize=(6.0,6.0))
            post.TaylorDiagram(np.asarray(std[region]),np.asarray(corr[region]),1.0,fig,colors)
            fig.savefig("%s/%s_spatial_variance.png" % (self.output_path,region))
            plt.close()

        
    def postProcessFromFiles(self,m):
        """
        Call determinePlotLimits first
        Html layout gets built in here
        """
        bname     = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname     = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        try:
            dataset   = Dataset(fname)
        except:
            return
        variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
        color     = dataset.getncattr("color")
        for vname in variables:

            # is this a variable we need to plot?
            pname = vname.split("_")[0]
            if dataset.variables[vname][...].size <= 1: continue
            var = Variable(filename=fname,variable_name=vname)
            
            if (var.spatial or (var.ndata is not None)) and not var.temporal:

                # grab plotting options
                if pname not in self.limits.keys(): continue
                opts = space_opts[pname]

                # add to html layout
                self.layout.addFigure(opts["section"],
                                      pname,
                                      opts["pattern"],
                                      side   = opts["sidelbl"],
                                      legend = opts["haslegend"])

                # plot variable
                for region in self.regions:
                    fig = plt.figure(figsize=(6.8,2.8))
                    ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                    var.plot(ax,
                             region = region,
                             vmin   = self.limits[pname]["min"],
                             vmax   = self.limits[pname]["max"],
                             cmap   = self.limits[pname]["cmap"])
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

                # Jumping through hoops to get the benchmark plotted and in the html output
                if self.master and (pname == "timeint" or pname == "phase"):

                    opts = space_opts[pname]

                    # add to html layout
                    self.layout.addFigure(opts["section"],
                                          "benchmark_%s" % pname,
                                          opts["pattern"].replace("MNAME","Benchmark"),
                                          side   = opts["sidelbl"].replace("MODEL","BENCHMARK"),
                                          legend = False)

                    # plot variable
                    obs = Variable(filename=bname,variable_name=vname)
                    for region in self.regions:
                        fig = plt.figure(figsize=(6.8,2.8))
                        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
                        obs.plot(ax,
                                 region = region,
                                 vmin   = self.limits[pname]["min"],
                                 vmax   = self.limits[pname]["max"],
                                 cmap   = self.limits[pname]["cmap"])
                        fig.savefig("%s/Benchmark_%s_%s.png" % (self.output_path,region,pname))
                        plt.close()
                    
            if not (var.spatial or (var.ndata is not None)) and var.temporal:
                
                # grab the benchmark dataset to plot along with
                obs = Variable(filename=bname,variable_name=vname)

                # grab plotting options
                opts = time_opts[pname]

                # add to html layout
                self.layout.addFigure(opts["section"],
                                      pname,
                                      opts["pattern"],
                                      side   = opts["sidelbl"],
                                      legend = opts["haslegend"])

                # plot variable
                for region in self.regions:
                    if region not in vname: continue
                    fig,ax = plt.subplots(figsize=(6.8,2.8),tight_layout=True)
                    obs.plot(ax,lw=2,color='k',alpha=0.5)
                    var.plot(ax,lw=2,color=color,label=m.name,
                             ticks     =opts["ticks"],
                             ticklabels=opts["ticklabels"])
                    ylbl = opts["ylabel"]
                    if ylbl == "unit": ylbl = post.UnitStringToMatplotlib(var.unit)
                    ax.set_ylabel(ylbl)
                    fig.savefig("%s/%s_%s_%s.png" % (self.output_path,m.name,region,pname))
                    plt.close()

        # each group is a variable-to-variable relationship object
        groups = [g for g in dataset.groups.keys()]

        dep_name = self.longname.split("/")[0] + "/" + m.name
        for g in groups:
            ind_name  = g.replace("relationship_","") + "/" + m.name
            grp       = dataset.groups[g]
            ind       = grp.variables["ind"][...]
            dep       = grp.variables["dep"][...]
            ind_bnd   = grp.variables["ind_bnd"][...]
            dep_bnd   = grp.variables["dep_bnd"][...]
            histogram = grp.variables["histogram"][...].T
            ind_edges = np.zeros(ind_bnd.shape[0]+1); ind_edges[:-1] = ind_bnd[:,0]; ind_edges[-1] = ind_bnd[-1,1]
            dep_edges = np.zeros(dep_bnd.shape[0]+1); dep_edges[:-1] = dep_bnd[:,0]; dep_edges[-1] = dep_bnd[-1,1]
            fig,ax    = plt.subplots(figsize=(6,5.25),tight_layout=True)
            pc        = ax.pcolormesh(ind_edges,dep_edges,histogram,
                                      norm=LogNorm(),
                                      cmap='plasma')
            x,y = grp.variables["ind_mean"],grp.variables["dep_mean"]
            ax.plot(x,y,'-w',lw=3,alpha=0.75)
            #ax.fill_between(grp.variables["ind_mean"][...],
            #                grp.variables["dep_mean"][...]-grp.variables["dep_std"][...],
            #                grp.variables["dep_mean"][...]+grp.variables["dep_std"][...],
            #                color='k',alpha=0.25,lw=0)
            
            div       = make_axes_locatable(ax)
            fig.colorbar(pc,cax=div.append_axes("right",size="5%",pad=0.05),
                         orientation="vertical",
                         label="Fraction of total datasites")
            ax.set_xlabel("%s,  %s" % (ind_name,post.UnitStringToMatplotlib(x.getncattr("unit"))))
            ax.set_ylabel("%s,  %s" % (dep_name,post.UnitStringToMatplotlib(y.getncattr("unit"))))
            fig.savefig("%s/%s_%s.png" % (self.output_path,g,m.name))
            self.layout.addFigure("Period Mean Relationships",
                                  g,
                                  "%s_%s.png" % (g,m.name),
                                  side   = g.replace("relationship_",""),
                                  legend = False)       
            plt.close()
                    
    def generateHtml(self):
        """
        """
        # only the master processor needs to do this
        if not self.master: return

        # build the metric dictionary
        metrics      = {}
        metric_names = { "period_mean"   : "Period Mean",
                         "bias_of"       : "Bias",
                         "rmse_of"       : "RMSE",
                         "shift_of"      : "Phase Shift",
                         "bias_score"    : "Bias Score",
                         "rmse_score"    : "RMSE Score",
                         "shift_score"   : "Phase Score",
                         "iav_score"     : "Interannual Variability Score",
                         "sd_score"      : "Spatial Distribution Score",
                         "overall_score" : "Overall Score" }
        for fname in glob.glob("%s/*.nc" % self.output_path):
            try:
                dataset   = Dataset(fname)
            except:
                continue
            variables = [v for v in dataset.variables.keys() if v not in dataset.dimensions.keys()]
            mname     = dataset.getncattr("name")
            metrics[mname] = {}
            for vname in variables:
                if dataset.variables[vname][...].size > 1: continue
                var  = Variable(filename=fname,variable_name=vname)
                name = "_".join(var.name.split("_")[:2])
                if not metric_names.has_key(name): continue
                metname = metric_names[name]
                for region in self.regions:
                    if region not in metrics[mname].keys(): metrics[mname][region] = {}
                    if region in var.name: metrics[mname][region][metname] = var
                    
        # write the HTML page
        f = file("%s/%s.html" % (self.output_path,self.name),"w")
        self.layout.setMetrics(metrics)
        f.write(str(self.layout))
        f.close()
