from Confrontation import Confrontation
from Variable import Variable
from netCDF4 import Dataset
from copy import deepcopy
import ilamblib as il
import Post as post
import numpy as np
import os

class ConfNBP(Confrontation):
    """A confrontation for examining the global net ecosystem carbon balance.

    """
    def __init__(self,**keywords):
        
        # Ugly, but this is how we call the Confrontation constructor
        super(ConfNBP,self).__init__(**keywords)

        # Now we overwrite some things which are different here
        self.regions        = ['global']
        self.layout.regions = self.regions
        
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
        
        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        """
        # Grab the data
        obs,mod = self.stageData(m)
        obs_sum = obs.accumulateInTime().convert("Pg")
        mod_sum = mod.accumulateInTime().convert("Pg")
        
        # End of period information
        yf = np.round(obs.time_bnds[-1,1]/365.+1850.)
        obs_end = Variable(name = "nbp(%4d)" % yf,
                           unit = obs_sum.unit,
                           data = obs_sum.data[-1])
        mod_end = Variable(name = "nbp(%4d)" % yf,
                           unit = mod_sum.unit,
                           data = mod_sum.data[-1])
        mod_diff = Variable(name = "diff(%4d)" % yf,
                            unit = mod_sum.unit,
                            data = mod_sum.data[-1]-obs_sum.data[-1])

        # Temporal distribution
        np.seterr(over='ignore',under='ignore')
        std0 = obs.data.std()
        std  = mod.data.std()
        np.seterr(over='raise' ,under='raise' )
        R0    = 1.0
        R     = obs.correlation(mod,ctype="temporal")
        std  /= std0
        score = Variable(name = "Temporal Distribution Score global",
                         unit = "1",
                         data = 4.0*(1.0+R.data)/((std+1.0/std)**2 *(1.0+R0)))
        
        # Change names to make things easier to parse later
        obs     .name = "spaceint_of_nbp_over_global"
        mod     .name = "spaceint_of_nbp_over_global"
        obs_sum .name = "accumulate_of_nbp_over_global"
        mod_sum .name = "accumulate_of_nbp_over_global"
        
        # Dump to files
        results = Dataset("%s/%s_%s.nc" % (self.output_path,self.name,m.name),mode="w")
        results.setncatts({"name" :m.name, "color":m.color})
        mod       .toNetCDF4(results,group="MeanState")
        mod_sum   .toNetCDF4(results,group="MeanState")
        mod_end   .toNetCDF4(results,group="MeanState")
        mod_diff  .toNetCDF4(results,group="MeanState")
        score     .toNetCDF4(results,group="MeanState",attributes={"std":std,"R":R.data})
        results.close()
        
        if self.master:
            results = Dataset("%s/%s_Benchmark.nc" % (self.output_path,self.name),mode="w")
            results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
            obs     .toNetCDF4(results,group="MeanState")
            obs_sum .toNetCDF4(results,group="MeanState")
            obs_end .toNetCDF4(results,group="MeanState")
            results.close()
            
        
    def compositePlots(self):

        # we want to run the original and also this additional plot
        super(ConfNBP,self).compositePlots()
        
        for fname in glob.glob("%s/*.nc" % self.output_path):
            dataset = Dataset(fname)
            if "MeanState" not in dataset.groups: continue
            dset    = dataset.groups["MeanState"]
            models.append(dataset.getncattr("name"))
            colors.append(dataset.getncattr("color"))
            key = [v for v in dset.groups["scalars"].variables.keys() if ("Spatial Distribution Score" in v and region in v)]
            if len(key) > 0:
                has_std = True
                sds     = dset.groups["scalars"].variables[key[0]]
                corr[region].append(sds.getncattr("R"  ))
                std [region].append(sds.getncattr("std"))
        
        # temporal distribution Taylor plot
        if has_std:
            page.addFigure("Temporally integrated period mean",
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
