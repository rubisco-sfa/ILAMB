import ilamblib as il
from Variable import *
from constants import four_code_regions
import os,glob
from netCDF4 import Dataset

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
        metrics = {}
        for fname in glob.glob("%s/*.nc" % self.output_path):
            mname     = fname[:-3]
            f         = Dataset(fname)
            variables = [v for v in f.variables.keys() if v not in f.dimensions.keys()]
            f.close()
            metrics[mname] = {}
            for vname in variables:
                var = Variable(filename=fname,variable_name=vname)
                print var.name
                for region in self.regions:
                    if region in var.name: 
                        metrics[mname][region] = var

        
if __name__ == "__main__":
    import os
    from ModelResult import ModelResult
    m   = ModelResult(os.environ["ILAMB_ROOT"]+"/MODELS/CMIP5/inmcm4",modelname="inmcm4")

    gpp = GenericConfrontation("GPPFluxnetGlobalMTE",os.environ["ILAMB_ROOT"]+"/DATA/gpp/FLUXNET-MTE/derived/gpp.nc",
                               "gpp",
                               regions=["amazon"])
    gpp.confront(m)
    gpp.postProcessFromFiles()
    
    hfls = GenericConfrontation("LEFluxnetSites",os.environ["ILAMB_ROOT"]+"/DATA/le/FLUXNET/derived/le.nc",
                                "hfls",
                                alternate_vars=["le"],
                                regions=["amazon"])
    hfls.confront(m)
    hfls.postProcessFromFiles()
