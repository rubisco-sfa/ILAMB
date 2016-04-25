from Confrontation import Confrontation
from Variable import Variable
from netCDF4 import Dataset
import ilamblib as il
import Post as post
import numpy as np
import os

class ConfTWSA(Confrontation):
    """A confrontation for examining the terrestrial water storage anomaly.

    """            
    def stageData(self,m):
        r"""Extracts model data.

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
        obs,mod = il.MakeComparable(obs,mod,clip_ref=True)
        mod.convert(obs.unit)

        # we only want the anomaly from the model, so subtract the mean
        mod.data -= mod.data.mean(axis=0)
        
        
        return obs,mod

