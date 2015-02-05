import numpy as np
import ilamblib as il
from constants import convert

class GPPFluxnetGlobalMTE():
    """
    A class for confronting model results with observational data.
    """
    def __init__(self):
        self.name     = "GPPFluxnetGlobalMTE"

    def getData(self,initial_time=-1e20,final_time=1e20,output_unit=""):
        """Retrieves the confrontation data on the desired time frame and in
        the desired unit.

        Parameters
        ----------
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        output_unit : string, optional
            if specified, will try to convert the units of the variable 
            extract to these units given (see convert in ILAMB.constants)

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ma.core.MaskedArray
            an array of the extracted variable
        unit : string
            a description of the extracted unit
        """
        return 

    def confront(self,m):
        """Confronts the input model with the observational data.

        Parameters
        ----------
        m : ILAMB.ModelResult.ModelResult
            the model results

        Returns
        -------
        cdata : dictionary
            contains all outputs/metrics
        """
        cdata = {}
        return cdata
