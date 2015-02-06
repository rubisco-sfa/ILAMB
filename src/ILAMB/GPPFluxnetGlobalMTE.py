from netCDF4 import Dataset
import numpy as np
import ilamblib as il
from constants import convert

class GPPFluxnetGlobalMTE():
    """
    A class for confronting model results with observational data.
    """
    def __init__(self):
        self.name = "GPPFluxnetGlobalMTE"
        self.path = "/home/ncf/data/ILAMB/DATA/FLUXNET-MTE/derived/"
        self.nlat = 360
        self.nlon = 720

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
        # why are these stored as separate netCDF files? Isn't I/O
        # latency worse if these are broken up and I have to build a
        # composite?
        y0   = max(int(initial_time/365.),1982)
        yf   = min(int(  final_time/365.),2005)
        ny   = yf-y0+1; nm = 12*ny
        t    = np.zeros(nm)
        var  = np.ma.zeros((nm,self.nlat,self.nlon))
        unit = ""
        for y in range(ny):
            yr = y+1982
            for m in range(12):
                ind   = 12*y+m
                fname = "%s%d/gpp_0.5x0.5_%d%02d.nc" % (self.path,yr,yr,m+1)
                f = Dataset(fname)
                v = f.variables["gpp"]
                t  [ind    ] = v.time
                var[ind,...] = v[...]
                unit = v.units
        return t,var,unit

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
