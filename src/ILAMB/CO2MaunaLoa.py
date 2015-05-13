import numpy as np
import ilamblib as il
from constants import convert
from Variable import Variable
import pylab as plt
import os

class CO2MaunaLoa():
    """
    Confront models with the CO2 concentration data from Mauna Loa.
    """
    def __init__(self):

        # Populate with observational data needed for the confrontation
        self.name    = "CO2MaunaLoa"
        fname        = "%s/%s" % (os.environ["ILAMB_ROOT"],"DATA/co2/MAUNA.LOA/original/co2_1958-2014.txt")
        if not os.path.isfile(fname):
            msg  = "I am looking for data for the %s confrontation here\n\n" % self.name
            msg += "%s\n\nbut I cannot find it. " % fname
            msg += "Did you download the data? Have you set the ILAMB_ROOT envronment variable?"
            raise il.MisplacedData(msg)
        data         = np.fromfile(fname,sep=" ").reshape((-1,7))
        self.t       = (data[:,2]-1850)*365.  # convert to days since 00:00:00 1/1/1850
        self.var     = np.ma.masked_array(data[:,5])
        self.unit    = "1e-6"
        self.lat     = 19.4
        self.lon     = 24.4
        self.nlayers = 3
        self.data    = {}

        # build output path if not already built
        self.output_path = "_build/%s" % self.name
        dirs = self.output_path.split("/")
        for i,d in enumerate(dirs):
            dname = "/".join(dirs[:(i+1)])
            if not os.path.isdir(dname): os.mkdir(dname)

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
        var : ILAMB.Variable.Variable
            the requested variable
        """
        begin = np.argmin(np.abs(self.t-initial_time))
        end   = np.argmin(np.abs(self.t-final_time))+1
        if output_unit is not "":
            try:
                self.var *= convert["co2"][output_unit][self.unit]
                self.unit = output_unit
            except:
                msg  = "The gpp variable is in units of [%s]. " % unit
                msg += "You asked for units of [%s] but I do not know how to convert" % output_unit
                raise il.UnknownUnit(msg)
        return Variable(self.var[begin:end],self.unit,time=self.t[begin:end],name="co2")

    def confront(self,m):
        r"""Confronts the input model with the observational data."""

        # get the observational data
        obs_co2 = self.getData(output_unit="ppm")

        # time limits for this confrontation (with a little padding)
        t0,tf = self.t.min()-7, self.t.max()+7

        # extract the time, variable, and unit of the model result
        mod_co2 = m.extractPointTimeSeries("co2",self.lat,self.lon,
                                           initial_time=t0,
                                           final_time=tf,
                                           alt_vars=["co2mass"],
                                           output_unit="ppm")

        # check that the variables is monthly as expected
        if not np.allclose(mod_co2.dt,30,atol=5):
            raise il.VarNotMonthly("Spacing of co2 data from the %s model is not monthly, dt = %f" % (m.name,mod_co2.dt))
        
        # some CO2 results are given in atmospheric layers, so we need
        # to handle this...incoming ugly code
        if mod_co2.data.ndim == 2:

            # ...however, many of the early layers are masked, so
            # determine which index is the first non-masked
            t = mod_co2.time; v = mod_co2.data
            index = np.apply_along_axis(np.sum,1,v.mask)

            # now we will average the first nlayers non-masked layers
            tmp = np.zeros(t.size)
            for i in range(self.nlayers):
                tmp += v.data[np.ix_(range(t.size)),(index+i).clip(0,v.shape[1])][0,:]
            mod_co2 = Variable(np.ma.masked_array(tmp)/self.nlayers,mod_co2.unit,time=mod_co2.time,name=mod_co2.name)

        cdata = {}
        self.data["co2"] = obs_co2
        cdata["co2"]     = mod_co2
        self.plot(m,cdata)

        return cdata

    def plotFromFiles(self):
        pass

    def plot(self,m=None,data=None):

        # model space integrated mean compared to benchmark
        fig = plt.figure(figsize=(6.8,2.8*0.8))
        ax  = fig.add_axes([0.06,0.025,0.88,0.965])
        self.data["co2"].plot(ax,lw=2,alpha=0.25,color='k',label="Obs")
        data["co2"].plot(ax,color=m.color,label=m.name)
        ax.set_xlabel("Year")
        ax.set_ylabel(data["co2"].unit)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5,1.2))
        fig.savefig("%s/%s_spaceint.png" % (self.output_path,m.name),
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
