from .Confrontation import Confrontation
from .Confrontation import getVariableList
import matplotlib.pyplot as plt
from . import Post as post
from scipy.interpolate import CubicSpline
from .Variable import Variable
from netCDF4 import Dataset
from . import ilamblib as il
import numpy as np
import os,glob
from .constants import lbl_months,bnd_months
from cf_units import Unit
import cftime

def _meanDiurnalCycle(var,n):
    begin = np.argmin(var.time[:(n-1)]%n)
    end   = begin+int(var.time[begin:].size/float(n))*n
    vmean = var.data[begin:end].reshape((-1,n))
    vmean = vmean[np.where(vmean.mask.any(axis=1)==False)]
    per   = np.percentile(vmean,[10,90],axis=0)
    per10 = per[0,:]
    per90 = per[1,:]
    np.seterr(under='ignore',over='ignore')
    vmean = vmean.mean(axis=0)
    np.seterr(under='raise',over='raise')
    t     = np.linspace(0,24,n+1)[:-1]
    tmax  = t[vmean.argmax()]
    return t,vmean,per10,per90,tmax

def DiurnalReshape(time,time_bnds,data):
    dt    = (time_bnds[:,1]-time_bnds[:,0])[:-1]
    dt    = dt.mean()
    spd   = int(round(1./dt))
    begin = np.argmin(time[:(spd-1)]%spd)
    end   = begin+int(time[begin:].size/float(spd))*spd
    shp   = (-1,spd) + data.shape[1:]
    cycle = data[begin:end].reshape(shp)
    t     = time[begin:end].reshape(shp).mean(axis=1)
    return cycle,t

def _findSeasonalTiming(t,x):
    """Return the beginning and ending time of the season of x.

    The data x is assumed to start out relatively small, pass through
    a seasonal period of high values, and then return to small values,
    similar to a bell curve. This routine returns the times where
    these seasonal changes occur. To do this, we accumulate the
    variable x and then try to fit 3 linear segments to each seasonal
    portion. We pose a problem then that finds the breaks which
    minimizes the residual of these three best fit lines.

    Parameters
    ----------
    time: numpy.ndarray
        time array
    x: numpy.ndarray
        a single cycle of data to extract the season from

    Returns
    -------
    tbnds: numpy.ndarray
        the beginning and ending time in an array of size 2
    """
    def cost(t,y):
        from scipy.stats import linregress
        out = linregress(t,y)
        return np.sqrt((((out.slope*t+out.intercept)-y)**2).sum())
    y = x.cumsum()
    b = int(y.size/2-1)
    e = int(y.size/2+1)
    I = np.asarray(range(2,b))
    C = np.zeros(I.shape)
    for a,i in enumerate(I): C[a] = cost(t[:i],y[:i]) + cost(t[i:e],y[i:e])
    b = I[C.argmin()]
    I = np.asarray(range(e,y.size-2))
    C = np.zeros(I.shape)
    for a,i in enumerate(I): C[a] = cost(t[b:i],y[b:i]) + cost(t[i:],y[i:])
    e = I[C.argmin()]
    return t[[b,e]]

def _findSeasonalCentroid(t,x):
    """Return the centroid of the season in polar and cartesian coordinates.

    Parameters
    ----------
    time: numpy.ndarray
        time array but scaled [0,2 pi]
    x: numpy.ndarray
        a single cycle of data to extract the season from

    Returns
    -------
    centroid: numpy.ndarray
        array of size 4, [r,theta,x,y]
    """
    x0 = (x*np.cos(t/365.*2*np.pi)).mean()
    y0 = (x*np.sin(t/365.*2*np.pi)).mean()
    r0 = np.sqrt(x0*x0+y0*y0)
    a0 = np.arctan2(y0,x0)
    return r0,a0,x0,y0

class ConfDiurnal(Confrontation):
    """A confrontation for examining the diurnal
    """
    def __init__(self,**keywords):

        # Calls the regular constructor
        super(ConfDiurnal,self).__init__(**keywords)

        # Setup a html layout for generating web views of the results
        pages = []

        # Mean State page
        pages.append(post.HtmlPage("MeanState","Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Seasonal Diurnal Cycle","Diurnal magnitude"])
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation","Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        with Dataset(self.source) as dset:
            for attr in dset.ncattrs():
                a = dset.getncattr(attr)
                if 'astype' in dir(a): a = a.astype('str')
                if 'encode' in dir(a): a = a.encode('ascii','ignore')                
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (attr,a)
        self.layout = post.HtmlLayout(pages,self.longname)

    def stageData(self,m):

        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars,
                       convert_calendar = False)
        if obs.time is None: raise il.NotTemporalVariable()
        self.pruneRegions(obs)

        # Try to extract a commensurate quantity from the model
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars     = self.alternate_vars,
                                  expression   = self.derived,
                                  convert_calendar = False,
                                  lats         = None if obs.spatial else obs.lat,
                                  lons         = None if obs.spatial else obs.lon)
        
        # Handle molar mass, migrate to ILAMB.Variable.convert()
        if (np.any([Unit(u).is_convertible("g")   for u in mod.unit.split()]) and
            np.any([Unit(u).is_convertible("mol") for u in obs.unit.split()])):
            if self.variable in ["gpp","nee","reco"]:
                mod.unit = str(Unit(mod.unit) / Unit("12.0107 g mol-1"))
            
        # When we make things comparable, sites can get pruned, we
        # also need to prune the site labels
        lat = np.copy(obs.lat); lon = np.copy(obs.lon)
        obs,mod = il.MakeComparable(obs,mod,clip_ref=True,prune_sites=True,allow_diff_times=True)
        
        return obs,mod

    def confront(self,m):

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        # Grab the data
        obs,mod = self.stageData(m)
                 
        # Number of data points per day
        nobs = int(np.round(1./np.diff(obs.time).mean()))
        nmod = int(np.round(1./np.diff(mod.time).mean()))
        
        # Analysis on a per year basis
        Tobs = cftime.num2date(obs.time,"days since 1850-1-1")
        Tmod = cftime.num2date(mod.time,"days since 1850-1-1")
        Yobs = np.asarray([t.year for t in Tobs],dtype=int)
        Ymod = np.asarray([t.year for t in Tmod],dtype=int)
        Y    = np.unique(Yobs)
        S1 = []; S2 = []; S3 = []; Lobs = []; Lmod = []
        Sobs  = {}; Smod  = {}
        omask = np.zeros(obs.time.size,dtype=int)
        for y in Y:

            # datum for this year
            datum = cftime.date2num(cftime.datetime(y,1,1),"days since 1850-1-1")
            
            # Reshape the year's worth of data
            iobs = np.where(y==Yobs)[0]
            imod = np.where(y==Ymod)[0]
            if (iobs.size < 0.9*nobs*365): continue
            if (imod.size < 0.9*nmod*365): continue
            vobs,tobs = DiurnalReshape(obs.time     [iobs] - datum,
                                       obs.time_bnds[iobs] - datum,
                                       obs.data     [iobs,0])
            vmod,tmod = DiurnalReshape(mod.time     [imod] - datum,
                                       mod.time_bnds[imod] - datum,
                                       mod.data     [imod,0])

            # Compute the diurnal magnitude
            vobs = vobs.max(axis=1)-vobs.min(axis=1)
            vmod = vmod.max(axis=1)-vmod.min(axis=1)
            Sobs[y] = Variable(name = "season_%d" % y,
                               unit = obs.unit,
                               time = tobs,
                               data = vobs)
            Smod[y] = Variable(name = "season_%d" % y,
                               unit = mod.unit,
                               time = tmod,
                               data = vmod)

            # Compute metrics
            To  = _findSeasonalTiming  (tobs,vobs)
            Ro  = _findSeasonalCentroid(tobs,vobs)
            Tm  = _findSeasonalTiming  (tmod,vmod)
            Rm  = _findSeasonalCentroid(tmod,vmod)
            dTo = To[1]-To[0]       # season length of the observation
            a   = np.log(0.1) / 0.5 # 50% relative error equals a score of 1/10
            s1  = np.exp(a* np.abs(To[0]-Tm[0])/dTo)
            s2  = np.exp(a* np.abs(To[1]-Tm[1])/dTo)
            s3  = np.linalg.norm(np.asarray([Ro[2]-Rm[2],Ro[3]-Rm[3]])) #  |Ro - Rm|
            den = np.linalg.norm(np.asarray([      Ro[2],      Ro[3]])) # /|Ro|
            if den < 1e-12 or np.isnan(den):
                s3  = 0.
            else:
                s3 /= den
                s3  = np.exp(-s3)
            S1.append(s1); S2.append(s2); S3.append(s3)
            Lobs.append(To[1]-To[0])
            Lmod.append(Tm[1]-Tm[0])

            # mask away the off season
            obs.data.mask[:,0] += (y == Yobs)*(((obs.time-datum) < To[0]) + ((obs.time-datum) > To[1]))
            mod.data.mask[:,0] += (y == Ymod)*(((mod.time-datum) < Tm[0]) + ((mod.time-datum) > Tm[1]))

        # Seasonal Mean Diurnal Cycle
        ot,omean,o10,o90,opeak = _meanDiurnalCycle(obs,nobs)
        mt,mmean,m10,m90,mpeak = _meanDiurnalCycle(mod,nmod)

        # Seasonal Mean Daily Uptake
        ouptake = obs.integrateInTime(mean=True)
        muptake = mod.integrateInTime(mean=True)

        # Score by mean values across years
        S1   = np.asarray(S1  ).mean()
        S2   = np.asarray(S2  ).mean()
        S3   = np.asarray(S3  ).mean()
        S4   = abs(opeak-mpeak)/12.
        S4   = 1 - ((S4>1)*(1-S4) + (S4<=1)*S4)
        S5   = np.exp(-np.abs(ouptake.data-muptake.data)/ouptake.data)
        Lobs = np.asarray(Lobs).mean()
        Lmod = np.asarray(Lmod).mean()

        with Dataset(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name)),mode="w") as results:
            results.setncatts({"name" :m.name, "color":m.color})
            Variable(name = "Season Length global",
                     unit = "d",
                     data = Lmod).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Beginning Score global",
                     unit = "1",
                     data = S1).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Ending Score global",
                     unit = "1",
                     data = S2).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Strength Score global",
                     unit = "1",
                     data = S3).toNetCDF4(results,group="MeanState")
            Variable(name = "Diurnal Max Timing Score global",
                     unit = "1",
                     data = S4).toNetCDF4(results,group="MeanState")
            Variable(name = "Daily Uptake Score global",
                     unit = "1",
                     data = S5).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_mean",
                     unit = mod.unit,
                     time = mt,
                     data = mmean).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_lower",
                     unit = mod.unit,
                     time = mt,
                     data = m10).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_upper",
                     unit = mod.unit,
                     time = mt,
                     data = m90).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Mean Daily Uptake",
                     unit = muptake.unit,
                     data = muptake.data).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Time of Maximum",
                     unit = "h",
                     data = mpeak).toNetCDF4(results,group="MeanState")
            for key in Smod.keys(): Smod[key].toNetCDF4(results,group="MeanState")
            results.setncattr("complete",1)
        if not self.master: return
        with Dataset(os.path.join(self.output_path,"%s_Benchmark.nc" % self.name),mode="w") as results:
            results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5])})
            Variable(name = "Season Length global",
                     unit = "d",
                     data = Lobs).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_mean",
                     unit = obs.unit,
                     time = ot,
                     data = omean).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_lower",
                     unit = obs.unit,
                     time = ot,
                     data = o10).toNetCDF4(results,group="MeanState")
            Variable(name = "cycle_upper",
                     unit = obs.unit,
                     time = ot,
                     data = o90).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Mean Daily Uptake",
                     unit = ouptake.unit,
                     data = ouptake.data).toNetCDF4(results,group="MeanState")
            Variable(name = "Season Time of Maximum",
                     unit = "h",
                     data = opeak).toNetCDF4(results,group="MeanState")
            for key in Sobs.keys(): Sobs[key].toNetCDF4(results,group="MeanState")
            results.setncattr("complete",1)

    def determinePlotLimits(self):

        self.limits = {}
        self.limits["season"] = 0.
        for fname in glob.glob(os.path.join(self.output_path,"*.nc")):
            with Dataset(fname) as dataset:
                if "MeanState" not in dataset.groups: continue
                group     = dataset.groups["MeanState"]
                variables = [v for v in group.variables.keys() if v not in group.dimensions.keys()]
                for vname in variables:
                    if "season" in vname:
                        self.limits["season"] = max(self.limits["season"],group.variables[vname].up99)

    def modelPlots(self,m):

        bname  = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname  = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        page.priority = ["Beginning","Ending","Strength","Score","Overall"]

        # list pf plots must be in both the benchmark and the model
        with Dataset(bname) as dset:
            bplts = [key for key in dset.groups["MeanState"].variables.keys() if "season" in key]
        with Dataset(fname) as dset:
            fplts = [key for key in dset.groups["MeanState"].variables.keys() if "season" in key]
        plots = [v for v in bplts if v in fplts]
        plots.sort()

        for plot in plots:

            obs = Variable(filename = bname, variable_name = plot, groupname = "MeanState")
            mod = Variable(filename = fname, variable_name = plot, groupname = "MeanState")

            page.addFigure("Diurnal magnitude",
                           plot,
                           "MNAME_%s.png" % plot,
                           side   = plot.split("_")[-1],
                           legend = False)
            plt.figure(figsize=(5,5),tight_layout=True)
            plt.polar(obs.time/365.*2*np.pi,obs.data,'-k',alpha=0.6,lw=2)
            plt.polar(mod.time/365.*2*np.pi,mod.data,'-',color=m.color)
            plt.xticks(bnd_months[:-1]/365.*2*np.pi,lbl_months)
            plt.ylim(0,self.limits["season"])
            plt.savefig("%s/%s_%s.png" % (self.output_path,m.name,plot))
            plt.close()

        # mean Diurnal Cycle
        obs = Variable(filename = bname, variable_name = "cycle_mean" , groupname = "MeanState")
        olo = Variable(filename = bname, variable_name = "cycle_lower", groupname = "MeanState")
        ohi = Variable(filename = bname, variable_name = "cycle_upper", groupname = "MeanState")
        mod = Variable(filename = fname, variable_name = "cycle_mean" , groupname = "MeanState")
        mlo = Variable(filename = fname, variable_name = "cycle_lower", groupname = "MeanState")
        mhi = Variable(filename = fname, variable_name = "cycle_upper", groupname = "MeanState")
        fig,ax = plt.subplots(figsize=(8,4.5),tight_layout=True)
        dt = np.diff(obs.time).mean()
        ax.plot        (obs.time+0.5*dt,obs.data,color='k',alpha=0.5,lw=2)
        ax.fill_between(obs.time+0.5*dt,olo.data,ohi.data,color='k',alpha=0.09,lw=0)
        dt = np.diff(mod.time).mean()
        ax.plot        (mod.time+0.5*dt,mod.data,color=m.color,lw=2)
        ax.fill_between(mod.time+0.5*dt,mlo.data,mhi.data,color=m.color,alpha=0.15,lw=0)
        xticks      = np.linspace(0,24,9)
        xticklabels = ["%2d:00" % t for t in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.grid(True)
        ax.set_ylabel(post.UnitStringToMatplotlib(obs.unit))
        ax.set_xlabel("local time")
        plt.savefig("%s/%s_cycle.png" % (self.output_path,m.name))
        plt.close()
        page.addFigure("Seasonal Diurnal Cycle",
                       "cycle",
                       "MNAME_cycle.png",
                       side   = "CYCLE",
                       legend = False)

    def compositePlots(self):
        pass
