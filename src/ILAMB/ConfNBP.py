from .Confrontation import Confrontation
from .Variable import Variable
from netCDF4 import Dataset
from copy import deepcopy
from . import ilamblib as il
import pylab as plt
from . import Post as post
import numpy as np
import os,glob

def SpaceLabels(y,ymin,maxit=1000):    
    for j in range(maxit):
        dy = np.diff(y)
        for i in range(1,y.size-1):
            if dy[i-1] < ymin:
                y[i] += ymin*0.1
                dy = np.diff(y)            
            if dy[i]   < ymin:
                y[i] -= ymin*0.1
                dy = np.diff(y)
        if dy.min() > ymin: return y
    return y

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
                       alternate_vars = self.alternate_vars,
                       t0 = None if len(self.study_limits) != 2 else self.study_limits[0],
                       tf = None if len(self.study_limits) != 2 else self.study_limits[1])
        
        # the model data needs integrated over the globe
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars = self.alternate_vars)
        mod = mod.integrateInSpace().convert(obs.unit)
        mod = mod.coarsenInTime(obs.time_bnds)      
        if not isinstance(mod.data.mask,np.ndarray):
            if mod.data.mask == True:
                mod.data=np.ma.masked_array(data=mod.data,mask=np.ones(mod.data.shape,dtype='bool'))
            else:
                mod.data=np.ma.masked_array(data=mod.data,mask=np.zeros(mod.data.shape,dtype='bool'))
        ind = np.where(mod.data.mask==False)[0]
        mod = mod.trim(t=mod.time_bnds[ind[[0,-1]],[0,1]])
        
        # sign convention is backwards
        obs.data *= -1.
        if obs.data_bnds is not None: obs.data_bnds *= -1.
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
        if self.master:
            obs_sum = obs.accumulateInTime().convert("Pg")
            yf = np.round(obs.time_bnds[-1,1]/365.+1850.)
            obs_end = Variable(name = "nbp(%4d)" % yf,
                               unit = obs_sum.unit,
                               data = obs_sum.data[-1])
            obs    .name = "spaceint_of_nbp_over_global"
            obs_sum.name = "accumulate_of_nbp_over_global"
            with Dataset(os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name)),mode="w") as results:
                results.setncatts({"name" :"Benchmark", "color":np.asarray([0.5,0.5,0.5]),"weight":self.cweight,"complete":0})
                obs     .toNetCDF4(results,group="MeanState")
                obs_sum .toNetCDF4(results,group="MeanState")
                obs_end .toNetCDF4(results,group="MeanState")
                results.setncattr("complete",1)

        # Now that we have written out the obs as they are, let's look at the maximum overlap
        t0 = max(obs.time_bnds[ 0,0],mod.time_bnds[ 0,0])
        tf = min(obs.time_bnds[-1,1],mod.time_bnds[-1,1])
        obs.trim(t=[t0,tf])
        mod.trim(t=[t0,tf])
        obs_sum = obs.accumulateInTime().convert("Pg")
        mod_sum = mod.accumulateInTime().convert("Pg")

        # Score by the trajectory
        traj_score = None
        if obs_sum.data_bnds is not None:
            dV  = obs_sum.data_bnds[:,0]-obs_sum.data
            eps = (np.abs(mod_sum.data-obs_sum.data)-dV).clip(0) # only count error outside uncertainty
            with np.errstate(under='ignore'):
                eps = eps / dV # normalize by the magnitude of the uncertainty
            s = np.exp(-eps)
            traj_score = Variable(name = "Trajectory Score global",
                                  unit = "1",
                                  data = s[1:].mean())
        
        # End of period information        
        yf = np.round(mod.time_bnds[-1,1])/365.+1850.
        obs_end = Variable(name = "nbp(%4d)" % yf,
                           unit = obs_sum.unit,
                           data = obs_sum.data[-1])
        mod_end = Variable(name = "nbp(%4d)" % yf,
                           unit = mod_sum.unit,
                           data = mod_sum.data[-1])
        mod_diff = Variable(name = "diff(%4d)" % yf,
                            unit = mod_sum.unit,
                            data = mod_sum.data[-1]-obs_sum.data[-1])

        # Difference score normlized by the uncertainty in the
        # accumulation at the end of the time period.
        normalizer = 21.6*0.5
        if "GCP"     in self.longname: normalizer = 21.6*0.5
        if "Hoffman" in self.longname: normalizer = 84.6*0.5
        dscore = Variable(name = "Difference Score global" % yf,
                          unit = "1",
                          data = np.exp(-0.287*np.abs(mod_diff.data/normalizer)))

        # Temporal distribution
        skip_taylor = self.keywords.get("skip_taylor",False)
        if not skip_taylor:
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
        mod     .name = "spaceint_of_nbp_over_global"
        mod_sum .name = "accumulate_of_nbp_over_global"

        # Dump to files
        results = Dataset(os.path.join(self.output_path,"%s_%s.nc" % (self.name,m.name)),mode="w")
        results.setncatts({"name" :m.name, "color":m.color,"weight":self.cweight,"complete":0})
        mod       .toNetCDF4(results,group="MeanState")
        mod_sum   .toNetCDF4(results,group="MeanState")
        mod_end   .toNetCDF4(results,group="MeanState")
        mod_diff  .toNetCDF4(results,group="MeanState")
        dscore    .toNetCDF4(results,group="MeanState")
        if traj_score is not None: traj_score.toNetCDF4(results,group="MeanState")
        if not skip_taylor:
            score .toNetCDF4(results,group="MeanState",attributes={"std":std,"R":R.data})
        results.setncattr("complete",1)
        results.close()

    def compositePlots(self):

        # we want to run the original and also this additional plot
        super(ConfNBP,self).compositePlots()

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        colors = {}
        corr   = {}
        std    = {}
        accum  = {}
        for fname in glob.glob(os.path.join(self.output_path,"*.nc")):
            dataset = Dataset(fname)
            if "MeanState" not in dataset.groups: continue
            dset  = dataset.groups["MeanState"]
            mname = dataset.getncattr("name")
            colors[mname] = dataset.getncattr("color")
            key = [v for v in dset.groups["scalars"].variables.keys() if ("Temporal Distribution Score" in v)]
            if len(key) > 0:
                sds = dset.groups["scalars"].variables[key[0]]
                corr[mname] = sds.R
                std [mname] = sds.std
            if "accumulate_of_nbp_over_global" in dset.variables.keys():
                v = Variable(filename      = fname,
                             variable_name = "accumulate_of_nbp_over_global",
                             groupname     = "MeanState")
                accum[mname] = v
                
        # temporal distribution Taylor plot
        if len(corr) > 0:
            page.addFigure("Spatially integrated regional mean",
                           "temporal_variance",
                           "temporal_variance.png",
                           side   = "TEMPORAL TAYLOR DIAGRAM",
                           legend = False)
            fig = plt.figure(figsize=(6.0,6.0))
            keys = corr.keys()
            post.TaylorDiagram(np.asarray([std [key] for key in keys]),
                               np.asarray([corr[key] for key in keys]),
                               1.0,fig,
                               [colors[key] for key in keys])
            fig.savefig(os.path.join(self.output_path,"temporal_variance.png"))
            plt.close()

        # composite accumulation plots
        if len(accum) > 1:

            # play with the limits
            bnd = accum["Benchmark"].data_bnds if accum["Benchmark"].data_bnds is not None else accum["Benchmark"].data
            bmin = bnd.min()
            bmax = bnd.max()
            brange = bmax - bmin
            vmin = min(self.limits["accumulate"]["global"]["min"],bmin)
            vmax = max(self.limits["accumulate"]["global"]["max"],bmax)
            vrange = vmax - vmin
            if bmax/brange < 0.25: bmax = 0.25*brange
            if vmax/vrange < 0.25: vmax = 0.25*vrange
            skip_detail = False
            if abs((vmax-bmax)/vrange) < 0.1 and abs((vmin-bmin)/vrange) < 0.1: skip_detail = True
            
            page.addFigure("Spatially integrated regional mean",
                           "compaccumulation",
                           "RNAME_compaccumulation.png",
                           side   = "ACCUMULATION",
                           legend = False)
            NBPplot(accum,vmin,vmax,colors,
                    os.path.join(self.output_path,"global_compaccumulation.png"))

            if not skip_detail:
                page.addFigure("Spatially integrated regional mean",
                               "compdaccumulation",
                               "RNAME_compdaccumulation.png",
                               side   = "ACCUMULATION DETAIL",
                               legend = False)
                NBPplot(accum,bmin,bmax,colors,
                        os.path.join(self.output_path,"global_compdaccumulation.png"))
            
            
def NBPplot(V,vmin,vmax,colors,fname):

    keys = V.keys()
    Y = []; L = []
    for key in V:
        if key == "Benchmark": continue
        if V[key].time[0] > V["Benchmark"].time[0]+10: continue
        L.append(key)
        Y.append(V[key].data[-1])
    Y = np.asarray(Y); L = np.asarray(L)
    ind = np.argsort(Y)
    Y = Y[ind]; L = L[ind]
            
    fig = plt.figure(figsize=(11.8,5.8))
    ax  = fig.add_subplot(1,1,1,position=[0.06,0.06,0.8,0.92])
    data_range = vmax-vmin
    fig_height = fig.get_figheight()
    font_size  = 10
    dy = 0.05*data_range
    y = SpaceLabels(Y.copy(),data_range/fig_height*font_size/50.)
    v = V["Benchmark"]
    for i in range(L.size):
        key = L[i]
        V[key].plot(ax,lw=2,color=colors[key],label=key,vmin=vmin-dy,vmax=vmax+dy)
        ax.text(v.time[-1]/365+1850+2,y[i],key,color=colors[key],va="center",size=font_size)

    v.plot(ax,lw=2,color='k',label="Benchmark",vmin=vmin-dy,vmax=vmax+dy)
    if v is None: v = V[keys[0]]
    ax.text(0.02,0.95,"Land Source",transform=ax.transAxes,size=20,alpha=0.5,va='top')
    ax.text(0.02,0.05,"Land Sink",transform=ax.transAxes,size=20,alpha=0.5)
    #ax.set_xticks(range(int(v.time[0]/365+1850),int(v.time[-1]/365+1850),25))
    ax.set_xlim(v.time[0]/365+1850,v.time[-1]/365+1850)
    ax.set_ylabel("[Pg]")
    fig.savefig(os.path.join(fname))
    plt.close()
