from .Regions import Regions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .Post import UnitStringToMatplotlib
    
class Relationship(object):
    
    def __init__(self,ind,dep,ind_log=False,dep_log=False,order=None,color=None):
        """A class for investigating the relationship, dep = f(ind).

        Parameters
        ----------
        ind : ILAMB.Variable
            a ILAMB Variable which represents the independent variable
        dep : ILAMB.Variable
            a ILAMB Variable which represents the dependent variable
        ind_log : boolean
            enable to indicate that the independent variable is to be
            studied on a logarithmic scale
        dep_log : boolean
            enable to indicate that the dependent variable is to be
            studied on a logarithmic scale
        order : int, optional
            the polynomial order of the model to be used to
            approximate the relationship
        color : ILAMB.Variable, optional
            a ILAMB.Variable to use to color the point cloud
        """
        self.ind = ind
        self.dep = dep
        self.ind_log = ind_log
        self.dep_log = dep_log
        self.color = color
        self.checkConsistency()
        self.limits = self.computeLimits()
        self.dist = {}
        self.order = None
        if order is not None:
            order = int(order)
            self.order = order
                        
    def __str__(self):
        s  = "Relationship:\n"
        s += "-------------\n"
        s += "{0:>20}: ".format("independent") + self.ind.name + "%s\n" % (" (log)" if self.ind_log else "")
        s += "{0:>20}: ".format("ind limits")  + "(%+1.3e,%+1.3e) [%s]\n" % (self.limits[1][0],
                                                                             self.limits[1][1],
                                                                             self.ind.unit)
        s += "{0:>20}: ".format("dependent")  + self.dep.name + "%s\n" % (" (log)" if self.dep_log else "")
        s += "{0:>20}: ".format("dep limits") + "(%+1.3e,%+1.3e) [%s]\n" % (self.limits[0][0],
                                                                            self.limits[0][1],
                                                                            self.dep.unit)
        if self.order is not None:
            s += "{0:>20}: ".format("polynomial") + "%d\n" % self.order
        return s

    def checkConsistency(self):
        assert np.allclose(self.dep.data.shape,self.ind.data.shape)
        if self.color: assert np.allclose(self.dep.data.shape,self.color.data.shape)
        if self.ind_log: assert self.ind.data.min() > 0
        if self.dep_log: assert self.dep.data.min() > 0
        
    def makeComparable(self,b,region=None):
        """Ensures that relationships a and b are consistent on given region for scoring.

        Parameters
        ----------
        b : Relationship
            the relationships to consider
        region : string, optional
            the optional region on which to compare
        """
        assert (type(b) == Relationship)
        assert (self.ind_log == b.ind_log)*(self.dep_log == b.dep_log)
        key = "default" if region is None else region
        recompute_a = False
        recompute_b = False
        
        # Are the limits different? 
        if not np.allclose(self.limits,b.limits):
            recompute_a = True
            recompute_b = True
            limits = self.computeLimits(dep_lim=b.limits[0],
                                        ind_lim=b.limits[1])
            self.limits = limits
            b.limits = limits

        # Have the distributions been tabulated?
        if key not in self.dist.keys(): recompute_a = True
        if key not in b.dist.keys(): recompute_b = True
        
        # Are the distributions binned the same way?
        if (key in self.dist.keys()) and (key in b.dist.keys()):
            if not np.allclose(self.dist[key][0].shape,b.dist[key][0].shape):
                recompute_a = True
                recompute_b = True
            if not np.allclose(self.dist[key][1],b.dist[key][1]):
                recompute_a = True
                recompute_b = True
            if not np.allclose(self.dist[key][2],b.dist[key][2]):
                recompute_a = True
                recompute_b = True

        # Recompute if needed
        if recompute_a: self.buildResponse(region=region)
        if recompute_b: b.buildResponse(region=region)

    def computeLimits(self,dep_lim=None,ind_lim=None):
        """Computes the limits of the dependent and independent. 

        Parameters
        ----------
        dep_lim : array-like of size 2, optional
            if specified, will return the most extensive limits of the
            dependent variable
        ind_lim : array-like of size 2, optional
            if specified, will return the most extensive limits of the
            independent variable

        """
        def _singlelimit(var,limit=None):
            lim     = [var.data.min(),var.data.max()]
            delta   = 1e-8*(lim[1]-lim[0])
            lim[0] -= delta
            lim[1] += delta
            if limit is None:
                limit = lim
            else:
                limit[0] = min(limit[0],lim[0])
                limit[1] = max(limit[1],lim[1])
            return limit
        return _singlelimit(self.dep,dep_lim),_singlelimit(self.ind,ind_lim)

    def buildResponse(self,region=None,nbin=25,eps=3e-3):
        """Creates a 2D distribution and a functional response.

        Also stores the values internally for later use.

        Parameters
        ----------
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region
        nbin : int, optional
            the number of bins to use in both dimensions
        eps : float, optional
            the fraction of points required for a bin in the
            independent variable be included in the funcitonal responses

        Returns
        -------
        dist : numpy.ndarray, shape = (nbin,nbin)
            the 2D distribution representing the relationship
        xedges : numpy.ndarray, shape = (nbin+1)
            the bin breaks of the independent variable
        yedges : numpy.ndarray, shape = (nbin+1)
            the bin breaks of the dependent variable
        mean : numpy.ndarray, max shape = (nbin)
            the average values of the relationship dep = f(ind)
        std : numpy.ndarray, max shape = (nbin)
            the standard deviation of the values of the relationship
            dep = f(ind)
        p : numpy.ndarray, shape = (order+1)
            the polynomial coefficient array, last entry is the constant
        """
        dep = self.dep
        ind = self.ind
        dep_lim = self.limits[0]
        ind_lim = self.limits[1]
        
        # Mask data
        mask = ind.data.mask + dep.data.mask
        if region is not None: mask += Regions().getMask(region,ind)
        x = ind.data[mask==False].flatten()
        y = dep.data[mask==False].flatten()
        xedges = nbin
        yedges = nbin
        if self.ind_log:
            xedges = 10**np.linspace(np.log10(ind_lim[ 0]),
                                     np.log10(ind_lim[-1]),nbin+1)
        if self.dep_log:
            yedges = 10**np.linspace(np.log10(dep_lim[ 0]),
                                     np.log10(dep_lim[-1]),nbin+1)
        
        # Compute normalized 2D distribution
        dist,xedges,yedges = np.histogram2d(x,y,
                                            bins  = [xedges,yedges],
                                            range = [ind_lim,dep_lim])
        dist  = np.ma.masked_values(dist.T,0).astype(float)
        dist /= dist.sum()

        # Compute the functional response
        which_bin = np.digitize(x,xedges).clip(1,xedges.size-1)-1
        mean = np.ma.zeros(xedges.size-1)
        std  = np.ma.zeros(xedges.size-1)
        cnt  = np.ma.zeros(xedges.size-1)
        with np.errstate(under='ignore'):
            for i in range(mean.size):
                yi = y[which_bin==i]
                cnt [i] = yi.size
                if self.dep_log:
                    yi = np.log10(yi)
                    mean[i] = 10**yi.mean()
                    std [i] = 10**yi.std()
                else:
                    mean[i] = yi.mean()
                    std [i] = yi.std()
            mean = np.ma.masked_array(mean,mask = (cnt/cnt.sum()) < eps)
            std  = np.ma.masked_array( std,mask = (cnt/cnt.sum()) < eps)

        # If there is a model order given, compute the regression and
        # the 50% prediction interval
        p = None
        i = None
        if self.order is not None:
            gauss_critval =  0.674 # for 50%, could make more abstract
            if self.dep_log:
                p = np.polyfit(x,np.log10(y),self.order)
                i = gauss_critval*(np.log10(y)-np.polyval(p,x)).std()
            else:
                p = np.polyfit(x,y,self.order)
                i = gauss_critval*(y-np.polyval(p,x)).std()
        
        # Save the arrays
        self.dist["default" if region is None else region] = dist,xedges,yedges,mean,std,p,i
        return dist,xedges,yedges,mean,std,p,i

    def plotPointCloud(self,ax,region=None,ms=1,color=None,vmin=None,vmax=None,cmap=None):
        """Plot the 2D point cloud.

        Parameters
        ----------
        ax : matplotlib axis
            the axis on which to plot the function
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region
        ms : float
            the size of the points to be plotted
        color : str or rbg-tuple
            the color of the points to be plotted
        vmin : float
            if the relationship was initialized with a variable to
            color by, the minimum value of that variable that will be
            plotted
        vmax : float
            if the relationship was initialized with a variable to
            color by, the maximum value of that variable that will be
            plotted
        cmap : str
            if the relationship was initialized with a variable to
            color by, the colormap that will be used in plotting            
        """
        mask = self.ind.data.mask + self.dep.data.mask
        if self.color is not None: mask += self.color.data.mask
        if region is not None: mask += Regions().getMask(region,self.ind)
        x = self.ind.data[mask==False].flatten()
        y = self.dep.data[mask==False].flatten()
        need_colorbar = False
        if self.color is not None and color is None:
            color = self.color.data[mask==False].flatten()
            need_colorbar = True
        sc = ax.scatter(x,y,c=color,s=ms,vmin=vmin,vmax=vmax,cmap=cmap)
        if need_colorbar:
            fig = ax.get_figure()
            fig.colorbar(sc,orientation='horizontal',pad=0.15,
                         label='%s [%s]'% (self.color.name,UnitStringToMatplotlib(self.color.unit)))
        xlabel = self.ind.name + " [%s]" % (UnitStringToMatplotlib(self.ind.unit))
        ylabel = self.dep.name + " [%s]" % (UnitStringToMatplotlib(self.dep.unit))
        ax.set_xlabel(xlabel,fontsize = 12)
        ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
        ax.set_xlim(self.limits[1][0],self.limits[1][1])
        ax.set_ylim(self.limits[0][0],self.limits[0][1])
        if self.dep_log: ax.set_yscale('log')
        if self.ind_log: ax.set_xscale('log')
        
    def plotDistribution(self,ax,region=None):
        """Plot the 2D histogram.

        Parameters
        ----------
        ax : matplotlib axis
            the axis on which to plot the function
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region
        """
        key = "default" if region is None else region
        if key not in self.dist.keys(): self.buildResponse(region=region)
        dist   = self.dist[key][0]
        xedges = self.dist[key][1]
        yedges = self.dist[key][2]
        xlabel = self.ind.name + " [%s]" % (UnitStringToMatplotlib(self.ind.unit))
        ylabel = self.dep.name + " [%s]" % (UnitStringToMatplotlib(self.dep.unit))
        fig = ax.get_figure()
        
        pc = ax.pcolormesh(xedges, yedges, dist,
                           norm = LogNorm(),
                           cmap = 'plasma' if 'plasma' in plt.cm.cmap_d else 'summer',
                           vmin = 1e-4, vmax = 1e-1)
        div = make_axes_locatable(ax)
        fig.colorbar(pc,cax=div.append_axes("right",size="5%",pad=0.05),
                     orientation="vertical",label="Fraction of total datasites")
        ax.set_xlabel(xlabel,fontsize = 12)
        ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
        ax.set_xlim(xedges[0],xedges[-1])
        ax.set_ylim(yedges[0],yedges[-1])
        if self.dep_log: ax.set_yscale('log')
        if self.ind_log: ax.set_xscale('log')
        
    def plotFunction(self,ax,region=None,color='k',shift=0):
        """Plot the mean response with standard deviation as error bars.

        Parameters
        ----------
        ax : matplotlib axis
            the axis on which to plot the function
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region
        color : str, optional
            the color to use in plotting the line
        shift : float, optional
            shift expressed as a fraction on [-0.5,0.5] used to shift
            the plotting of the errorbars so that multiple functions do
            not overlab
        """
        key = "default" if region is None else region
        if key not in self.dist.keys(): self.buildResponse(region=region)            
        y = self.dist[key][3]
        e = self.dist[key][4]
        xedges = self.dist[key][1]
        yedges = self.dist[key][2]
        xlabel = self.ind.name + " [%s]" % (UnitStringToMatplotlib(self.ind.unit))
        ylabel = self.dep.name + " [%s]" % (UnitStringToMatplotlib(self.dep.unit))
        x = 0.5*(xedges[:-1]+xedges[1:]) + shift*np.diff(xedges).mean()
        mask = y.mask
        if type(mask) == np.bool_: mask = np.asarray([mask]*y.size)
        x = x[mask==False]
        y = y[mask==False]
        e = e[mask==False]
        if self.dep_log:
            e = np.asarray([10**(np.log10(y)-np.log10(e)),10**(np.log10(y)+np.log10(e))])
        ax.errorbar(x,y,yerr=e,fmt='-o',color=color)
        ax.set_xlabel(xlabel,fontsize = 12)
        ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
        ax.set_xlim(xedges[0],xedges[-1])
        ax.set_ylim(yedges[0],yedges[-1])
        if self.dep_log: ax.set_yscale('log')
        if self.ind_log: ax.set_xscale('log')

    def plotModel(self,ax,region=None,color='k',prediction=False):
        """Plot the mean response with standard deviation as error bars.

        Parameters
        ----------
        ax : matplotlib axis
            the axis on which to plot the function
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region
        color : str, optional
            the color to use in plotting the line
        shift : float, optional
            shift expressed as a fraction on [-0.5,0.5] used to shift
            the plotting of the errorbars so that multiple functions do
            not overlab
        """
        key    = "default" if region is None else region
        if key not in self.dist.keys(): self.buildResponse(region=region)
        if self.dist[key][5] is None: return
        p = self.dist[key][5]        
        i = self.dist[key][6]
        xedges = self.dist[key][1]
        yedges = self.dist[key][2]
        x = np.linspace(xedges[0],xedges[-1],200)
        y = np.polyval(p,x)
        if self.dep_log:
            y = 10**y
        xlabel = self.ind.name + " [%s]" % (UnitStringToMatplotlib(self.ind.unit))
        ylabel = self.dep.name + " [%s]" % (UnitStringToMatplotlib(self.dep.unit))
        ax.plot(x,y,'-',color=color)
        if prediction:
            if self.dep_log:
                yt = 10**(np.log10(y)-i)
                ax.plot(x,10**(np.log10(y)-i),'--',color=color)
                yt = 10**(np.log10(y)+i)
                ax.plot(x,10**(np.log10(y)+i),'--',color=color)
            else:
                ax.plot(x,y,'--',color=color)
                ax.plot(x,y,'--',color=color)                
        ax.set_xlabel(xlabel,fontsize = 12)
        ax.set_ylabel(ylabel,fontsize = 12 if len(ylabel) <= 60 else 10)
        ax.set_xlim(xedges[0],xedges[-1])
        ax.set_ylim(yedges[0],yedges[-1])
        if self.dep_log: ax.set_yscale('log')
        if self.ind_log: ax.set_xscale('log')
        
    def scoreRMSE(self,r,region=None):
        """Given another relationship, computes a RMSE score based on the mean functional responses.

        Parameter
        ---------
        r : Relationship
            a relationship to compare to this relationship
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region

        Returns
        -------
        S : float
            the relative RMSE error in the functional representations
            of the relationship, mapped to a score on the unit
            interval where higher values are better

        """
        key = "default" if region is None else region
        self.makeComparable(r,region=region)
        
        # Compute the relative RMSE of the functions
        ref = self.dist[key][3].copy()
        com = r   .dist[key][3].copy()
        mask = ref.mask + com.mask
        ref = np.ma.masked_array(ref.data,mask=mask).compressed()
        com = np.ma.masked_array(com.data,mask=mask).compressed()
        if self.dep_log:
            ref = np.log10(ref)
            com = np.log10(com)
        S = np.exp(-np.linalg.norm(ref-com)/np.linalg.norm(ref))

        return S

    def scoreHellinger(self,r,region=None):
        """Given another relationship, computes the Hellenger score, which is 1 minus the Hellinger distance.

        Parameter
        ---------
        r : Relationship
            a relationship to compare to this relationship
        region : str, optional
            if the variables are spatial, restricts the response to
            cover only the cells defined by the given ILAMB Region

        Returns
        -------
        H : float
            the Hellinger distance, a measure of how well the
            relationship r approximates this relationship where small
            values are better

        """
        key = "default" if region is None else region
        self.makeComparable(r,region=region)
        
        # Compute the Hellinger Distance
        ref = self.dist[key][0].copy()
        com = r   .dist[key][0].copy()
        mask = ref.mask + com.mask
        ref = np.ma.masked_array(ref.data,mask=mask).compressed()
        com = np.ma.masked_array(com.data,mask=mask).compressed()
        H = 1-np.sqrt(((np.sqrt(ref)-np.sqrt(com))**2).sum())/np.sqrt(2)

        return H
    
