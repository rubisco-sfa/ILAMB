import glob
import ilamblib as il
from constants import convert
import numpy as np
from netCDF4 import Dataset
import os
from Variable import Variable

class ModelResult():
    """A class for exploring model results.
    """
    def __init__(self,path,modelname="unamed",color=(0,0,0),filter=""):
        self.path           = path
        self.color          = color
        self.filter         = filter
        self.name           = modelname
        self.confrontations = {}
        self.cell_areas     = None
        self.land_fraction  = None
        self.land_areas     = None
        self.land_area      = None
        self.lat            = None
        self.lon            = None        
        self.lat_bnds       = None
        self.lon_bnds       = None        
        self._getGridInformation()

    def _fileExists(self,contains):
        """Looks through the model result path for a file that contains the text specified in "constains". Returns "" if not found.
        """
        fname = ""
        for subdir, dirs, files in os.walk(self.path):
            for f in files:
                if contains not in f: continue
                if ".nc" not in f: continue
                fname = "%s/%s" % (subdir,f)
                return fname
        return fname

    def _getGridInformation(self):
        # Look for a file named areacella...
        fname = self._fileExists("areacella")
        if fname == "": return # there are no areas associated with this model result

        # Now grab area information for this model
        f = Dataset(fname)
        self.cell_areas    = f.variables["areacella"][...]
        self.lat           = f.variables["lat"][...]
        self.lon           = f.variables["lon"][...]
        self.lat_bnds      = np.zeros(self.lat.size+1)
        self.lat_bnds[:-1] = f.variables["lat_bnds"][:,0]
        self.lat_bnds[-1]  = f.variables["lat_bnds"][-1,1]
        self.lon_bnds      = np.zeros(self.lon.size+1)
        self.lon_bnds[:-1] = f.variables["lon_bnds"][:,0]
        self.lon_bnds[-1]  = f.variables["lon_bnds"][-1,1]

        # Now we do the same for land fractions
        fname = self._fileExists("sftlf")
        if fname == "": 
            self.land_areas = self.cell_areas 
        else:
            self.land_fraction = (Dataset(fname).variables["sftlf"])[...]
            # some models represent the fraction as a percent 
            if np.ma.max(self.land_fraction) > 1: self.land_fraction *= 0.01 
            self.land_areas = self.cell_areas*self.land_fraction
        self.land_area = np.ma.sum(self.land_areas)
        return

    def diagnose(self):
        print "Diagnosing the %s model..." % self.name
        if self.land_areas is not None:
            from pylab import subplots
            from Post import GlobalPlot
            fig,ax = subplots(nrows=3,figsize=(12,18))
            GlobalPlot(self.lat,self.lon,self.cell_areas,shift=True,ax=ax[0],region="global.large")
            GlobalPlot(self.lat,self.lon,self.land_fraction,shift=True,ax=ax[1],region="global.large")
            GlobalPlot(self.lat,self.lon,self.land_areas,shift=True,ax=ax[2],region="global.large")
            ax[0].set_title("areacella")
            ax[1].set_title("sftlf")
            ax[2].set_title("areacella*sftlf")
            fig.savefig("land_areas_%s.png" % self.name)
        return

    def __str__(self):
        out  = "Model Result\n"
        out += "------------\n"
        out += "  Name: %s\n" % self.name
        out += "\n"
        return out
                
    def extractTimeSeries(self,variable,lats=None,lons=None,alt_vars=[],initial_time=-1e20,final_time=1e20,output_unit=""):
        """Extracts a time series of the given variable from the model
        results given a latitude and longitude.

        This routine will look for netCDF files with the "nc" suffix
        in the model directory. It will open all such files looking
        for the specified variable name. If the variable is found at
        the given latitude and longitude as defined by
        ILAMB.ilamblib.ExtractPointTimeSeries and at least partially
        on the desired time interval, this data is added to a
        list. Optionally a user may specify alternative variables, or
        alternative names of variables and the function will look for
        these also, giving preference to the given variable. After
        examining all files, then the routine will sort the list in
        ascending time and then check/disgard overlapping time
        segments. Finally, a composite data array is returned.

        Parameters
        ----------
        variable : string
            name of the variable to extract
        alt_vars: list of strings, optional
            alternate variables to search for if `variable' is not found
        initial_time : float, optional
            include model results occurring after this time
        final_time : float, optional
            include model results occurring before this time
        output_unit : string, optional
            if specified, will try to convert the units of the variable 
            extract to these units given. (See convert in ILAMB.constants)

        Returns
        -------
        t : numpy.ndarray
            a 1D array of times in days since 00:00:00 1/1/1850
        var : numpy.ma.core.MaskedArray
            an array of the extracted variable
        unit : string
            a description of the extracted unit

        """
        altvars = list(alt_vars)
        altvars.insert(0,variable)

        # create a list of data which has a non-null intersection over the desired time range
        data   = []
        ntimes = 0
        for subdir, dirs, files in os.walk(self.path):
            for f in files:
                if ".nc"       not in f: continue
                if self.filter not in f: continue
                fname = "%s/%s" % (subdir,f)
                found = False
                for vname in altvars:
                    try:
                        if lats is None and lons is None:
                            t,var,unit,lat,lon = il.ExtractTimeSeries(fname,vname)
                        else:
                            lat = lats
                            lon = lons
                            t,var,unit = il.ExtractPointTimeSeries(fname,vname,lat,lon)
                        nt      = ((t>=initial_time)*(t<=final_time)).sum()
                        ntimes += nt
                        if nt == 0: continue
                        data.append((t,var,vname,nt))
                    except il.VarNotInFile: 
                        continue
        if ntimes == 0: 
            raise il.VarNotInModel("These variable(s) do not exist in this model on that time frame: %s" % (",".join(altvars)))
            
        # a model might have the variable and its alternates, only use the highest preference variable present
        thin   = []
        for vname in altvars:
            for d in data:
                if d[2] == vname: thin.append(d)
            if len(thin) > 0: break
        data = thin

        # now check again that data exists on this time frame
        ntimes = 0
        for d in data: ntimes += d[3]
        if ntimes == 0: 
            raise il.VarNotInModel("These variable(s) do not exist in this model on that time frame: %s" % (",".join(altvars)))

        # sort the list by the first time, create a composite array
        data = sorted(data,key=lambda entry: entry[0][0])
        mono = np.asarray([entry[0][-1] for entry in data])
        mono = mono[:-1]>mono[1:]
        if mono.sum() > 0: # there seems to be some overlapping data so I will remove it
            for i in range(mono.shape[0]): 
                if mono[i]: 
                    tmp     = data.pop(i)
                    ntimes -= ((tmp[0]>=initial_time)*(tmp[0]<=final_time)).sum()
        shp  = [ntimes]; shp.extend(data[0][1].shape[1:])
        tc   = np.zeros(ntimes)
        varc = np.zeros(shp)
        masc = np.zeros(shp,dtype=bool)
        begin = 0
        for d in data:
            t,var,vname,nt = d
            mask = (t>=initial_time)*(t<=final_time)
            n = mask.sum(); end = begin+n
            tc  [begin:end] =        t[mask]
            varc[begin:end] = var.data[mask]
            if var.mask.size == 1: # whole array is either completely masked or not
                masc[begin:end] = var.mask
            else:
                masc[begin:end] = var.mask[mask]
            begin = end

        # if you asked for a specific unit, try to convert
        if output_unit is not "":
            try:
                varc *= convert[variable][output_unit][unit]
                unit = output_unit
            except:
                raise il.UnknownUnit("Variable is in units of [%s], you asked for [%s] but I do not know how to convert" % (unit,output_unit))

        if lats is None:
            return Variable(np.ma.masked_array(varc,mask=masc),unit,time=tc,lat=self.lat,lon=self.lon,area=self.land_areas,name=vname)
        else:
            return Variable(np.ma.masked_array(varc,mask=masc),unit,time=tc,lat=lat,lon=lon,name=vname)
            
    def globalPlot(self,var,region="global",ax=None):
        from mpl_toolkits.basemap import Basemap
        from pylab import cm
        from matplotlib.colors import from_levels_and_colors
        from constants import regions
        lats,lons = regions[region]
        bmap = Basemap(projection='cyl',
                       llcrnrlon=lons[ 0],llcrnrlat=lats[ 0],
                       urcrnrlon=lons[-1],urcrnrlat=lats[-1],
                       resolution='c',ax=ax)
        nroll = np.argmin(np.abs(self.lon-180))
        lon   = np.roll(self.lon,nroll); lon[:nroll] -= 360
        x,y   = bmap(lon,self.lat)
        cmap,norm = from_levels_and_colors([0,0.075,0.15,0.25,0.75,1.5,3.5,7.5,12.5,17.5,22.0],
                                           ['deepskyblue',
                                            'cyan',
                                            'green',
                                            'greenyellow',
                                            'yellow',
                                            'sandybrown',
                                            'darkorange',
                                            'r',
                                            'firebrick',
                                            'indigo'])
        ax    = bmap.pcolormesh(x,y,np.roll(var,nroll,axis=1),zorder=2) #,cmap=cmap,norm=norm)
        bmap.drawmeridians(np.arange(-150,151,30),labels=[0,0,0,1],zorder=1,dashes=[1000000,1],linewidth=0.5)
        bmap.drawparallels(np.arange( -90, 91,30),labels=[1,0,0,0],zorder=1,dashes=[1000000,1],linewidth=0.5)
        bmap.drawcoastlines(linewidth=0.5)
        bmap.colorbar(ax) #,ticks=[0.05,0.1,0.2,0.5,1,2,5,10,15,20])
        return ax

