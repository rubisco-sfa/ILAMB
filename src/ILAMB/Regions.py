from netCDF4 import Dataset
import numpy as np

class Regions(object):

    _regions = {}

    @property
    def regions(self):

        return Regions._regions.keys()

    def addRegionLatLonBounds(self,label,name,lats,lons):

        lat  = np.hstack([[- 90.],np.asarray(lats),[ 90.]])
        lon  = np.hstack([[-180.],np.asarray(lons),[180.]])
        mask = np.asarray([[1,1,1],
                           [1,0,1],
                           [1,1,1]],dtype=bool)
        Regions._regions[label] = [name,lat,lon,mask]

    def addRegionNetCDF4(self,filename):

        dset = Dataset(filename)

        # look for integer datasets defined on regular grids
        for var in dset.variables:
            v = dset.variables[var]
            if len(v.dimensions) == 2:
                lat = dset.variables[v.dimensions[0]][...]
                lon = dset.variables[v.dimensions[1]][...]
                lbl = dset.variables[v.labels       ][...]
                ids = np.ma.compressed(np.unique(v[...]))
                assert ids.max() < lbl.size
                for i in ids:
                    label = lbl[i].lower()
                    name  = lbl[i]
                    mask  = v[...].data != i
                    Regions._regions[label] = [name,lat,lon,mask]

    def getRegionName(self,label):

        return Regions._regions[label][0]

    def getMask(self,label,var):

        name,lat,lon,mask = Regions._regions[label]
        if lat.size == 4 and lon.size == 4:
            # if lat/lon bounds, find which bounds we are in
            rows = ((var.lat[:,np.newaxis]>=lat[:-1])*(var.lat[:,np.newaxis]<=lat[1:])).argmax(axis=1)
            cols = ((var.lon[:,np.newaxis]>=lon[:-1])*(var.lon[:,np.newaxis]<=lon[1:])).argmax(axis=1)
        else:
            # if more globally defined, nearest neighbor is fine
            rows = (np.abs(lat[:,np.newaxis]-var.lat)).argmin(axis=0)
            cols = (np.abs(lon[:,np.newaxis]-var.lon)).argmin(axis=0)
        if var.ndata: return mask[np.ix_(rows,cols)].diagonal()
        return mask[np.ix_(rows,cols)]
            
    def hasData(self,label,var):
        
        axes = range(var.data.ndim)
        if var.spatial: axes = axes[:-2]
        if var.ndata  : axes = axes[:-1]
        keep  = (var.data.mask == False).any(axis=tuple(axes))
        keep *= (self.getMask(label,var)==False)
        if keep.sum() > 0: return True
        return False
        
if "global" not in Regions().regions:
    
    # Populate some regions
    r = Regions()    
    r.addRegionLatLonBounds("global","Globe",(-89.75, 89.75),(-179.75, 179.75))

    # GFED regions
    r.addRegionLatLonBounds("bona","Boreal North America",             ( 49.75, 79.75),(-170.25,- 60.25))
    r.addRegionLatLonBounds("tena","Temperate North America",          ( 30.25, 49.75),(-125.25,- 66.25))
    r.addRegionLatLonBounds("ceam","Central America",                  (  9.75, 30.25),(-115.25,- 80.25))
    r.addRegionLatLonBounds("nhsa","Northern Hemisphere South America",(  0.25, 12.75),(- 80.25,- 50.25))
    r.addRegionLatLonBounds("shsa","Southern Hemisphere South America",(-59.75,  0.25),(- 80.25,- 40.25))
    r.addRegionLatLonBounds("euro","Europe",                           ( 35.25, 70.25),(- 10.25,  30.25))
    r.addRegionLatLonBounds("mide","Middle East",                      ( 20.25, 40.25),(- 10.25,  60.25))
    r.addRegionLatLonBounds("nhaf","Northern Hemisphere Africa",       (  0.25, 20.25),(- 20.25,  45.25))
    r.addRegionLatLonBounds("shaf","Southern Hemisphere Africa",       (-34.75,  0.25),(  10.25,  45.25))
    r.addRegionLatLonBounds("boas","Boreal Asia",                      ( 54.75, 70.25),(  30.25, 179.75))
    r.addRegionLatLonBounds("ceas","Central Asia",                     ( 30.25, 54.75),(  30.25, 142.58))
    r.addRegionLatLonBounds("seas","Southeast Asia",                   (  5.25, 30.25),(  65.25, 120.25))
    r.addRegionLatLonBounds("eqas","Equatorial Asia",                  (-10.25, 10.25),(  99.75, 150.25))
    r.addRegionLatLonBounds("aust","Australia",                        (-41.25,-10.50),( 112.00, 154.00))
    
if __name__ == "__main__":

    from Variable import Variable
    import os
    Vs = []
    Vs.append(Variable(filename = os.environ["ILAMB_ROOT"] + "/DATA/gpp/FLUXNET/gpp.nc",
                       variable_name = "gpp"))
    Vs.append(Variable(filename = os.environ["ILAMB_ROOT"] + "/DATA/gpp/FLUXNET-MTE/gpp_0.5x0.5.nc",
                       variable_name = "gpp"))
    Vs.append(Vs[-1].integrateInTime(mean=True))
    
    r   = Regions()
    for v in Vs:
        print "-----------"
        for region in r.regions:
            print region,r.getRegionName(region),v.name,r.hasData(region,v)
        
