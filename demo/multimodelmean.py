"""This demo script is intended to show how this package's data
structures may be used to run the benchmark on the model results
cateloged in Mingquan's ftp site.
"""
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
from netCDF4 import Dataset
import ILAMB.Post as post
import pylab as plt
import numpy as np
import os,time

# Initialize the models
M    = []
root = "%s/MODELS/CMIP5" % (os.environ["ILAMB_ROOT"])
print "\nSearching for model results in %s..." % root
maxL = 0
for subdir, dirs, files in os.walk(root):
    mname = subdir.replace(root,"")
    if mname.count("/") != 1: continue
    mname = mname.replace("/","").upper()
    if "MEAN" in mname: continue
    maxL  = max(maxL,len(mname))
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
M = sorted(M,key=lambda m: m.name.upper())
for m in M: print ("    {0:<%d}" % (maxL)).format(m.name)

t,lat,lon,mean_gpp,num_model,models = il.MultiModelMean(M,"gpp","g m-2 s-1",0.5,res_lat=0.9375,res_lon=1.25)

f = Dataset("gpp_CMIP5_Multimodelmean_historical_r1i1p1_195601-200512.nc",mode="w")
f.createDimension("time")
f.createDimension("lat",size=lat.size)
f.createDimension("lon",size=lon.size)

T = f.createVariable("time","double",("time"))
T.setncattr("units","days since 1850-01-01 00:00:00")
T.setncattr("calendar","noleap")
T.setncattr("axis","T")
T.setncattr("long_name","time")
T.setncattr("standard_name","time")
T[...] = t

X = f.createVariable("lon","double",("lon"))
X.setncattr("units","degrees_east")
X.setncattr("axis","X")
X.setncattr("long_name","longitude")
X.setncattr("standard_name","longitude")
X[...] = lon

Y = f.createVariable("lat","double",("lat"))
Y.setncattr("units","degrees_north")
Y.setncattr("axis","Y")
Y.setncattr("long_name","latitude")
Y.setncattr("standard_name","latitude")
Y[...] = lat

G = f.createVariable("gpp","double",("time","lat","lon"))
G.setncattr("standard_name","gross_primary_productivity_of_carbon")
G.setncattr("long_name","Carbon Mass Flux out of Atmosphere due to Gross Primary Production on Land")
G.setncattr("units","g m-2 s-1")
G.setncattr("original_name","GPP")
G.setncattr("comment","Mean GPP computed from an average over models (%s). See num variable for number of contributing models at a point in time and space." % models)
G.setncattr("_FillValue",mean_gpp.fill_value)
G[...] = mean_gpp

N = f.createVariable("num","int",("time","lat","lon"))
N.setncattr("standard_name","Number of contributing models")
N.setncattr("long_name","Number of models which contribute to the mean value")
N.setncattr("units","")
N.setncattr("original_name","num")
N[...] = num_model

f.close()

