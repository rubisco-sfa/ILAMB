from __future__ import division
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
#import pylab as plt
import numpy as np
import os,glob

if 0:
    M = ModelResult("/chrysaor/CMIP5/CanESM2/esmHistorical",filter="r1i1p1")
    t,co2,unit = M.extractPointTimeSeries("co2",19.4,24.4)
    #plt.plot(t,co2,'-')
    #plt.show()

if 0:
    from netCDF4 import Dataset
    root = "./data"
    lat,lon = 19.4,24.4
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        M = ModelResult(subdir,filter="r1i1p1")
        for fname in glob.glob("%s/*%s*.nc" % (M.path,M.filter)):
            t,co2,unit = il.ExtractPointTimeSeries(fname,"co2",19.4,24.4)
            print np.ma.mean(co2)
            plt.plot(t,co2,'-')
    plt.show()

if 1:
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        M = ModelResult(subdir,filter="r1i1p1")
        try:
            t,co2,unit = M.extractPointTimeSeries("co2",19.4,24.4)
        except:
            continue
        model = subdir.replace(root,"").replace("esmHistorical","").replace("/","")
        print "%s\t%s\t%f" % (model,unit,np.ma.mean(co2))
        #plt.plot(t,co2,'-',label="%s" % model)

    #plt.legend(loc=2)
    #plt.show()


# BNU-ESM
# CanESM2
