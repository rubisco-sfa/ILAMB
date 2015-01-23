from __future__ import division
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
#import pylab as plt
import numpy as np
import os

if 0:
    M = ModelResult("/chrysaor/CMIP5/CanESM2/esmHistorical",filter="r1i1p1")
    t,co2,unit = M.extractPointTimeSeries("co2",19.4,24.4)
    #plt.plot(t,co2,'-')
    #plt.show()

if 1:
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        M = ModelResult(subdir,filter="r1i1p1")
        try:
            t,co2,unit = M.extractPointTimeSeries("co2",19.4,24.4)
        except:
            continue
        print subdir.replace(root,"").replace("/esmHistorical",""),co2.mask.sum(),co2.shape,unit


if 0:
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        M = ModelResult(subdir,filter="r1i1p1")
        try:
            t,co2 = M.extractPointTimeSeries("co2",19.4,24.4)
        except:
            continue
        print subdir,t.shape,co2.shape,type(co2)
        plt.plot(t,co2,'-',label="%s" % (subdir.split("/")[3]))

    plt.legend(loc=2)
    plt.show()

