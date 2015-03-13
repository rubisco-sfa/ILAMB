"""This demo script is intended to show how this package's data
structures may be used to run the benchmark on the model results
cateloged in Mingquan's ftp site.
"""
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
from ILAMB.Post import ConfrontationTableASCII
import pylab as plt
import numpy as np
import os,time

# Some color constants for printing to the terminal
OKGREEN = '\033[92m'
FAIL    = '\033[91m'
ENDC    = '\033[0m'

# Initialize the models
M    = []
root = "%s/MODELS/CMIP5" % (os.environ["ILAMB_ROOT"])
for subdir, dirs, files in os.walk(root):
    mname = subdir.replace(root,"")
    if mname.count("/") != 1: continue
    mname = mname.replace("/","")
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))

M = sorted(M,key=lambda m: m.name.upper())

# Assign colors
clrs = il.GenerateDistinctColors(len(M))
maxL = 0
for m in M:
    clr     = clrs.pop(np.random.randint(0,high=len(clrs)))
    m.color = clr
    maxL    = max(maxL,len(m.name))

# Confront models
C = Confrontation().list()
print "\nRunning confrontations..."
for c in C:
    t0 = time.time()
    print "\n  %s" % c.name
    for m in M:
        try:
            m.confrontations[c.name] = c.confront(m)  
            print ("    {0:<%d} %sCompleted%s" % (maxL,OKGREEN,ENDC)).format(m.name)
        except il.VarNotInModel:
            print ("    {0:<%d} %sVarNotInModel%s" % (maxL,FAIL,ENDC)).format(m.name) 
            continue
        except il.AreasNotInModel:
            print ("    {0:<%d} %sAreasNotInModel%s" % (maxL,FAIL,ENDC)).format(m.name)
            continue
        except il.VarNotMonthly:
            print ("    {0:<%d} %sVarNotMonthly%s" % (maxL,FAIL,ENDC)).format(m.name)
            continue
    dt = time.time()-t0
    print "  Completed in %.1f seconds" % dt

# Postprocess
for c in C:
    print ""
    print ConfrontationTableASCII(c,M)

