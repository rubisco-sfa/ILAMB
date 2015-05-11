"""This demo script is intended to show how this package's data
structures may be used to run the benchmark on the model results
cateloged in Mingquan's ftp site.
"""
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
import ILAMB.Post as post
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
print "\nSearching for model results in %s...\n" % root
maxML = 0
for subdir, dirs, files in os.walk(root):
    mname = subdir.replace(root,"")
    if mname.count("/") != 1: continue
    mname = mname.replace("/","")
    maxML  = max(maxML,len(mname))
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
M = sorted(M,key=lambda m: m.name.upper())
for m in M: print ("    {0:<%d}" % (maxML)).format(m.name)

# Assign colors
clrs = il.GenerateDistinctColors(len(M))
for m in M:
    clr     = clrs.pop(np.random.randint(0,high=len(clrs)))
    m.color = clr
    
# Confront models
C = Confrontation().list()

# Build work list, ModelResult+Confrontation pairs
W = []
maxCL = 0
for c in C:
    maxCL = max(maxCL,len(c.name))
    for m in M:
        W.append([m,c])

print "\nRunning model-confrontation pairs...\n"
for w in W:
    m,c = w
    t0  = time.time()
    try:
        print ("    {0:>%d} {1:>%d} " % (maxCL,maxML)).format(c.name,m.name),
        m.confrontations[c.name] = c.confront(m)  
        dt = time.time()-t0
        print ("%sCompleted%s {0:>5.1f} s" % (OKGREEN,ENDC)).format(dt)
    except il.VarNotInModel:
        print "%sVarNotInModel%s" % (FAIL,ENDC)
        continue
    except il.AreasNotInModel:
        print "%sAreasNotInModel%s" % (FAIL,ENDC)
        continue
    except il.VarNotMonthly:
        print "%sVarNotMonthly%s" % (FAIL,ENDC)
        continue

print "\nPost-processing...\n"

# Postprocess
for c in C:

    print "  %s\n" % c.name
    t0 = time.time()

    # HTML Google-chart table
    f = file("%s/%s.html" % (c.output_path,c.name),"w")
    f.write(post.ConfrontationTableGoogle(c,M))
    f.close()

    # generate plots
    c.plot(M)
    
    dt = time.time()-t0
    print "  Completed in %.1f seconds" % dt
