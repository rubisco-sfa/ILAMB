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
print "\nSearching for model results in %s..." % root
maxL = 0
for subdir, dirs, files in os.walk(root):
    mname = subdir.replace(root,"")
    if mname.count("/") != 1: continue
    mname = mname.replace("/","").upper()
    maxL  = max(maxL,len(mname))
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
M = sorted(M,key=lambda m: m.name.upper())
for m in M: print ("    {0:<%d}" % (maxL)).format(m.name)

# Assign colors
clrs = il.GenerateDistinctColors(len(M))
for m in M:
    clr     = clrs.pop(np.random.randint(0,high=len(clrs)))
    m.color = clr
    
# Confront models
C = Confrontation()
print "\n%s" % C
C = C.list()

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

print "\nPost-processing..."

# Put everything here
build = "./_build"
try:
    os.mkdir(build)
except:
    pass

# Postprocess
for c in C:

    print "\n  %s" % c.name
    t0 = time.time()
    path = "%s/%s" % (build,c.name)
    try:
        os.mkdir(path)
    except:
        pass

    # HTML Google-chart table
    f = file("%s/%s.html" % (path,c.name),"w")
    f.write(post.ConfrontationTableGoogle(c,M))
    f.close()

    # generate plots
    c.plot(M,path=path)
    
    dt = time.time()-t0
    print "  Completed in %.1f seconds" % dt
