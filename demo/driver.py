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
from mpi4py import MPI

# MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Some color constants for printing to the terminal
OK   = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# Initialize the models
M    = []
root = "%s/MODELS/CMIP5" % (os.environ["ILAMB_ROOT"])
if rank == 0: print "\nSearching for model results in %s...\n" % root
maxML = 0
for subdir, dirs, files in os.walk(root):
    mname = subdir.replace(root,"")
    if mname.count("/") != 1: continue
    mname = mname.replace("/","")
    maxML  = max(maxML,len(mname))
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
M = sorted(M,key=lambda m: m.name.upper())
if rank == 0: 
    for m in M: 
        print ("    {0:<%d}" % (maxML)).format(m.name)

# Assign colors
clrs = il.GenerateDistinctColors(len(M))
for m in M:
    clr     = clrs.pop(np.random.randint(0,high=len(clrs)))
    m.color = clr
    
# Build work list, ModelResult+Confrontation pairs
W     = []
C     = Confrontation().list()
maxCL = 0
for c in C:
    maxCL = max(maxCL,len(c.name))
    for m in M:
        W.append([m,c])

if rank==0: print "\nRunning model-confrontation pairs...\n"
comm.Barrier()

# Divide work list and go
wpp   = float(len(W))/size
begin = int(round( rank   *wpp))
end   = int(round((rank+1)*wpp))
T0    = time.time()
for w in W[begin:end]:
    m,c = w
    t0  = time.time()
    try:
        c.confront(m)  
        dt = time.time()-t0
        print ("    {0:>%d} {1:>%d} %sCompleted%s {2:>5.1f} s" % (maxCL,maxML,OK,ENDC)).format(c.name,m.name,dt)
    except il.VarNotInModel:
        print ("    {0:>%d} {1:>%d} %sVarNotInModel%s" % (maxCL,maxML,FAIL,ENDC)).format(c.name,m.name)
        continue
    except il.AreasNotInModel:
        print ("    {0:>%d} {1:>%d} %sAreasNotInModel%s" % (maxCL,maxML,FAIL,ENDC)).format(c.name,m.name)
        continue
    except il.VarNotMonthly:
        print ("    {0:>%d} {1:>%d} %sVarNotMonthly%s" % (maxCL,maxML,FAIL,ENDC)).format(c.name,m.name)
        continue

comm.Barrier()

if rank==0:
    for c in C:
        c.plotFromFiles()

comm.Barrier()

if rank==0: print "\nCompleted in {0:>5.1f} s\n".format(time.time()-T0)





