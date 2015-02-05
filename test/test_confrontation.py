from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
import pylab as plt
import numpy as np
import os

# Initialize the models
M    = []
root = "/chrysaor/CMIP5/"
for subdir, dirs, files in os.walk(root):
    if "esmHistorical" not in subdir: continue
    mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))

Con = Confrontation()
print Con
C   = Con.list()[0]
plt.plot(C.t,C.var,'-k')

for m in M:
    try:
        data = C.confront(m)
        print m.name
        plt.plot(data["model"]["t"],data["model"]["var"],'-')
        
    except il.VarNotInModel:
        print m.name,"X"
        continue

plt.show()

