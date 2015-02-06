from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
from ILAMB.Post import ConfrontationTableASCII
import pylab as plt
import numpy as np
import os,pickle

if not os.path.isfile("CMIP5_CO2_MaunaLoa.pkl"):
    # Initialize the models
    M    = []
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()
        M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
        
    Cs = Confrontation()
    C  = (Cs.list())[0]

    for m in M:
        try:
            m.confrontations[C.name] = C.confront(m)        
            print m.name
        except il.VarNotInModel:
            print m.name,"X"
            continue

    f = file("CMIP5_CO2_MaunaLoa.pkl","wb")
    pickle.dump(M,f)
    f.close()

f = file("CMIP5_CO2_MaunaLoa.pkl","rb")
M = pickle.load(f)
f.close()

Cs = Confrontation()
C  = (Cs.list())[0]
M  = sorted(M,key=lambda model: model.name.lower())
print ConfrontationTableASCII(C.name,M)

