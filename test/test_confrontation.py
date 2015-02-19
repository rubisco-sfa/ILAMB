from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
from ILAMB.Post import ConfrontationTableASCII
import pylab as plt
import numpy as np
import os,pickle

if not os.path.isfile("XCMIP5_CO2_MaunaLoa.pkl"):
    # Initialize the models
    M    = []
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):
        if "esmHistorical" not in subdir: continue
        mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()
        M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))
        
    Cs = Confrontation()
    C  = (Cs.list())[1]

    for m in M:
        try:
            m.confrontations[C.name] = C.confront(m)  
            print m.name
        except il.VarNotInModel:
            continue

