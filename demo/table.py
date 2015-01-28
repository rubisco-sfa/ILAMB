from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
import numpy as np
import os,pickle

# Initialize CO2 confrontation (Mauna Loa CO2 mole fraction)
C = Confrontation("./data")

if not os.path.isfile("CMIP5_CO2_MaunaLoa.pkl"):

    # Initialize the models to confront
    M = []
    root = "/chrysaor/CMIP5/"
    for subdir, dirs, files in os.walk(root):

        # If not a historical simulation, then skip
        if "esmHistorical" not in subdir: continue

        # Parse the model name from its directory and choose a unique color
        mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()

        # Initialize the model result, only use files with a "r1i1p1" in the filename
        M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))

    # Generate colors
    clrs = il.GenerateDistinctColors(len(M))

    # Confront away!
    for m in M: 
        m.color = clrs.pop(np.random.randint(0,high=len(clrs)))
        try:
            m.confrontations[C.name] = C.extractModelResult(m)
        except il.VarNotInModel:
            pass

    f = file("CMIP5_CO2_MaunaLoa.pkl","wb")
    pickle.dump(M,f)
    f.close()

f = file("CMIP5_CO2_MaunaLoa.pkl","rb")
M = pickle.load(f)
f.close()

M = sorted(M,key=lambda model: model.name.lower())

s = "\n{0:^16}{1:^26}{2:^26}{3:^26}{4:^26}{5:^26}".format("ModelName","MonthMean [ppm]","DecadeMeanAmplitude [ppm]","DecadeStdAmplitude [ppm]","DecadeTrend [ppm/yr]","DecadePhaseShift [d]")
print s
s = "{0:^16}{1:>13}{2:>13}{3:>13}{4:>13}{5:>13}{6:>13}{7:>13}{8:>13}{9:>13}{10:>13}".format("","Bias ","RMSE ","Bias ","RMSE ","Bias ","RMSE ","Bias ","RMSE ","","")
print s
for m in M:
    if C.name in m.confrontations.keys():
        metrics = m.confrontations[C.name]["metrics"]
        s = "{0:>15}{1:>+13,.3f}{2:>13,.3f}{3:>+13,.3f}{4:>13,.3f}{5:>+13,.3f}{6:>13,.3f}{7:>+13,.3f}{8:>13,.3f}{9:>+26,.3f}".format(m.name,metrics["RawBias"],metrics["RawRMSE"],metrics["AmpMeanBias"],metrics["AmpMeanRMSE"],metrics["AmpStdBias"],metrics["AmpStdRMSE"],metrics["TrendBias"],metrics["TrendRMSE"],metrics["PhaseShiftMean"]*365.)
        print s
    else:
        s = "{0:>15}{1:>13}{2:>13}{3:>13}{4:>13}{5:>13}{6:>13}{7:>13}{8:>13}{9:>26}".format(m.name,"~","~","~","~","~","~","~","~","~")
        print s
print ""

