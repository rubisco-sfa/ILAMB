from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
import pylab as plt
import numpy as np
import os

print "Checking models for CO2 data..."

# Setup a plot
fsize  = 18
params = {'axes.titlesize':fsize,
          'axes.labelsize':fsize,
          'font.size':fsize,
          'legend.fontsize':fsize,
          'xtick.labelsize':fsize,
          'ytick.labelsize':fsize}
plt.rcParams.update(params)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
fig,ax = plt.subplots(figsize=(6*1.618034,6),tight_layout=True)
ax.set_xlabel("Year")
ax.set_ylabel("CO$_{2}$ Concentration [ppm]")
ax.set_xlim(1955,2027)

# Initialize CO2 confrontation (Mauna Loa CO2 mole fraction)
C = Confrontation("./data")

# Generate colors for 20 models
clrs = il.GenerateDistinctColors(20)

# Loop through the models and look for confrontation data
root = "/chrysaor/CMIP5/"
for subdir, dirs, files in os.walk(root):

    # If not a historical simulation, then skip
    if "esmHistorical" not in subdir: continue

    # Parse the model name from its directory and choose a unique color
    mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()
    clr = clrs.pop(np.random.randint(0,high=len(clrs)))

    # Initialize the model result, only use files with a "r1i1p1" in the filename
    M = ModelResult(subdir,modelname=mname,color=clr,filter="r1i1p1")

    # Try to extract the model result which the confrontation
    # needs. If the model does not have a usable quantity to compare
    # against the confrontation, then the extractModelResult()
    # function will return an exception of the type
    # ILAMB.ilamblib.VarNotInModel. Rather than fail, here we will
    # catch this exception, print a ".", and continue
    print M.name
    try:
        # cdata will be a dictionary of all the variables from the
        # model result which this confrontation needs. In this case,
        # it should only return one variable, "co2".
        cdata = C.extractModelResult(M)            
        for key in cdata.keys():

            # For each variable, print the variable name, unit, and mean
            data = cdata[key]
            print "\t%s [%s]\tmean = %f" % (key,data["unit"],np.ma.mean(data["var"]))

            # Also plot and annotate
            t = data["t"]/365.+1850
            v = data["var"]
            ax.plot(t,v,'-',color=M.color,label=M.name)
            ax.text(t[-1],v[-1],M.name,color=M.color,ha="left",va="center")

    except il.VarNotInModel:
        # just print a "." to reflect no data was found
        print "\t."

# plot the Mauna Loa data
t = C.t/365.+1850
ax.plot(t,C.var,'-k',lw=3,alpha=0.25)
ax.text(t[-1],np.ma.min(C.var[-10:]),"Mauna Loa",color='k',alpha=0.5,ha="left",va="center")
fig.savefig("summaryco2.pdf")

