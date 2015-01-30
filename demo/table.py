from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
import pylab as plt
import numpy as np
import os,pickle,sys

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

# Load the model data and sort
f = file("CMIP5_CO2_MaunaLoa.pkl","rb")
M = pickle.load(f)
f.close()
M = sorted(M,key=lambda model: model.name.lower())

# Generate an ASCII table
s  = "{0:^16}{1:^26}{2:^26}{3:^26}{4:^26}{5:^26}\n".format("ModelName","MonthMean [ppm]","DecadeMeanAmplitude [ppm]","DecadeStdAmplitude [ppm]","DecadeTrend [ppm/yr]","DecadePhaseShift [d]")
s += "{0:^16}{1:>13}{2:>13}{3:>13}{4:>13}{5:>13}{6:>13}{7:>13}{8:>13}{9:>13}{10:>13}\n".format("","Bias ","RMSE ","Bias ","RMSE ","Bias ","RMSE ","Bias ","RMSE ","","")
for m in M:
    if C.name in m.confrontations.keys():
        metrics = m.confrontations[C.name]["metrics"]
        s += "{0:>15}{1:>+13,.3f}{2:>13,.3f}{3:>+13,.3f}{4:>13,.3f}{5:>+13,.3f}{6:>13,.3f}{7:>+13,.3f}{8:>13,.3f}{9:>+26,.3f}\n".format(m.name,metrics["RawBias"],metrics["RawRMSE"],metrics["AmpMeanBias"],metrics["AmpMeanRMSE"],metrics["AmpStdBias"],metrics["AmpStdRMSE"],metrics["TrendBias"],metrics["TrendRMSE"],metrics["PhaseShiftMean"]*365.)
    else:
        s += "{0:>15}{1:>13}{2:>13}{3:>13}{4:>13}{5:>13}{6:>13}{7:>13}{8:>13}{9:>26}\n".format(m.name,"~","~","~","~","~","~","~","~","~")
f = file("CMIP5_CO2_summary.txt","w")
f.write(s)
f.close()
print "\n%s" % s

def SetFancyPlotOptions():
    fsize  = 14
    params = {'axes.titlesize':fsize,
              'axes.labelsize':fsize,
              'font.size':fsize,
              'legend.fontsize':fsize,
              'xtick.labelsize':fsize,
              'ytick.labelsize':fsize}
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

SetFancyPlotOptions()

# Generate plots
for m in M:
    if C.name not in m.confrontations.keys(): continue
    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(8*1.618034,8),tight_layout=True)

    # Plot raw data
    lbls = {"co2":m.name ,"data":"Mauna Loa"}
    clrs = {"co2":m.color,"data":'k'}
    lws  = {"co2":1      ,"data":2  }
    als  = {"co2":1      ,"data":0.5}
    inds = {"co2":-1      ,"data":0}
    algs = {"co2":"left"  ,"data":"right"}
    shift = 0.5
    for key in ["co2","data"]:
        c = m.confrontations[C.name]
        t = c[key]["t"]/365.+1850
        v = c[key]["var"]
        ax[0,0].plot(t,v,'-',color=clrs[key],lw=lws[key],alpha=als[key],label=lbls[key])
        ax[0,0].plot(t[-1]      ,v[-1],'o'           ,color=clrs[key],mew=0,markersize=4,alpha=als[key])
        ax[0,0].text(t[-1]+shift,v[-1],"%.2f" % v[-1],color=clrs[key],ha='left',va='baseline',alpha=als[key])

        t = c[key]["Decadal"]["t"]/365.+1850
        v = c[key]["Decadal"]["amplitude_mean"]
        e = c[key]["Decadal"]["amplitude_std"]
        ax[0,1].errorbar(t,v,yerr=e,fmt='-',color=clrs[key],lw=lws[key],alpha=als[key])
        ax[0,1].plot(t[-1]      ,v[-1],'o'           ,color=clrs[key],mew=0,markersize=4,alpha=als[key])
        ax[0,1].text(t[-1]+shift,v[-1],"%.2f" % v[-1],color=clrs[key],ha='left',va='baseline',alpha=als[key])

        t = c[key]["t"]/365.+1850
        v = c[key]["Trend"]
        ax[1,0].plot(t,v,'-',color=clrs[key],lw=lws[key],alpha=als[key],label=lbls[key])
        ax[1,0].plot(t[-1]      ,v[-1],'o'           ,color=clrs[key],mew=0,markersize=4,alpha=als[key])
        ax[1,0].text(t[-1]+shift,v[-1],"%.2f" % v[-1],color=clrs[key],ha='left',va='baseline',alpha=als[key])

        t  = c[key]["Decadal"]["t"]/365.+1850
        vn = c[key]["Decadal"]["tmin"]*365
        vx = c[key]["Decadal"]["tmax"]*365
        ax[1,1].plot(t,vn,'-',color=clrs[key],lw=lws[key],alpha=als[key],label=lbls[key])
        ax[1,1].text(t[inds[key]]+shift,vn[inds[key]],"min(CO$_2$)" % v[-1],color=clrs[key],ha=algs[key],va='baseline',alpha=als[key])
        ax[1,1].plot(t,vx,'--',color=clrs[key],lw=lws[key],alpha=als[key],label=lbls[key])
        ax[1,1].text(t[inds[key]]+shift,vx[inds[key]],"max(CO$_2$)" % v[-1],color=clrs[key],ha=algs[key],va='baseline',alpha=als[key])

    ax[0,0].legend(loc=2)
    ax[0,0].set_xlim(1955,2022)
    xmin,xmax = ax[0,0].get_xlim(); dx = xmax-xmin
    ymin,ymax = ax[0,0].get_ylim(); dy = ymax-ymin
    ax[0,0].text(xmax-0.05*dx,ymin+0.05*dy,
                 "Bias = %+.2f\nRMSE = %.2f" % (c["metrics"]["RawBias"],c["metrics"]["RawRMSE"]),
                 ha="right",va="bottom")
    ax[0,0].set_xlabel("Year")
    ax[0,0].set_ylabel("CO$_{2}$ Concentration [ppm]")
    ax[0,0].set_title("Monthly Mean")

    ax[0,1].set_xlim(ax[0,0].get_xlim())
    xmin,xmax = ax[0,1].get_xlim(); dx = xmax-xmin
    ymin,ymax = ax[0,1].get_ylim(); dy = ymax-ymin
    ax[0,1].text(xmax-0.05*dx,ymin+0.05*dy,
                 "Bias = %+.2f (+%.2f)\nRMSE = %.2f (%.2f)" % (c["metrics"]["AmpMeanBias"],
                                                               c["metrics"]["AmpStdBias"],
                                                               c["metrics"]["AmpMeanRMSE"],
                                                               c["metrics"]["AmpStdRMSE"]),
                 ha="right",va="bottom")
    ax[0,1].set_xlabel("Year")
    ax[0,1].set_ylabel("CO$_{2}$ Concentration [ppm]")
    ax[0,1].set_title("Mean Decadal Amplitude Shift")

    ax[1,0].set_xlim(ax[0,0].get_xlim())
    xmin,xmax = ax[1,0].get_xlim(); dx = xmax-xmin
    ymin,ymax = ax[1,0].get_ylim(); dy = ymax-ymin
    ax[1,0].text(xmax-0.05*dx,ymax-0.05*dy,
                 "Bias = %+.2f\nRMSE = %.2f" % (c["metrics"]["TrendBias"],c["metrics"]["TrendRMSE"]),
                 ha="right",va="top")
    ax[1,0].set_xlabel("Year")
    ax[1,0].set_ylabel("CO$_{2}$ Concentration Trend [ppm/year]")
    ax[1,0].set_title("Windowed decadal trend")

    ax[1,1].set_ylim(0,365)
    ax[1,1].set_yticks([15.5,45.,74.5,105.,135.5,166.,196.5,227.5,258.,288.5,319.,349.5])
    ax[1,1].set_yticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax[1,1].set_xlim(ax[0,0].get_xlim())
    ax[1,1].set_xlabel("Year")
    ax[1,1].set_ylabel("Time of year [day]")
    ax[1,1].set_title("Decadal mean min/max CO$_2$ concentration time of year")

    fig.savefig("%s.pdf" % m.name)
    plt.close('all')

