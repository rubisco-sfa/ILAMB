from ILAMB.ModelResult import ModelResult
from ILAMB.Confrontation import Confrontation
from ILAMB import ilamblib as il
import pylab as plt
import numpy as np

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

shift = 0.75

# initialize model and confrontation
M = ModelResult  ("./data")
C = Confrontation("./data")
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(8*1.618034,8),tight_layout=True)

# plot raw data 
t ,co2  = C.extractModelResult(M)
dt,dco2 = C.getData(t.min(),t.max())

pt = dt/365.+1850
ax[0,0].plot(pt,dco2,'-g',label='observational data')
ax[0,0].plot(pt[-1],dco2[-1],'og',mew=0,markersize=4)
ax[0,0].text(pt[-1]+shift,dco2[-1],"%.2f" % dco2[-1],color='g',ha='left',va='baseline')

pt = dt/365.+1850
ax[0,0].plot(pt,co2,'-b',label='model data')
ax[0,0].plot(pt[-1],co2[-1],'ob',mew=0,markersize=4)
ax[0,0].text(pt[-1]+shift,co2[-1],"%.2f" % co2[-1],color='b',ha='left',va='baseline')

ax[0,0].legend(loc=2)
ax[0,0].set_xlim(1955,2015)
ax[0,0].set_ylim(310,405)
ax[0,0].set_xlabel("Year")
ax[0,0].set_ylabel("CO$_{2}$ Concentration [ppm]")
ax[0,0].set_title("Raw data")

xmin,xmax = ax[0,0].get_xlim(); dx = xmax-xmin
ymin,ymax = ax[0,0].get_ylim(); dy = ymax-ymin
ax[0,0].text(xmax-0.05*dx,ymin+0.05*dy,
             "Bias = %+.2f\nRMSE = %.2f" % (il.ComputeNormalizedBias(dco2,co2),
                                            il.ComputeNormalizedRootMeanSquaredError(dco2,co2)),
             ha="right",va="bottom")

# plot annual mean
tmean,  co2mean, co2min, co2max = il.ComputeAnnualMean( t, co2)
dtmean,dco2mean,dco2min,dco2max = il.ComputeAnnualMean(dt,dco2)
pt = tmean/365.+1850
for i in range(pt.shape[0]):
    ax[0,1].fill_between([pt[i]-0.5,pt[i]+0.5],[co2min[i],co2min[i]],[co2max[i],co2max[i]],
                         color='b',alpha=0.25,lw=0)
    ax[0,1].plot([pt[i]-0.5,pt[i]+0.5],[co2mean[i],co2mean[i]],'-b')
    ax[0,1].fill_between([pt[i]-0.5,pt[i]+0.5],[dco2min[i],dco2min[i]],[dco2max[i],dco2max[i]],
                         color='g',alpha=0.25,lw=0)
    ax[0,1].plot([pt[i]-0.5,pt[i]+0.5],[dco2mean[i],dco2mean[i]],'-g')
ax[0,1].plot(pt[-1]+0.5, co2mean[-1],'ob',mew=0,markersize=4)
ax[0,1].text(pt[-1]+shift+0.5, co2mean[-1]," %.2f" %  co2mean[-1],color='b',ha='left',va='baseline')
ax[0,1].plot(pt[-1]+0.5,dco2mean[-1],'og',mew=0,markersize=4)
ax[0,1].text(pt[-1]+shift+0.5,dco2mean[-1]," %.2f" % dco2mean[-1],color='g',ha='left',va='baseline')
ax[0,1].text(xmax-0.05*dx,ymin+0.05*dy,
             "Bias = %+.2f\nRMSE = %.2f" % (il.ComputeNormalizedBias(dco2mean,co2mean),
                                            il.ComputeNormalizedRootMeanSquaredError(dco2mean,co2mean)),
             ha="right",va="bottom")
ax[0,1].set_xlim(ax[0,0].get_xlim())
ax[0,1].set_xlabel("Year")
ax[0,1].set_ylabel("CO$_{2}$ Concentration [ppm]")
ax[0,1].set_title("Annual mean and envelope")
ax[0,1].set_ylim(ax[0,0].get_ylim())

# plot trend with 10 year window
tt , dco2_dt = il.ComputeTrend(dt, co2,window=10.)
dtt,ddco2_dt = il.ComputeTrend(dt,dco2,window=10.)
pt = tt/365.+1850
ax[1,0].plot(pt, dco2_dt,'-b')
ax[1,0].plot(pt,ddco2_dt,'-g')
ax[1,0].plot(pt[-1], dco2_dt[-1],'ob',mew=0,markersize=4)
ax[1,0].text(pt[-1]+shift, dco2_dt[-1]," %.2f" %  dco2_dt[-1],color='b',ha='left',va='baseline')
ax[1,0].plot(pt[-1],ddco2_dt[-1],'og',mew=0,markersize=4)
ax[1,0].text(pt[-1]+shift,ddco2_dt[-1]," %.2f" % ddco2_dt[-1],color='g',ha='left',va='baseline')
ymin,ymax = ax[1,0].get_ylim(); dy = ymax-ymin
ax[1,0].text(xmax-0.05*dx,ymax-0.05*dy,
             "Bias = %+.2f\nRMSE = %.2f" % (il.ComputeNormalizedBias(ddco2_dt,dco2_dt),
                                            il.ComputeNormalizedRootMeanSquaredError(ddco2_dt,dco2_dt)),
             ha="right",va="top")

ax[1,0].set_xlim(ax[0,0].get_xlim())
ax[1,0].set_xlabel("Year")
ax[1,0].set_ylabel("CO$_{2}$ Concentration Trend [ppm/year]")
ax[1,0].set_title("Decadal Trend")




fig.savefig("tmp.pdf")
plt.show()




