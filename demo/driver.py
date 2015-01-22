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
ax[0,0].plot(pt,dco2,'-k',label='observational data')
ax[0,0].plot(pt[-1],dco2[-1],'ok',mew=0,markersize=4)
ax[0,0].text(pt[-1]+shift,dco2[-1],"%.2f" % dco2[-1],color='k',ha='left',va='baseline')

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
             "Bias = %+.2f\nRMSE = %.2f" % (il.Bias(dco2,co2),
                                            il.RootMeanSquaredError(dco2,co2)),
             ha="right",va="bottom")

# plot annual mean
tmean,  co2mean = il.AnnualMean( t, co2)
dtmean,dco2mean = il.AnnualMean(dt,dco2)
pt = tmean/365.+1850
for i in range(pt.shape[0]):
    #ax[0,1].fill_between([pt[i]-0.5,pt[i]+0.5],[co2min[i],co2min[i]],[co2max[i],co2max[i]],
    #                     color='b',alpha=0.25,lw=0)
    ax[0,1].plot([pt[i]-0.5,pt[i]+0.5],[co2mean[i],co2mean[i]],'-b')
    #ax[0,1].fill_between([pt[i]-0.5,pt[i]+0.5],[dco2min[i],dco2min[i]],[dco2max[i],dco2max[i]],
    #                     color='k',alpha=0.25,lw=0)
    ax[0,1].plot([pt[i]-0.5,pt[i]+0.5],[dco2mean[i],dco2mean[i]],'-k')
ax[0,1].plot(pt[-1]+0.5, co2mean[-1],'ob',mew=0,markersize=4)
ax[0,1].text(pt[-1]+shift+0.5, co2mean[-1]," %.2f" %  co2mean[-1],color='b',ha='left',va='baseline')
ax[0,1].plot(pt[-1]+0.5,dco2mean[-1],'ok',mew=0,markersize=4)
ax[0,1].text(pt[-1]+shift+0.5,dco2mean[-1]," %.2f" % dco2mean[-1],color='k',ha='left',va='baseline')
ax[0,1].text(xmax-0.05*dx,ymin+0.05*dy,
             "Bias = %+.2f\nRMSE = %.2f" % (il.Bias(dco2mean,co2mean),
                                            il.RootMeanSquaredError(dco2mean,co2mean)),
             ha="right",va="bottom")
ax[0,1].set_xlim(ax[0,0].get_xlim())
ax[0,1].set_xlabel("Year")
ax[0,1].set_ylabel("CO$_{2}$ Concentration [ppm]")
ax[0,1].set_title("Annual mean")
ax[0,1].set_ylim(ax[0,0].get_ylim())

# plot trend with 10 year window
dco2_dt  = il.WindowedTrend(dt, co2,window=3650.)
ddco2_dt = il.WindowedTrend(dt,dco2,window=3650.)
pt = dt/365.+1850
ax[1,0].plot(pt, dco2_dt,'-b')
ax[1,0].plot(pt,ddco2_dt,'-k')
ax[1,0].plot(pt[-1], dco2_dt[-1],'ob',mew=0,markersize=4)
ax[1,0].text(pt[-1]+shift, dco2_dt[-1]," %.2f" %  dco2_dt[-1],color='b',ha='left',va='baseline')
ax[1,0].plot(pt[-1],ddco2_dt[-1],'ok',mew=0,markersize=4)
ax[1,0].text(pt[-1]+shift,ddco2_dt[-1]," %.2f" % ddco2_dt[-1],color='k',ha='left',va='baseline')
ax[1,0].set_ylim(1.1)
ymin,ymax = ax[1,0].get_ylim(); dy = ymax-ymin
ax[1,0].text(xmax-0.05*dx,ymin+0.05*dy,
             "Bias = %+.2f\nRMSE = %.2f" % (il.Bias(ddco2_dt,dco2_dt),
                                            il.RootMeanSquaredError(ddco2_dt,dco2_dt)),
             ha="right",va="bottom")
ax[1,0].set_xlim(ax[0,0].get_xlim())
ax[1,0].set_xlabel("Year")
ax[1,0].set_ylabel("CO$_{2}$ Concentration Trend [ppm/year]")
ax[1,0].set_title("Decadal trend")

# plot decadal changes in amplitude
ta, co2amp, co2ampstd = il.DecadalAmplitude(dt,co2)
ta,dco2amp,dco2ampstd = il.DecadalAmplitude(dt,dco2)
pt = ta/365.+1850
pt = np.round(pt/10.)*10.+5.
ax[1,1].errorbar(pt, co2amp,yerr= co2ampstd,color='b',fmt='o',markeredgecolor='b',markersize=4)
ax[1,1].errorbar(pt,dco2amp,yerr=dco2ampstd,color='k',fmt='o',markeredgecolor='k',markersize=4)
ax[1,1].text(pt[-1]+shift, co2amp[-1]," %.2f" %  co2amp[-1],color='b',ha='left',va='baseline')
ax[1,1].text(pt[-1]+shift,dco2amp[-1]," %.2f" % dco2amp[-1],color='k',ha='left',va='baseline')
ymin,ymax = ax[1,1].get_ylim(); dy = ymax-ymin
ax[1,1].text(xmax-0.05*dx,ymin+0.05*dy,
             "Bias = %+.2f (%.2f)\nRMSE = %.2f (%.2f)" % (il.Bias(dco2amp,co2amp),
                                                          il.Bias(dco2ampstd,co2ampstd),
                                                          il.RootMeanSquaredError(dco2amp,co2amp),
                                                          il.RootMeanSquaredError(dco2ampstd,co2ampstd)),
             ha="right",va="bottom")
ax[1,1].set_xlim(ax[0,0].get_xlim())
ax[1,1].set_xlabel("Year")
ax[1,1].set_ylabel("CO$_{2}$ Concentration Amplitude [ppm]")
ax[1,1].set_title("Decadal amplitude shift")

fig.savefig("sample.pdf")
plt.show()




