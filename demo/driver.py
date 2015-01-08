from ILAMB.ModelResult import ModelResult
from ILAMB.Confrontation import Confrontation
import pylab as plt

# initialize model and confrontation
M = ModelResult  ("./data")
C = Confrontation("./data")

# plot results
fig,ax = plt.subplots(figsize=(12,4),tight_layout=True)

t,co2 = M.extractPointTimeSeries("co2",C.lat,C.lon)
ax.plot(t/365.+1850,co2,'-r',label='entire model result')

t,co2 = C.extractModelResult(M)
ax.plot(C.t/365.+1850,C.var,'-g',label='observational data')
ax.plot(t/365.+1850,co2,'-k',alpha=0.5,label='intersection of model and obs')
ax.legend(loc=2)

print "NRMSE:",C.computeNormalizedRootMeanSquaredError(M)             # 238 ms per loop
print "NRMSE:",C.computeNormalizedRootMeanSquaredError(M,t=t,var=co2) # 720 us per loop
print "Nbias:",C.computeNormalizedBias(M,t=t,var=co2)

plt.show()


