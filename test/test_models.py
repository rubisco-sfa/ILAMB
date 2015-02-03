from ILAMB.ModelResult import ModelResult
from ILAMB import ilamblib as il
from ILAMB.constants import convert
import pylab as plt
import numpy as np
import os

# Initialize the models
M = []
root = "./data/"
for subdir, dirs, files in os.walk(root):

    # If not a historical simulation, then skip
    if "esmHistorical" not in subdir: continue
    
    # Parse the model name from its directory and choose a unique color
    mname = subdir.replace(root,"").replace("esmHistorical","").replace("/","").upper()
    
    # Initialize the model result, only use files with a "r1i1p1" in the filename
    M.append(ModelResult(subdir,modelname=mname,filter="r1i1p1"))

print "\nFirst try\n"

# First pass, just checking for the variable name "co2"
for m in M:
    print "Looking in %s for CO2 data..." % m.name
    try:
        t,var,unit = m.extractPointTimeSeries("co2",19.4,24.4)
    except il.VarNotInModel,e:
        print "\t%s" % e
        continue
    print "\tFound co2 in units of [%s]" % unit

print "\nSecond try\n"

# Let's try again, but now we specify an alternate variable: co2mass
for m in M:
    print "Looking in %s for CO2 data..." % m.name
    try:
        t,var,unit = m.extractPointTimeSeries("co2",19.4,24.4,alt_vars=["co2mass"])
    except il.VarNotInModel,e:
        print "\t%s" % e
        continue
    print "\tFound co2 in units of [%s]" % unit

print "\nThird try\n"

# That's great, but what if we had wanted the unit converted
for m in M:
    print "Looking in %s for CO2 data..." % m.name
    try:
        t,var,unit = m.extractPointTimeSeries("co2",19.4,24.4,alt_vars=["co2mass"],output_unit="1e-6")
    except il.VarNotInModel,e:
        print "\t%s" % e
        continue
    print "\tFound co2 in units of [%s]" % unit

