from ILAMB.Variable import Variable
import ILAMB.ilamblib as il
import numpy as np
import os

def test_integrateInTime(variables):
    head = "\n--- Testing integrateInTime() "
    print "%s%s\n" % (head,"-"*(120-len(head)))
    for vdict in variables:
        var = vdict["var"]
        try:
            vdict["timeint"]      = var.integrateInTime()
            vdict["timeint_mean"] = var.integrateInTime(mean=True)
            print vdict["timeint"]
            print vdict["timeint_mean"]
        except il.NotTemporalVariable:
            pass

def test_integrateInSpace(variables):
    head = "\n--- Testing integrateInSpace() "
    print "%s%s\n" % (head,"-"*(120-len(head)))
    for vdict in variables:
        var = vdict["var"]
        try:
            vdict["spaceint"]      = var.integrateInSpace()
            vdict["spaceint_mean"] = var.integrateInSpace(mean=True)
            vdict["spaceint_amazon"]      = var.integrateInSpace(region="amazon")
            vdict["spaceint_amazon_mean"] = var.integrateInSpace(region="amazon",mean=True)
            print vdict["spaceint"]
            print vdict["spaceint_mean"]
            print vdict["spaceint_amazon"]
            print vdict["spaceint_amazon_mean"]
        except il.NotSpatialVariable:
            pass

def test_annualCycle(variables):
    head = "\n--- Testing annualCycle() "
    print "%s%s\n" % (head,"-"*(120-len(head)))
    for vdict in variables:
        var = vdict["var"]
        try:
            # note: not testing std, max, and min (assuming ok since functions are similar)
            vdict["cycle"],junk,junk,junk = var.annualCycle()
            print vdict["cycle"]
        except il.NotTemporalVariable:
            pass


def test_correlation(variables):
    head = "\n--- Testing correlation() "
    print "%s%s\n" % (head,"-"*(120-len(head)))
    for vdict in variables:
        var = vdict["var"]
        try:
            if var.spatial or var.ndata:
                vdict["corr_spatial"]  = var.correlation(var,"spatial")
                print vdict["corr_spatial"]
            if var.temporal:
                vdict["corr_temporal"] = var.correlation(var,"temporal")
                print vdict["corr_temporal"]
            if var.spatial and var.temporal:
                vdict["corr_both"] = var.correlation(var,"spatiotemporal")
                print vdict["corr_both"]

        except il.NotTemporalVariable:
            pass
        
def test_bias(variables):
    head = "\n--- Testing bias() "
    print "%s%s\n" % (head,"-"*(120-len(head)))
    for vdict in variables:
        var = vdict["var"]
        try:
            vdict["bias"] = var.bias(var)
            print vdict["bias"]
        except il.NotSpatialVariable:
            pass
        
# Setup different types of variables
gpp = {}
gpp["var"] = Variable(filename = os.environ["ILAMB_ROOT"]+"/DATA/gpp/FLUXNET-MTE/derived/gpp.nc",
                      variable_name = "gpp")
le = {}
le["var"]  = Variable(filename = os.environ["ILAMB_ROOT"]+"/DATA/le/FLUXNET/derived/le.nc",
                      variable_name = "le")
co2 = {}
co2["var"] = Variable(filename = os.environ["ILAMB_ROOT"]+"/DATA/co2/MAUNA.LOA/derived/co2_1959-2013.nc",
                      variable_name = "co2")    
pi = {}
pi["var"]  = Variable(data = np.pi,
                      unit = "-",
                      name = "pi")

variables = [gpp,le,co2,pi]

head = "\n--- Found the following variables for testing "
print "%s%s\n" % (head,"-"*(120-len(head)))
for vdict in variables:
    print vdict["var"]

test_integrateInTime(variables)
test_integrateInSpace(variables)
test_annualCycle(variables)
test_timeOfExtrema(variables)
test_interpolate(variables)
test_phaseShift(variables)
test_correlation(variables)
test_bias(variables)
    
