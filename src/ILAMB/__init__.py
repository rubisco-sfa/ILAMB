__author__       = 'Nathan Collier'
__date__         = 'Sep 2019'
__version__      = '2.5'

from distutils.version import LooseVersion
import platform

# These are guesses at actual requirements
requires = {
    "numpy"                : "1.9.2",
    "matplotlib"           : "1.4.3",
    "netCDF4"              : "1.1.4",
    "cf_units"             : "2.0.0",
    "cartopy"              : "0.17.0",
    "sympy"                : "0.7.6",
    "mpi4py"               : "1.3.1"
}

for key in requires.keys():
    pkg = __import__(key)
    if LooseVersion(pkg.__version__) < LooseVersion(requires[key]):
        raise ImportError(
            "Bad %s version: ILAMB %s requires %s >= %s got %s" %
            (key,__version__,key,requires[key],pkg.__version__))
