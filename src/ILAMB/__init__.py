__author__       = 'Nathan Collier'
__date__         = 'May 2016'
__version__      = '2.0.1'

from distutils.version import LooseVersion
import platform

# These are guesses at actual requirements
requires = {
    "numpy"                : "1.9.2",
    "matplotlib"           : "1.4.3",
    "netCDF4"              : "1.1.4",
    "cfunits"              : "1.1.4",
    "mpl_toolkits.basemap" : "1.0.7",
    "sympy"                : "0.7.6"
}

froms = {
    "mpl_toolkits.basemap" : "Basemap"
}

for key in requires.keys():
    if "." in key:
        pkg = __import__(key, globals(), locals(), [froms[key]])
    else:
        pkg = __import__(key)
    if LooseVersion(pkg.__version__) < LooseVersion(requires[key]):
        raise ImportError(
            "Bad %s version: ILAMB %s requires %s >= %s got %s" %
            (key,__version__,key,requires[key],pkg.__version__))




