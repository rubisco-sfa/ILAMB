__author__       = 'Nathan Collier'
__date__         = 'March 2016'
__version__      = '0.1'

from distutils.version import LooseVersion
import platform

# These are guesses at actual requirements
requires = {
    "numpy"                : "1.10",
    "matplotlib"           : "1.5",
    "netCDF4"              : "1.1.4",
    "mpi4py"               : "1.3.1",
    "cfunits"              : "1.1.4",
    "mpl_toolkits.basemap" : "1.0.7"
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




