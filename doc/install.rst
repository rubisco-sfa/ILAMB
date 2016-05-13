Installation
============

The ILAMB benchmarking software is written in python and depends on a
few packages which extend the language's usefulness in scientific
applications. While a relatively straightforward process, there are a
number of factors which can make installing python packages
confusing. Your system may have several versions of python
simultaneously installed. While not a problem itself, it can be a
source of confusion. You can easily install a package into one
version's list of packages and then discover that it is not available
because you are actually using a second version of python.

If you are brand new to python, I recommend using a distribution of
python which targets scientific applications. The Anaconda_
distribution will come with many commonly used packages pre-installed
which circumvents the need to install many things. If you find then
that you need further packages, the distribution comes with a package
manager of its own called conda_. Packages may be installed by the syntax::

  conda install name_of_python_package

See their website for further details.

If you do not wish to use Anaconda, then you can install python
packages with pip_ in much the same way. See their website for details
on how to install and use this package manager. The syntax will be similar::

  pip install name_of_python_package --user

Note that I have post-pended a ``--user`` flag to the command. This is
not strictly necessary yet recommended as it will cause the package to
be installed to a local directory in place of the system
directory. This, for example, will allow packages to be installed
without administrator privileges. It also leaves your system
installation untouched, which may be important if you need to revert
to a previous state for some reason.

Install from source
-------------------

If a package is not available in conda_ or pip_ then you may need to
install it from source. This is the case with the ILAMB package itself
as we do not yet have it listed in either package manager. To do this,
first locate the package's source repository. For the ILAMB package,
this amounts to cloning the repository_ found on bitbucket::

  git clone https://bitbucket.org/ncollier/ilamb.git
  
Enter into the cloned repository and you will find a python file named
``setup.py``. This is a common file in all python packages. Package
developers use it to put information about how their package is
compiled and which source files to include. At this point simply type::

  python setup.py install --user

As before, the ``--user`` flag is a recommendation. To check if the
package was successfully installed, type the following::

  python -c "import ILAMB; print ILAMB.__version__"

If you get a numerical output, then the ILAMB package was successfully
installed. If you see something like::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ImportError: No module named ILAMB

Then the package is not correctly installed and you need to look at
the screen output from the install to see what went wrong. You may
also have observed an import error of a different sort. When you
import the ILAMB package, we check the version of all the packages on
which we depend. You will see error text like the following::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/local/lib/python2.7/site-packages/ILAMB/__init__.py", line 29, in <module>
      (key,__version__,key,requires[key],pkg.__version__))
  ImportError: Bad numpy version: ILAMB 0.1 requires numpy >= 1.9.2 got 1.7

This means that while the ``numpy`` package is installed on your
system, its version is too old and you need to use conda_ or pip_ to
upgrade it to at least the version listed. You may also see a message
like the following::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/local/lib/python2.7/site-packages/ILAMB/__init__.py", line 25, in <module>
      pkg = __import__(key)
  ImportError: No module named numpy

This means that we require the ``numpy`` module but you do not have it
installed at all. Again, use conda_ or pip_ to resolve this
problem.

Dependencies
------------

In order to use the ILAMB python package, you will need to install the
following packages and their dependencies in some form:

* numpy_, the fundamental package for scientific computing with python
* matplotlib_, a 2D plotting library which produces publication quality figures
* netCDF4_, a python/numpy interface to the netCDF C library
* cfunits_, a python interface to UNIDATA’s Udunits-2 library with CF extensions
* basemap_, a matplotlib toolkit which is a library for plotting 2D data on maps
* sympy_, a python library for symbolic mathematics
* mpi4py_, a python wrapper around the MPI library

While the ILAMB package itself is not built to exploit parallelism,
the application we have written using the ILAMB package to perform
model-data comparisions does a basic map-reduce utilizing. Thus while
mpi4py_ is not a strict dependency, you should also install it as
well.

If you are planning to run the ILAMB package on an institutional
computing resource, many of our dependencies might be preinstalled but
not enabled by default. You might check first if your system
administrator has installed these. You can consult the user guide for
your particular resource, or just try::

  module avail numpy

for example. If your system uses environment modules, this should list
any available preinstalled versions of numpy or other python package
for which you search.

.. _Anaconda:   https://www.continuum.io/why-anaconda
.. _conda:      http://conda.pydata.org/docs/
.. _pip:        https://pip.pypa.io/en/stable/
.. _repository: https://bitbucket.org/ncollier/ilamb
.. _numpy:      http://www.numpy.org/
.. _matplotlib: http://matplotlib.org/
.. _netCDF4:    https://github.com/Unidata/netcdf4-python
.. _cfunits:    http://pythonhosted.org/cfunits/
.. _basemap:    http://matplotlib.org/basemap/
.. _sympy:      http://www.sympy.org/
.. _mpi4py:     http://pythonhosted.org/mpi4py/
