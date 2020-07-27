Installation
============

The ILAMB benchmarking software is written in python 3 and depends on
a few packages which extend the language's usefulness in scientific
applications. This tutorial is meant to guide you in installing these
dependencies. We acknowledge that installing software is a huge
barrier for many scientists, and so we aim to not only give basic
instructions that should work, but also teach the user what they are
doing and how to interpret things when they go wrong.

To make matters more complicated, there are many ways to install
ILAMB. Here are a few of your options explained in more detailed in
separate sections below.

1. Install from the Python Package Index (pypi_). This is good if you
   only need to run a released version of ILAMB.
2. Install by cloning the git repository and then performing a local
   install with pip_. This is good if you need features that are
   currently being developed and not yet in a major release.
3. Install by creating a virtual environment using conda_. This is a
   good option for installing dependencies if you are intending to run
   on an institutional machine where you lack permissions to install
   dependencies.

Method 1: The Python Package Index
----------------------------------

The canonical method of installing python packages is via the Python
Package Index (pypi_). Developers can choose to list their projects
here for the world to discover using the pip_ utility which also
automatically installs the dependencies. To install ILAMB using pip_
you type::

  pip install ILAMB --user

at the commandline and pip_ will install most everything
automatically. Please note that I have post-pended a ``--user`` flag
to the command. This is not strictly necessary yet recommended as it
will cause the packages to be installed to a *local* directory in
place of the *system* directory. This allows packages to be installed
without administrator privileges, and leaves your system installation
untouched, which may be important if you need to revert to a previous
state.

You should see that a number of packages in addition to ILAMB
had their versions checked or were upgraded/installed as needed. These
include:

* numpy_, the fundamental package for scientific computing with python
* matplotlib_, a 2D plotting library which produces publication quality figures
* sympy_, a python library for symbolic mathematics
* netCDF4_, a python/numpy interface to the netCDF C library (you must have the C library installed)
* mpi4py_, a python wrapper around the MPI library (you must have a MPI implementation installed)
* cf-units_, a python interface to UNIDATA’s Udunits-2 library with CF extensions (you must have the Udunits library installed)

I have designated that a few of these dependencies are python
interfaces to C libraries and so the library must also be installed
separately. See the individual package websites for more
details. Ideally, pip_ would be able to install all our dependencies
automatically. If these underlying C-libraries (MPI, netCDF4, UDUNITS)
are not already installed on your machine and you are unable to get
them installed, you might consider using conda_ and explained in
method 3.

Unfortunately, one of our dependencies must be installed
manually. Despite being listed in the Python Package Index, basemap_
cannot be installed with pip_. The meta information is listed, but the
package source is too large to be hosted and so installation fails. We
will need to install basemap_ from the source hosted on github_. This
is a useful process to understand as any python package can be
installed in this way. First, clone the git repository::

  git clone https://github.com/matplotlib/basemap.git

This will take some time as the repository is large (>100Mb) due to it
including some high resolution map data used in plotting. Enter into
the cloned directory and take note of a file called ``setup.py``. All
python packages will contain a file called ``setup.py`` in the top
level directory. This is where a developer tells python how to install
the package. Now we type::

  pip install ./ --user

and the package should install. (NOTE: for those of you accustomed to
using ``python setup.py install --user``, this method is no longer
recommended and instead we let pip_ install the locally downloaded
package. This is because pip installs additional metadata about which
files were installed, making uninstalling the package possible. In
recent versions of pip, removal of a package not installed by pip is
not allowed.)

You can test your installation by the following command::
  
  python -c "import ILAMB; print(ILAMB.__version__)"

If you get a numerical output, then the package has been successfully
installed.

Method 2: Clone the Source
--------------------------

The main difference in this method is that we will be installing the
source code from the cloned repository instead of a tagged version
listed in the PyPI. If you are involved in developing features or
needing bleeding edge developments, this is the method of installation
you will need. First, clone the git repository::

  git clone https://github.com/rubisco-sfa/ILAMB.git

and then navigate into the newly created directory. Then type::

  pip install ./ --user

after which, you may follow instructions as listed in Method 1.  
  
Method 3: Conda Environments
----------------------------

The last method we will describe makes use of a different package
manager, conda_. Conda goes beyond what pip can do and installs
*everything* that the python packages depend on, including the
underlying C-libraries.

What makes this more powerful is that conda_ also allows for the
creation of environments. This means that we can use conda to
automatically create a special environment for ILAMB which will not
conflict with any other software you may have.

To proceed with this method, you will need to have installed anaconda
or miniconda (see documentation for conda_). If you are wanting to run
on an institutional machine, this might be installed for you
already. Look among the possible modules which the computing center
provides. For example, at the time of this writing, on OLCF/Rhea you
would need to type::

  module load python_anaconda3

Once conda is properly installed/loaded, then clone the ILAMB
repository and enter the directory::

  git clone https://github.com/rubisco-sfa/ILAMB.git
  cd ilamb

There are two files in this directory with the ``.yml`` suffix. These
are the files that conda_ will use to create the environment. To do
so, type::

  conda env create -f ilamb.yml

You may notice that there are two ``.yml`` files. The ``ilamb.yml``
file will create an environment by adding a channel called
``conda-forge`` and then installing packages listed there. This
includes ``openmpi`` with ``mpi4py``. This is the file you should use
if you are running on your own personal machine and wish to create an
environment for running ILAMB. You may, however, wish that ``mpi4py``
wrap the system installed MPI implementation instead. For this you
should rather create the environment by::

  conda env create -f ilamb-sysmpi.yml
  
You want to do this if you are on an institutional machine where jobs
are queued. The MPI implementation on these machines is specially
configured and thus you want to wrap ``mpi4py`` around this. Once the
environment is built, you will need to activate it::

  conda activate ilamb

Once inside this environment, we can install ILAMB easily using pip::

  pip install ./

In this case, we omit the ``--user`` as pip in this environment is
automatically configured to install the packages into the environment
itself. As in the other methods, run::

  python -c "import ILAMB; print(ILAMB.__version__)"
  
to ensure correct installation. Note that each time you wish to use
ILAMB, you will need to activate this environment including inside
submission scripts on institutional machines.

Now what?
---------

If you got the installation to work, then you should proceed to
working on the next tutorial. Before leaving this page, there are a
few extra steps we recommend you perform. If you installed ILAMB using
the ``--user`` option, the executeable script ``ilamb-run`` will be
placed inside ``${HOME}/.local/bin``. You may need to postpend this
location to your ``PATH`` environment variable::

  export PATH=${PATH}:${HOME}/.local/bin

assuming you are using a ``bash`` environment. This will make the
``ilamb-run`` script executeable from any directory. Also, if you are
connecting to a machine remotely in order to run ILAMB, you may wish
to change the matplotlib_ backend to something that does not generate
interactive graphics::

  export MPLBACKEND=Agg

This will allow ILAMB to run without needing to connect with the
``-X`` option.
  
What can go wrong?
------------------

In an ideal world, this will work just as I have typed it to
you. However, if you are here, something has happened and you need
help. Installing software is frequently all about making sure things
get put in the correct place. You may be unaware of it, but you may
have several versions of python floating around your machine. The pip_
software we used to install packages needs to match the version of
python that we are using. Try typing::

  pip --version
  which python
  python --version

where you should see something like::

  pip 9.0.1 from /usr/local/lib/python2.7/site-packages (python 2.7)
  /usr/local/bin/python
  Python 2.7.13
  
Notice that in my case the pip_ I am using matches the version and
location of the python. This is important as pip_ will install
packages into the locations which my python will find. If your pip_
is, say, for python 3 but you are using python 2.7 then you will
install packages successfully, but they will seem to not be available
to you. The same thing can happen if you have the right version of
python, but it is installed in some other location.

Now we provide some interpretation of the possible output you got from
the test. If you ran::

  python -c "import ILAMB; print(ILAMB.__version__)"

and you see something like::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ImportError: No module named ILAMB

Then the package did not correctly install and you need to look at the
screen output from the install process to see what went wrong. You may
also have observed an import error of a different sort. When you
import the ILAMB package, we check the version of all the packages on
which we depend. You could see an error text like the following::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/local/lib/python2.7/site-packages/ILAMB/__init__.py", line 29, in <module>
      (key,__version__,key,requires[key],pkg.__version__))
  ImportError: Bad numpy version: ILAMB 0.1 requires numpy >= 1.9.2 got 1.7

This means that while the ``numpy`` package is installed on your
system, its version is too old and you need to use pip_ to upgrade it
to at least the version listed. You may also see a message like the
following::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/local/lib/python2.7/site-packages/ILAMB/__init__.py", line 25, in <module>
      pkg = __import__(key)
  ImportError: No module named numpy

This means that we require the ``numpy`` package but you do not have
it installed at all. This should not happen, but if it does, use pip_
to resolve this problem. It is possible that despite a seemingly
smooth installation of basemap_, ILAMB complains about there not being
a module called basemap::

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/local/lib/python2.7/site-packages/ILAMB/__init__.py", line 24, in <module>
      pkg = __import__(key, globals(), locals(), [froms[key]])
  ImportError: No module named basemap

Basemap is a little trickier than other python packages because it is
a *plugin* to the maplotlib package. My recommendation if you are
seeing this message is to install matplotlib in a local location and
upgrade it to the most up to date version::

  pip install matplotlib --user --upgrade

and then install basemap also using the ``--user`` option. This should
ensure that matplotlib toolkits find the basemap extension.


.. _pypi:       https://pypi.python.org/pypi
.. _pip:        https://pip.pypa.io/en/stable/
.. _repository: https://github.com/rubisco-sfa/ILAMB.git
.. _numpy:      https://www.numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _netCDF4:    https://github.com/Unidata/netcdf4-python
.. _cf-units:   https://github.com/SciTools/cf-units
.. _basemap:    https://github.com/matplotlib/basemap
.. _sympy:      https://www.sympy.org/
.. _mpi4py:     https://pythonhosted.org/mpi4py/
.. _github:     https://github.com
.. _conda:      https://conda.io/docs/
