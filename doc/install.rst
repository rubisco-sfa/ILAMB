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
python packages. Here are a few of your options explained in more
detailed in separate sections below.

1. Install from the `conda-forge` channel of Anaconda python
   (conda_). This is a good option for installing all dependencies
   automatically if you are unfamiliar with how to do so.
2. Install from the Python Package Index (pypi_). While this is the
   canonical method of installing python packages, it will not install
   underlying C-libraries. 

In either method, a good check to ensure ILAMB was installed properly is::

  python -c "import ILAMB; print(ILAMB.__version__)"

which should return a numeric value reflecting the installed
version. Once you have ILAMB installed, you will need to execute::

  ilamb-setup

just once. This will download some `cartopy` assets needed for plotting.

Both methods will install the latest *release* of ILAMB. However, we
continue to develop the software and methodology and you may rather
run from our latest commits. To do this, first use either method above
to install ILAMB. This will get all the dependencies installed. Then
you will clone our repository::

  git clone https://github.com/rubisco-sfa/ILAMB.git
  cd ilamb
  pip install ./

This will overwrite the released version installation with whatever
was last committed. We will now describe in more detail each method.

Method 1: Conda
---------------

The recommended method of installing ILAMB makes use of a package
manager, conda_. Conda goes beyond what python's native `pip` can do
and installs *everything* that the python packages depend on,
including the underlying C-libraries. 

To proceed with this method, you will need to have installed anaconda
or miniconda (see documentation for conda_) or be on an institutional
cluster that has already provided it for you (check for available
modules if the command `conda` is not found). Once installed, you will
need to make sure that the `conda-forge` channel is available::

  conda config --add channels conda-forge

Then you only need to install via::

  conda install ILAMB
  
Conda will then look at all the packages you have installed and try to
solve for what it needs to install/upgrade/downgrade so that all the
software can work together. Note that we support builds for Linux and
OSX, but not Windows.

It may be that `conda` was unable to solve your environment to install
ILAMB. This could happen because, perhaps, you already use `conda` and
have other things installed. What makes `conda` powerful is that it
also allows for the creation of environments. This means that we can
use `conda` to automatically create a special environment for ILAMB
which will not conflict with any other software you may have::

  conda create --name ilamb
  conda activate ilamb
  conda install ilamb
  
If you are running on an institutional cluster that requires you to
submit job scripts to run in parallel, this method may not work for
you. ILAMB achieves parallelism using `mpi4py` which wraps an
installation of the C-library MPI which conda will install for
you. However, on an institutional cluster you may need it to rather wrap
the system's installation of MPI in order to submit jobs via a
queue. In this case, we have provided files in the repository that you
can use to create an environment that will wrap the system's MPI::

  git clone https://github.com/rubisco-sfa/ILAMB.git
  cd ilamb
  conda env create -f ilamb-sysmpi.yml
  pip install ./

Aside on Institutional Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have found that the above may still fail to successfully install
`mpi4py` and therefore also ILAMB. This has to do with the Cray
compile system and is a problem that we have not been able to solve
with the computing staff at these centers. We have found that the
following approach works. First, build an ILAMB environment with
`ilamb.yml` from our repository::

  conda env create -f ilamb.yml

This should succeed, but will install the *wrong* MPI and
`mpi4py`. Next, before activating the new environment, take note of
where the system-installed `mpi4py` is located::

  python -c "import mpi4py as m; print(m.__path__)"

This should return some ugly path where we find two directories::

  /sw/rhea/python/3.7/anaconda3/2018.12/lib/python3.7/site-packages/mpi4py
  /sw/rhea/python/3.7/anaconda3/2018.12/lib/python3.7/site-packages/mpi4py-3.0.2-py3.7.egg-info

Your paths will be a little different, depending on the location that
your system admins have installed this software. For now, just take
note of their location. Now, activate your environment and repeat the
`mpi4py` test::

  conda activate ilamb
  python -c "import mpi4py as m; print(m.__path__)"

which, on my system, results in the following location::

  /ccs/home/nate/.conda/envs/ilamb/lib/python3.7/site-packages/mpi4py

In short, we are going to go into your environment directory, remove
the installed `mpi4py` manually, and then create a symbolic link to
the system version. If this sounds like a hack, you are correct! Just
imagine all the new friends you will make with your new found
skills. So, following my paths, yours will differ, we first navigate
to my environment's `site-packages` directory and remove the current `mpi4py`::

  cd /ccs/home/nate/.conda/envs/ilamb/lib/python3.7/site-packages
  rm -rf mpi4py*

Then, we will link to the system versions. Again, your paths will be
different. Use the ones you noted from above::

  ln -s /sw/rhea/python/3.7/anaconda3/2018.12/lib/python3.7/site-packages/mpi4py
  ln -s /sw/rhea/python/3.7/anaconda3/2018.12/lib/python3.7/site-packages/mpi4py-3.0.2-py3.7.egg-info

Now you can activate your new environment which will use the system
MPI and allow you to submit jobs to make use of multiple
nodes. Finally, install ILAMB using::

  pip install ./
  
Method 2: The Python Package Index
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
automatically. I recommend using a package manager
(yum, apt-get,or dnf on Linux, homebrew on OSX) if you have access
to one. If these underlying C-libraries (MPI, netCDF4, UDUNITS) are
not already installed on your machine and you are unable to get them
installed, you might consider using conda_ explained in method 1.

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
to resolve this problem.

.. _pypi:       https://pypi.python.org/pypi
.. _pip:        https://pip.pypa.io/en/stable/
.. _repository: https://github.com/rubisco-sfa/ILAMB.git
.. _numpy:      https://www.numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _netCDF4:    https://github.com/Unidata/netcdf4-python
.. _cf-units:   https://github.com/SciTools/cf-units
.. _sympy:      https://www.sympy.org/
.. _mpi4py:     https://pythonhosted.org/mpi4py/
.. _github:     https://github.com
.. _conda:      https://conda.io/docs/
