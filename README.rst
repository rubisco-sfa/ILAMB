ILAMB: A Prototype Model Benchmarking System
============================================

Installation
------------

The project source code is being hosted via a `Git <http://git-scm.com/>`_ repository hosted at `Bitbucket <https://bitbucket.org/ncollier/ilamb>`_. You will need to clone this repository:

  $ git clone https://ncollier@bitbucket.org/ncollier/ilamb.git

Enter the top level directory, next build and install the package using the standard distutils's ``setup.py`` script::

  $ cd ilamb
  $ python setup.py install --user

Note that this project makes use of several other python packages. While many are standard, you may need to install some python packages. I would suggest using `pip <https://pypi.python.org/pypi/pip>`_.

  $ pip install numpy matplotlib basemap netCDF4 mpi4py --user

You will note the ``--user`` flag in both install lines--this installs the packages into a local directory instead of the system directory. I suggest that you always install python packages this way as it keeps your system level python clean. Please send questions/problems about installation to nathaniel.collier@gmail.com.

Next Steps
----------

The software project contains no data. To see if your installation works as intended you will need to download and extract a `sample <http://www.climatemodeling.org/~nate/ILAMB/minimal_ILAMB_data.tgz>`_ dataset. Then you need to set an environment variable to point to this dataset::

  $ export ILAMB_ROOT=PATH_TO_THE_EXTRACTED_DATA

Once you have done this, execute the driver script found in the directory of demos::

  $ cd demos
  $ python driver.py

The driver is setup to use the ILAMB python package to compare the GPP of two models included in the sample data to the Fluxnet Global MTE benchmark dataset. The script produces a directory structure which contains a webpage that summarizes results from the confrontation. To view it, open ``./demo/_build/GPPFluxnetGlobalMTE/GPPFluxnetGlobalMTE.html`` in a web browser or explore the contents of the directory manually.

More Information
----------------

Documentation for this package may be found `here <http://www.climatemodeling.org/~nate/ILAMB/>`_. You will find the complete package API documented as well as a growing list of tutorials which highlight basic usage. 
