ILAMB: A Prototype Model Benchmarking System
============================================

Installation
------------

The project source code is being hosted via a `Git <http://git-scm.com/>`_:: repository hosted at `Bitbucket <https://bitbucket.org/ncollier/ilamb>`_::. You will need to clone this repository:

  $ git clone https://ncollier@bitbucket.org/ncollier/ilamb.git

Enter the top level directory, next build and install the package using the standard distutils's ``setup.py`` script::

  $ cd ilamb
  $ python setup.py install --user

Note that this project makes use of several other python packages. While many are standard, you may need to install some python packages. I would suggest using `pip <https://pypi.python.org/pypi/pip>`_::.

  $ pip install numpy matplotlib matplotlib-basemap scipy netCDF4 h5py

Please send questions/problems about installation to nathaniel.collier@gmail.com.

Getting Data
------------

The software project contains no data. I have written it to use data hosted at UCI by Mingquan Mu. With his permission, I will add details here about how to access and download this data.

Documentation for the project is currently hosted `here <http://www.climatemodeling.org/~nate/ILAMB/>_`::.
