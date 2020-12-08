Running the ocean (IOMB) configuration
======================================

While the ILAMB software package was originally envisioned for
comparisons involving the land domain of global earth system models,
the software is written generically to facilitate any model-reference
comparisons. The key is the input argument to ``ilamb-run`` which
specifies which configuration file to use ``--config``. As covered in
previous tutorials, this option allows you to use the ILAMB
methodology and software to setup comparisons to any reference source
you may desire.

However, we also curate a collection of reference datasets and a
configuration file for comparisons to output from ocean models. As
part of the installation process, ILAMB will place this and other
supported configuration files in the installed location of the ILAMB
software. In order to find where this file is located, try to
execute::

  find `python -c 'import site; print(site.getsitepackages()[0])'` -name "iomb.cfg"

This command uses the location in which python looks for packages to
start a search for the file ``iomb.cfg``. We recommend copying this
file to a local location for convenience or creating a symbolic
link. Note that we make updates to this file periodically and so to
keep updated, you will need to either create a symbolic link or
remember to periodically copy the configure file to your run
location. If this is confusing for you, you can always clone the ILAMB
repository and look in the source to locate the IOMB configuration
`file <https://github.com/rubisco-sfa/ILAMB/blob/master/src/ILAMB/data/iomb.cfg>`_.
  
The reference observational data can be obtained by using
`ilamb-fetch <./ilamb_fetch.html>`_ and specificying the ``--remote_root``::

  ilamb-fetch --remote_root https://www.ilamb.org/IOMB-Data

Note that the ``ilamb-fetch`` command requires that ILAMB package is
`installed <./install.html>`_. Once you have the reference data
downloaded and the ``iomb.cfg`` file, you may run the IOMB comparisons
against model data you have or have downloaded from locations such as
the Earth System Grid Federation (`ESGF <https://esgf-node.llnl.gov/search/cmip6/>`_).
