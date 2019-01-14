The ILAMB Benchmarking System
=============================

The International Land Model Benchmarking (ILAMB) project is a
model-data intercomparison and integration project designed to improve
the performance of land models and, in parallel, improve the design of
new measurement campaigns to reduce uncertainties associated with key
land surface processes. Building upon past model evaluation studies,
the goals of ILAMB are to:

* develop internationally accepted benchmarks for land model
  performance, promote the use of these benchmarks by the
  international community for model intercomparison,
* strengthen linkages between experimental, remote sensing, and
  climate modeling communities in the design of new model tests and
  new measurement programs, and
* support the design and development of a new, open source,
  benchmarking software system for use by the international community.

It is the last of these goals to which this repository is
concerned. We have developed a python-based generic benchmarking
system, initially focused on assessing land model performance.
  
Useful Information
------------------

* `Documentation <https://www.ilamb.org/doc/>`_ - installation and
  basic usage tutorials
* Sample Output
  
  * `CLM <http://www.ilamb.org/CLM/>`_ - land comparison against 3 CLM versions
  * `CMIP5 <http://www.ilamb.org/CMIP5/esmHistorical/>`_ - land comparison against a collection of CMIP5 models
  * `IOMB <http://www.ilamb.org/IOMB/>`_ - ocean comparison against a few ocean models

* `Paper <https://doi.org/10.1029/2018MS001354>`_ published in JAMES
  which details the design and methodology employed in the ILAMB
  package. If you find the package or the ouput helpful in your
  research or development efforts, we kindly ask you to cite this
  work.

ILAMB 2.4 Release
-----------------

This release marks an important technical shift in ILAMB
development--ILAMB v2.4 and onward will be python3 only. If you are
new to python, it might seem strange that python3 has been released
for 10 years and yet python2 is still ubiquitous. There is now an
official `announcement <https://pythonclock.org/>`_ that python2 will
reach its end of life at the end of 2019. Furthermore there is a
growing `list <https://python3statement.org/>`_ of popular python
packages (most of our dependencies) that are phasing out support for
python2 during this year. So in keeping with this community trend, the
last version of ILAMB which will be compatible with python2.7x is
2.3.1, version 2.4 and beyond will by python3 only.

Part of my reason for sticking with python2 for so long was that ILAMB
was designed to run on large machines whose software stacks are often
not frequently updated. I wanted to ensure that ILAMB would run on old
software. However, this is less an issue as computing centers are
moving away from providing users with python packages they load via
center-supported environment modules and towards users creating
personalized environments using `conda
<https://conda.io/docs/>`_. Look for the ``ilamb.yml`` file in the
repository which conda can use to create an environment that will
support an ILAMB installation. If these words do not mean anything to
you, look for a more detailed explanation in the `tutorials
<https://www.ilamb.org/doc/install.html>`_ which have be rewritten to
reflect this shift.

We have published a `paper <https://doi.org/10.1029/2018MS001354>`_ in
JAMES which details the methodology which this package implements. If
you find ILAMB helpful in your research, we would appreciate a
citation to this work as it helps us communicate the impact that these
investments have on the community.

The collection of land surface confrontations now includes the
emulation of CO\ :sub:`2`\ fluxes. The default setup is to compare
``nbp`` fluxes to those recorded at a subset of NOAA sites, but this
is configurable from inside the configure file. Browse the CMIP5
`output
<https://www.ilamb.org/CMIP5/esmHistorical/EcosystemandCarbonCycle/CarbonDioxide/NOAA.Emulated/NOAA.Emulated.html>`_
for an overview of what this addition provides.

Finally, we are making some shifts in how we support ILAMB. Until now,
I have directed user questions to my personal email. This is still ok,
however consider `joining
<https://www.ilamb.org/mailman/listinfo/ilamb-users>`_ the ILAMB
mailing list and sending your questions there. Not only does this open
up your question to being answered more quickly by the community, but
the answers are searchable which may help the next user. In addition
to this, we have a Slack `channel
<https://ilamb-community.slack.com/>`_ if you prefer to ask your
questions there. This has more of a chat interface but the
conversations are all still public and searchable by the members.

Funding
-------

This research was performed for the *Reducing Uncertainties in Biogeochemical Interactions through Synthesis and Computation* (RUBISCO) Scientific Focus Area, which is sponsored by the Regional and Global Climate Modeling (RGCM) Program in the Climate and Environmental Sciences Division (CESD) of the Biological and Environmental Research (BER) Program in the U.S. Department of Energy Office of Science.
