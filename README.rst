ILAMB 2.6 Release
=================

It has been a while since our last release, but ILAMB continues to
evolve. Many of the changes are 'under the hood' or bugfixes that are
not readily seen. In the following, we present a few key changes and
draw attention in particular to those that will change scores. We also
have worked to make ILAMB ready to integrate with tools being
developed as part of the Coordinated Model Evaluation Capabilities (`CMEC
<https://cmec.llnl.gov/>`_).

Changes - May 2021
------------------

CMEC
~~~~

* Added CMEC-compliant JSON output to the standard outputs
* Added an alternative landing page for ILAMB results which uses the
  `LMT Unified Dashboard
  <https://github.com/climatemodeling/unified-dashboard>`_
* Added support files for using `cmec-driver
  <https://github.com/cmecmetrics/cmec-driver>`_ as an alternative run
  environment

Quality of Life
~~~~~~~~~~~~~~~

* Top page overhaul moving to a single result panel with a colorblind
  friendly palette
* Shifted score colormaps to be more qualitative and colorblind
  friendly
* ILAMB now has continuous integration testing using Azure Pipelines
  on each commit or pull request
* ModelResults can be passed a list of paths to search for results,
  objects are cached as pickle files
* Plotting limits are now based on the middle 98% across all models to
  help reduce the effect of a single model with extreme values washing
  out all the map plots
* The configure file used to generate a run is now copied into the
  output directory as `ilamb.cfg`
* ILAMB logfiles will now provide an estimate for peak memory usage in
  each confrontation which can be used in debugging and when running
  on large clusters with limited memory

Scoring
~~~~~~~

* For scoring coupled models, we find that scoring the RMSE of the
  annual cycle is more reasonable. While the default is still set to
  score the full time series, this may be changed at runtime with
  `--rmse_score_basis {series|cycle}`
* We have found that when comparing a set of models which contain a
  multimodel mean, the mean model's interannual variability is
  typically lower which serendipitously better matches that of our
  reference data products. This makes the multimodel mean look even
  better relative to individual models but not for good reasons. We
  have disabled the interannual variability in our scoring.
* We have updated a number of reference datasets to their most current
  version as well as many new datasets and comparions, run
  `ilamb-fetch` to update
* Support for using observational uncertainty in scoring, currently
  disabled
  

Useful Information
------------------

* `Documentation <https://www.ilamb.org/doc/>`_ - installation and
  basic usage tutorials
* Sample Output
  
  * `ILAMB <https://www.ilamb.org/CMIP5v6/historical/>`_ - land
    comparison against a collection of CMIP5 and CMIP6 models
  * `IOMB <https://www.ilamb.org/CMIP5v6/IOMB/>`_ - ocean comparison
    against a collection of CMIP5 and CMIP6 models

* `Paper <https://doi.org/10.1029/2018MS001354>`_ published in JAMES
  which details the design and methodology employed in the ILAMB
  package. If you find the package or the output helpful in your
  research or development efforts, we kindly ask you to cite this
  work.

Description
-----------

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

Funding
-------

This research was performed for the *Reducing Uncertainties in
Biogeochemical Interactions through Synthesis and Computation*
(RUBISCO) Scientific Focus Area, which is sponsored by the Regional
and Global Climate Modeling (RGCM) Program in the Climate and
Environmental Sciences Division (CESD) of the Biological and
Environmental Research (BER) Program in the U.S. Department of Energy
Office of Science.
