ILAMB 2.5 Release
=================

We are pleased to announce a new version of the ILAMB package. In
addition to a new version, the ILAMB repository is now hosted at:

  `https://github.com/rubisco-sfa/ILAMB <https://github.com/rubisco-sfa/ILAMB>`_

This release includes some interface improvements as well as core
technology enhancements increasing speed and reliability. See the
following list for major changes:

* The landing `page <http://www.ilamb.org/CMIP6/historical/>`_
  generated from `ilamb-run` has received an overhaul. We have merged
  the two tabs with the image and data into one dynamic table which
  can be clicked. Clicking on a row header will either expand the row
  or take you to the dataset page. Clicking on a particular model's
  square will take you to the dataset page with that particular model
  highlighted. In addition to this, you can now select which scalar
  you wish to plot in the table (i.e. Overall Score, Bias Score) over
  any region included in the study.
* The appearance of the `Data Information` tabs on the dataset pages
  has been greatly enhanced. References can be included in the netCDF
  files in Bibtex format and will be rendered inside of
  ILAMB. Hyperlinks also will be detected and rendered as clickable
  links in the output pages. Thanks to Mingquan Mu for this addition.
* Added a soil carbon temperature sensitivity metric from Charlie
  Koven, added this to our curated configure file `cmip.cfg`.
* The CO2 emulation code will now account for ocean and fossil fuel
  fluxes when emulating the land model's `nbp`. Thanks to Ke Xu for this
  contribution.
* We have added new datasets `LORA
  <http://dx.doi.org/10.25914/5b612e993d8ea>`_ for runoff and `DOLCE
  <http://dx.doi.org/10.4225/41/58980b55b0495>`_ for latent
  heat. While these datasets include uncertainty estimates, we are
  currently not making use of them in our analysis.
* We have replaced `basemap <https://github.com/matplotlib/basemap>`_
  in favor of `cartopy <https://github.com/SciTools/cartopy>`_ as the
  tool for plotting on maps. Not only is this needed as basemap is
  being deprecated, but plotting is now approximately 10x faster. For
  the most part, this change will be invisible to the user.
* Added options and structure to `ilamb-run` to improve runtimes. If
  running a large set of models on a cluster, we recommend first
  running with the `--skip_plots` option and using a low number of
  processes per node. This is because memory utilization tends to
  dominate the analysis phase and you do not want to run out. Then you
  can submit a second job without `--skip_plots` and using a large
  number of processes per node.
* Intermediate files generated during `ilamb-run` will now include a
  `complete` flag, initialized to `False` and only flagged true if the
  file closed at the end of the analysis phase without error. This
  helps us reinitialize `ilamb-run` when a parallel run crashes and
  leaves file present and not corrupted, but neither complete.
* If the `psutil` python package is installed, `ilamb-run` will now
  log the peak memory being used during the analysis phase in the
  logfiles along with the node name and process rank. This is to help
  in memory debugging for when high resolution models are being run.
* In addition to this, `ilamb-run` now caches the model initialization
  process which should speed up re-initialization for when multiple
  jobs must be submitted.
* Added initial support for uncertainty bounds in the
  ILAMB.Variable. If uncertainty is included in the observations, such
  as in the Hoffman `nbp` dataset, then ILAMB will automatically
  operate of it and show it as a shading in plots without changes to
  your scripts.
* `ilamb-fetch` now will correctly try to decode server SSL
  certificates before downloading files. However, the authority that
  `www.ilamb.org` uses to create its certificates is not in the list
  that python supports. Your browser maintains a different list of
  authorities, which is why you can navigate to sites like `this
  <http://www.ilamb.org/CMIP6/historical/>`_. You will likely need to
  run with the `--no-check-certificate` option which implies that you
  trust that we are who we say we are.

Useful Information
------------------

* `Documentation <https://www.ilamb.org/doc/>`_ - installation and
  basic usage tutorials
* Sample Output
  
  * `CMIP6 <http://www.ilamb.org/CMIP6/historical/>`_ - land comparison against a collection of CMIP6 models
  * `CMIP5 vs CMIP6 <http://www.ilamb.org/CMIP6/historical/>`_ - land comparison against a collection of CMIP5 and CMIP6 models
  * `IOMB <http://www.ilamb.org/IOMB/>`_ - ocean comparison against a few ocean models

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

This research was performed for the *Reducing Uncertainties in Biogeochemical Interactions through Synthesis and Computation* (RUBISCO) Scientific Focus Area, which is sponsored by the Regional and Global Climate Modeling (RGCM) Program in the Climate and Environmental Sciences Division (CESD) of the Biological and Environmental Research (BER) Program in the U.S. Department of Energy Office of Science.
