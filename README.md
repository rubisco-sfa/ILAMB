[<img width=250px src=https://www.ilamb.org/assets/images/RUBISCO1.png>](https://www.bgc-feedbacks.org/)

# ILAMB - International Land Model Benchmarking

The python package designed to help confront earth system models with reference data products, and then present the results in a hierarchical set of webpages. Please see [ilamb.org](https://www.ilamb.org) where we have details about the datasets we use, the results we catalog, and the methods we employ.

## v2.7 Release - June 2023

* Release of the International Ocean Model Benchmarking (IOMB) configuration. For more details see this [post](https://www.ilamb.org/2023/06/24/IOMB-Release.html).
* Assets used in a virtual hackathon for watershed analysis, organized by the ESS Cyberinfrastructure Model-Data Integration Working Group. For more details, see this [post](https://www.ilamb.org/2023/04/27/Watersheds.html) or these hosted [results](https://www.ilamb.org/~nate/ILAMB-Watersheds/). This includes capabilities to read raw E3SM output, even over smaller regions as well as point models.
* We have implemented an alternative scoring methodology for bias and RMSE which is based on regional quantiles of error across a selection of CMIP5 and CMIP6 models. The main idea is to normalize errors by a regional value which constitutes a poor error with respect to what earth system models have produce in the last generations of models. To use, set the `--df_errs database.parquet` option in `ilamb-run` and point to this pandas [database](https://github.com/rubisco-sfa/ILAMB/blob/master/src/ILAMB/data/quantiles_Whittaker_cmip5v6.parquet) or create your own. This interactive [plot](https://www.climatemodeling.org/~nate/score_comparison_CMIP.html) shows how this changes our understanding of performance of CMIP5 to CMIP6. In the future this will become the default scoring methodology.
* Added many [datasets](https://www.ilamb.org/datasets.html) for use in ILAMB, also available via our [intake](https://github.com/nocollier/intake-ilamb) catalog. These include:
  * Biological Nitrogen Fixation from Davies-Barnard
  * Gross Primary Productivity, Sensible and Latent Heat from WECANN (Water, Energy, and Carbon with Artificial Neural Networks)
  * Latent, Sensible, and Ground Heat Flux, Runoff, Precipitation and Net Radiation from CLASS (Conserving Land-Atmosphere Synthesis Suite)
  * Biomass from ESACCI (European Space Agency, Biomass Climate Change Initiative) and XuSaatchi2021
  * Surface Soil Moisture from WangMao
  * Growing Season Net Carbon Flux from Loechli
  * Methane from Fluxnet
* In particular for biomass comparisons, where the measured quantity can be in total or carbon units and represent the above ground portion or total biomass, we have added a `scale_factor` to the configure language which can apply any factor to the reference data.
* ILAMB regions may now also be defined by shapefile, of particular use when comparing results over watersheds.
* Many bugfixes and visual tweaks.

## Funding

This research was performed for the *Reducing Uncertainties in
Biogeochemical Interactions through Synthesis and Computation*
(RUBISCO) Scientific Focus Area, which is sponsored by the Regional
and Global Climate Modeling (RGCM) Program in the Climate and
Environmental Sciences Division (CESD) of the Biological and
Environmental Research (BER) Program in the U.S. Department of Energy
Office of Science.
