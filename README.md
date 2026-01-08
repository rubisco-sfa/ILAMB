[<img width=250px src=https://www.ilamb.org/assets/images/RUBISCO1.png>](https://www.bgc-feedbacks.org/)

# ILAMB - International Land Model Benchmarking

The python package designed to help confront earth system models with reference data products, and then present the results in a hierarchical set of webpages. Please see [ilamb.org](https://www.ilamb.org) where we have details about the datasets we use, the results we catalog, and the methods we employ.

## v2.7.3 Release - January 2025

This update contains many bugfixes found while supporting papers and other model-data comparison work in the community. You may sense that the development of ILAMB has slowed down. Code development at *this* repository has slowed, but ILAMB continues to change. Many datasets are being added to our comparisons but that activity happens in a different repository (see [ILAMB-Data](https://github.com/rubisco-sfa/ILAMB-Data)). 

We have also been developing a rewrite of the ILAMB software which leverages more modern analysis packages, in particular xarray. The new ILAMB version also is meant to address a shift in how scientists work. Originally ILAMB was envisioned as a replacement for diagnostics packages that modeling centers run. However in the past 10 years, it has become more common for scientists to run small analyses in notebooks as part of their model development process. This new version of ILAMB will find a balance between the operational diagnostic package and the ability to import functionality for creative uses. At the moment, we are developing this as a separate package we call [ilamb3](https://github.com/rubisco-sfa/ilamb3). Check it out along with its accompanying [documentation](https://ilamb3.readthedocs.io/en/latest/).

## Funding

This research was performed for the *Reducing Uncertainties in
Biogeochemical Interactions through Synthesis and Computation*
(RUBISCO) Scientific Focus Area, which is sponsored by the Regional
and Global Climate Modeling (RGCM) Program in the Climate and
Environmental Sciences Division (CESD) of the Biological and
Environmental Research (BER) Program in the U.S. Department of Energy
Office of Science.
