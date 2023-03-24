"""Implements the growing season net flux (GSNF) metric described in this
[paper](https://doi.org/10.1029/2022GB007520). It requires the observational
HIPTOM data encoded [here](https://github.com/rubisco-sfa/ILAMB-Data)"""
import os
from copy import deepcopy
from functools import partial

import numpy as np
from .Confrontation import Confrontation
from .Variable import Variable
from netCDF4 import Dataset


class ConfGSNF(Confrontation):
    """Implements the growing season net flux (GSNF) metric described in this
    [paper](https://doi.org/10.1029/2022GB007520)."""

    def __init__(self, **keywords):

        super(ConfGSNF, self).__init__(**keywords)

        # overwrite some of the constructor with our info
        self.regions = ["global"]
        self.layout.cname = self.layout.cname.replace("2001-2002", "2009-2017")
        for page in self.layout.pages:
            page.cname = self.layout.cname
        self.layout.regions = self.regions

    def stageData(self, m):
        """Loads the reference data and computes the matching annual cycle from
        the model."""
        model_flux = self.keywords.get("model_flux", "nee")
        obs = Variable(filename=self.source, variable_name=self.variable)
        mod = (
            m.extractTimeSeries(
                model_flux,
                expression=model_flux,
                initial_time=(2009 - 1850) * 365,
                final_time=(2017 - 1850) * 365,
            )
            .trim(lat=[20, 90])
            .integrateInSpace()
            .annualCycle()
            .convert(obs.unit)
        )
        # just a hack to make sure the cycles are on the same time frame
        obs.time = mod.time
        obs.time_bnds = mod.time_bnds
        return obs, mod

    def confront(self, m):
        """Confronts the model with the reference."""
        # get the data and rename it as a cycle
        obs, mod = self.stageData(m)
        obs.name = "cycle_of_gsnf_over_global"
        mod.name = "cycle_of_gsnf_over_global"

        def _carbon(var):
            """Compute total carbon but only where flux is positive"""
            var = deepcopy(var)
            var.data = var.data.clip(0)
            var = var.integrateInTime()
            var.convert("Pg")
            var.name = "Growing Season Net Flux"
            return var

        # total carbon
        obs_sum = _carbon(obs)
        mod_sum = _carbon(mod)

        # simple score of total carbon based on relative error
        score = deepcopy(mod_sum)
        score.data = np.exp(-np.abs(mod_sum.data - obs_sum.data) / obs_sum.data)
        score.name = "Net Flux Score global"
        score.unit = "1"

        # outputs
        _path = partial(os.path.join, self.output_path)
        if self.master:
            with Dataset(_path("HIPTOM_Benchmark.nc"), mode="w") as dset:
                obs.toNetCDF4(dset, group="MeanState")
                obs_sum.toNetCDF4(dset, group="MeanState")
                dset.setncatts({"name": "Benchmark", "color": "k", "complete": 1})
        with Dataset(_path(f"HIPTOM_{m.name}.nc"), mode="w") as dset:
            mod.toNetCDF4(dset, group="MeanState")
            mod_sum.toNetCDF4(dset, group="MeanState")
            score.toNetCDF4(dset, group="MeanState")
            dset.setncatts(
                {
                    "name": m.name,
                    "color": m.color,
                    "complete": 1,
                    "weight": self.cweight,
                }
            )
