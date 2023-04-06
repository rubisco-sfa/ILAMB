"""."""
import os
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cf_units import Unit
from dataretrieval import nwis

from . import Post as post
from .Confrontation import Confrontation


def convert_unit(
    series: pd.Series,
    tar_unit: str,
    mass_density: float = 998.2,
    molar_mass: str = 18.01528,
):
    """Converts units of a pandas series where the unit string is expected to be
    stored in the `attrs` of the input series.

    Example:
    temperature.attrs['unit'] = 'degC'
    """
    if "unit" not in series.attrs:
        raise ValueError("Input series has no unit.")
    src_unit = Unit(series.attrs["unit"])
    tar_unit = Unit(tar_unit)
    molar_density = mass_density * 1000 / molar_mass
    mass_density_unit = Unit("kg m-3")
    molar_density_unit = Unit("mol m-3")
    fct = 1.0
    if ((src_unit / tar_unit) / mass_density_unit).is_dimensionless():
        fct /= mass_density
        src_unit /= mass_density_unit
    elif ((tar_unit / src_unit) / mass_density_unit).is_dimensionless():
        fct *= mass_density
        src_unit *= mass_density_unit
    if ((src_unit / tar_unit) / molar_density_unit).is_dimensionless():
        fct /= molar_density
        src_unit /= molar_density_unit
    elif ((tar_unit / src_unit) / molar_density_unit).is_dimensionless():
        fct *= molar_density
        src_unit *= molar_density_unit
    fct = src_unit.convert(fct, tar_unit)
    series *= fct
    series.attrs["unit"] = str(tar_unit)
    return series


def add_time_bounds(dset: xr.Dataset):
    """."""
    delt = dset["time"].isel({"time": 1}) - dset["time"].isel({"time": 0})
    dset["time_bnds"] = xr.DataArray(
        np.asarray([dset["time"], dset["time"] + delt]).T, dims=["time", "nb"]
    )
    dset["time"].attrs["bounds"] = "time_bnds"
    return dset


def add_global_attributes(filename: str, name: str, color: Any):
    """."""
    dset = xr.Dataset()
    dset.attrs = {"name": name, "color": color, "complete": 1, "weight": 1}
    dset.to_netcdf(filename, mode="a")


class ConfUSGS(Confrontation):
    """An ILAMB confrontation which requires no source data specified and
    instead downloads sources from the specified `sitecode` from `time_start` to
    `time_end`."""

    _table = (
        pd.read_html("https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY")[0]
    ).set_index("Parameter Code")

    def __init__(self, **keywords):
        # ILAMB requires sources, but we will download them. Need to fake a source
        xr.Dataset().to_netcdf("dummy_source.nc")
        keywords["source"] = "dummy_source.nc"
        super(ConfUSGS, self).__init__(**keywords)
        os.system("rm dummy_source.nc")

        self.name = "USGS"
        self.regions = ["global"]
        self.layout.regions = self.regions
        self.sitecode = keywords.get("sitecode", None)
        self.time_start = keywords.get("time_start", None)
        self.time_end = keywords.get("time_end", None)
        self.usgs_varid = keywords.get("usgs_varid", "00060")
        assert self.sitecode
        assert self.time_start
        assert self.time_end

        # Populate some information about this site
        try:
            resp = nwis.get_info(sites=self.sitecode)
            df = resp[0]
            name = (df.iloc[0]["station_nm"]).split(" near ")[0]
        except ValueError:
            name = self.name
        self.longname = name

        # Setup a html layout for generating web views of the results
        pages = []
        pages.append(post.HtmlPage("MeanState", "Model View"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Discharge"])
        pages.append(post.HtmlAllModelsPage("AllModels", "All Models"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation", "Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        self.layout = post.HtmlLayout(
            pages,
            self.longname,
            years=(
                pd.Timestamp(self.time_start).year,
                pd.Timestamp(self.time_end).year,
            ),
        )

    def stageData(self, model):
        """Downloads reference data from USGS servers, finds a matching model
        object, and handles units.

        * We probably want to cache this data before computing the model
          intersection to keep from hitting USGS servers too much.
        * The data comes in the form of pandas Series, but they do not store
          attributes when dumped so we convert to xarray objects.
        """
        ref = nwis.get_record(
            sites=self.sitecode,
            service="dv",
            start=self.time_start,
            end=self.time_end,
            parameterCd=self.usgs_varid,
        )[f"{self.usgs_varid}_Mean"]
        row = self._table.loc[int(self.usgs_varid)]
        vname = row["Parameter Name/Description"].split(", ")[0].lower()
        unit = row["Parameter Unit"]
        ref.attrs["unit"] = unit
        mod = model.get_variable(vname, self.sitecode)
        mod = convert_unit(mod, ref.attrs["unit"])
        # merging into a combined dataframe where the index is the datetime will
        # automatically produce nan's where either source is lacking variables.
        cmb = pd.DataFrame({"mod": mod, "ref": ref}).dropna()
        ref = xr.DataArray(
            cmb["ref"].values, coords=[("time", cmb.index.values)], dims="time"
        )
        mod = xr.DataArray(
            cmb["mod"].values, coords=[("time", cmb.index.values)], dims="time"
        )
        ref.attrs["units"] = unit
        mod.attrs["units"] = unit
        ref = xr.Dataset({vname: ref})
        mod = xr.Dataset({vname: mod})
        return ref, mod

    def confront(self, model):
        """."""
        _path = partial(os.path.join, self.output_path)
        ref, mod = self.stageData(model)

        # simple mean discharge metric
        ref_mean = ref["discharge"].mean(dim="time")
        ref_mean.attrs["units"] = ref["discharge"].attrs["units"]
        ref_mean.name = "Mean Discharge global"
        mod_mean = mod["discharge"].mean(dim="time")
        mod_mean.attrs["units"] = mod["discharge"].attrs["units"]
        mod_mean.name = "Mean Discharge global"
        score = np.exp(-np.abs(ref_mean-mod_mean)/ref_mean)
        score.attrs["units"] = "1"
        score.name = "Discharge Score global"
        
        # output to intermediate netcdf files
        if self.master:
            ref.to_netcdf(_path(f"{self.name}_Benchmark.nc"), group="MeanState")
            ref_mean.to_netcdf(
                _path(f"{self.name}_Benchmark.nc"), group="MeanState/scalars", mode="a"
            )
        mod.to_netcdf(_path(f"{self.name}_{model.name}.nc"), group="MeanState")
        mod_mean.to_netcdf(
            _path(f"{self.name}_{model.name}.nc"), group="MeanState/scalars", mode="a"
        )
        score.to_netcdf(
            _path(f"{self.name}_{model.name}.nc"), group="MeanState/scalars", mode="a"
        )
        
        # just to get global attributes in the right place
        add_global_attributes(_path(f"{self.name}_Benchmark.nc"), "Benchmark", "k")
        add_global_attributes(
            _path(f"{self.name}_{model.name}.nc"), model.name, model.color
        )
        
    def modelPlots(self, model):
        """."""
        _path = partial(os.path.join, self.output_path)
        # try to load the datasets
        try:
            ref = xr.open_dataset(_path(f"{self.name}_Benchmark.nc"), group="MeanState")
            mod = xr.open_dataset(
                _path(f"{self.name}_{model.name}.nc"), group="MeanState"
            )
        except FileNotFoundError:
            return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]

        # the figure size depends on how much time
        page.addFigure("Discharge", "discharge", "MNAME_discharge.png")
        nyears = float(ref["time"].max() - ref["time"].min()) * 1e-9 / 3600 / 24 / 365
        fig, pax = plt.subplots(
            figsize=(2 / 10 * nyears, 2), tight_layout=True, dpi=200
        )
        ref["discharge"].plot(ax=pax, color="k", label="USGS")
        mod["discharge"].plot(ax=pax, color=model.color, label=model.name)
        pax.set_xlabel("")
        pax.legend()
        fig.savefig(_path(f"{model.name}_discharge.png"))
        plt.close()

    def compositePlots(self):
        pass
