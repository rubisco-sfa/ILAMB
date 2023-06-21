"""The data used in this confrontation is queried from USGS servers. As part of the comparison, we compute two metrics to measure performance:

1. Nash-Sutcliffe Efficiency (NSE)

    ``NSE = 1 - SUM((mod - ref)^2) / SUM(ref - MEAN(ref))^2``

2. Kling-Gupta Efficiency (KGE)

    ``KGE = 1 - SQRT( (CORR(ref,mod)-1)^2 + (STD(mod)/STD(ref)-1)^2 + (MEAN(mod)/MEAN(ref)-1)^2 )``

"""
import os
import re
from functools import partial
from typing import Any

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cf_units import Unit
from dataretrieval import nwis
from pynhd import NLDI, NHDPlusHR

from ILAMB import Post as post
from ILAMB.Confrontation import Confrontation


def markdown_to_html(doc):
    """A very simple parser for rendering dosctrings in this module."""
    style = """
    <style type='text/css'>
      p.code {font-family:courier, courier new, serif;}
    </style>\n"""
    lines = doc.split("\n")
    for i, line in enumerate(lines):
        match = re.search("``.*``", line)
        if match:
            span = list(match.span())
            span[0] += 2
            span[-1] -= 2
            lines[i] = f'<p class="code">{line[slice(*span)]}</p>'
    out = style + "<br>".join(lines)
    return out


def convert_unit(
    series: pd.Series,
    tar_unit: str,
    mass_density: float = 998.2,
    molar_mass: str = 18.01528,
) -> pd.Series:
    """Convert the units of a pandas series.

    The unit string is expected to be stored in the `attrs` of the input series.
    See the example below for how to set the unit.

    Parameters
    ----------
    series
        The series whose units are to be converted.
    tar_unit
        The target unit to convert.
    mass_density
        The mass density of a substance in ``kg m-3``. The default is set to
        water and is used for converting units of the type ``kg m-2 s-1`` to
        ``mm d-1``.
    molar_mass
        The molar density of a substance in ``g mol-1``. The default is set to
        water and is used for converting units of the type from ``mol`` to
        ``kg``.

    Example
    -------
    >>> temperature.attrs['unit'] = 'degC'

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


def add_time_bounds(dset: xr.Dataset) -> xr.Dataset:
    """Add bounds to the time variable of the input dataset."""
    delt = dset["time"].isel({"time": 1}) - dset["time"].isel({"time": 0})
    dset["time_bnds"] = xr.DataArray(
        np.asarray([dset["time"], dset["time"] + delt]).T, dims=["time", "nb"]
    )
    dset["time"].attrs["bounds"] = "time_bnds"
    return dset


def add_global_attributes(filename: str, name: str, color: Any) -> None:
    """Add global attributes to the input dataset used by ILAMB."""
    dset = xr.Dataset()
    dset.attrs = {"name": name, "color": color, "complete": 1, "weight": 1}
    dset.to_netcdf(filename, mode="a")


def plot_usgs_site(sitecode: str, filename: str = None) -> None:
    """Plot the site in the context of CONUS and the immediate area.

    Parameters
    ----------
    sitecode
        The USGS sitecode.
    filename
        The optional name in which to save the plot. Defaults to
        ``{sitecode}.png``.
    """
    try:
        usa = gpd.read_file("USA_states_epsg4326.geojson")
        usa = usa[~usa.NAME_1.isin(["Alaska", "Hawaii"])]
    except:
        usa = None

    # We need to cache this information
    info = nwis.get_info(sites=sitecode)[0]
    area = NLDI().get_basins([sitecode]).to_crs("EPSG:4326")
    hucs = NHDPlusHR("huc12").bygeom(area.geometry[0].bounds)

    # Determine bounds and set the figure size
    bounds = area.bounds.iloc[0]
    delx = bounds["maxx"] - bounds["minx"]
    dely = bounds["maxy"] - bounds["miny"]
    figwidth = 20
    figheight = (0.5 * figwidth) * dely / delx

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, figheight), tight_layout=True)
    if usa is not None:
        usa.boundary.plot(edgecolor="k", linewidth=0.5, ax=axs[0], color=None)
    hucs.boundary.plot(
        ax=axs[1],
        edgecolor="k",
        linewidth=0.8,
        linestyle=(0, (5, 5)),
        color=None,
        label="Nearby Watershed (HUC12) Boundaries",
    )
    area.plot(ax=axs[1], color="blue", alpha=0.4)
    axs[0].plot(info.dec_long_va, info.dec_lat_va, "o", ms=10, color="red")
    axs[1].plot(
        info.dec_long_va, info.dec_lat_va, "o", ms=10, color="red", label="USGS Site"
    )

    # Legend tricks
    handles, labels = axs[1].get_legend_handles_labels()
    patch = mpatches.Patch(
        color="blue", alpha=0.4, linewidth=0, label="Contributing Area"
    )
    handles.append(patch)
    labels.append(patch._label)
    axs[0].legend(
        handles,
        labels,
        bbox_to_anchor=(0, 1.00, 1, 0.25),
        loc="lower left",
        mode="expand",
        ncol=3,
        borderaxespad=0,
        frameon=False,
    )

    # Finalize
    cx.add_basemap(axs[0], crs=area.crs)
    cx.add_basemap(axs[1], crs=area.crs)
    filename = filename if filename else f"{sitecode}.png"
    fig.suptitle(f"{info.station_nm[0]} ({sitecode})")
    fig.savefig(filename)
    plt.close()


def aggregate_gridded_data(
    dset: xr.Dataset, varname: str, basin: gpd.GeoDataFrame = None, mean: bool = False
) -> pd.Series:
    """."""

    var = dset[varname]
    measure = xr.where(var.notnull(), dset["area"] * dset["landfrac"], np.nan)
    if basin is not None:  # if a basin is given, mask points outside
        lon, lat = np.meshgrid(var["lon"], var["lat"])
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lon.flatten(), lat.flatten()), crs="EPSG:4326"
        )
        keep = gdf.within(basin.iloc[0]["geometry"]).to_numpy().reshape(lon.shape)
        measure = xr.where(keep, measure, np.nan)
    agg = (var * measure).sum(dim=["lat", "lon"])
    if mean:
        agg /= measure.sum()
    out = pd.Series(agg, index=agg["time"])
    out.attrs["unit"] = (
        var.attrs["units"]
        if mean
        else str(Unit(var.attrs["units"]) * Unit(dset["area"].attrs["units"]))
    )
    return out


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
        self.contributing_area = NLDI().get_basins([self.sitecode]).to_crs("EPSG:4326")

        # Setup a html layout for generating web views of the results
        pages = []
        pages.append(post.HtmlPage("MeanState", "Model View"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Site Information", "Discharge"])
        pages.append(post.HtmlAllModelsPage("AllModels", "All Models"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation", "Data Information"))
        pages[-1].setSections([])
        pages[-1].text = markdown_to_html(__doc__)
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
        ref.index = ref.index.tz_localize(None)
        row = self._table.loc[int(self.usgs_varid)]
        vname = row["Parameter Name/Description"].split(", ")[0].lower()
        unit = row["Parameter Unit"]
        ref.attrs["unit"] = unit
        mod = model.get_variable(vname, self.sitecode, frequency="D")

        # if the model returns a gridded object, try to aggregate
        if isinstance(mod, xr.Dataset):
            mod = aggregate_gridded_data(mod, vname, basin=self.contributing_area)

        # merging into a combined dataframe where the index is the datetime will
        # automatically produce nan's where either source is lacking variables.
        mod = convert_unit(mod, ref.attrs["unit"])
        try:
            cmb = pd.DataFrame({"mod": mod, "ref": ref})
        except TypeError:
            # assuming that the error is calendar related in the model
            mod.index = [pd.Timestamp(str(i)) for i in mod.index]
            cmb = pd.DataFrame({"mod": mod, "ref": ref})
        cmb = cmb.dropna()

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

        # discharge
        ref_mean = ref["discharge"].mean(dim="time")
        ref_mean.attrs["units"] = ref["discharge"].attrs["units"]
        ref_mean.name = "Mean Discharge global"
        mod_mean = mod["discharge"].mean(dim="time")
        mod_mean.attrs["units"] = mod["discharge"].attrs["units"]
        mod_mean.name = "Mean Discharge global"

        # Nash-Sutcliffe Efficiency
        nse = 1 - ((mod["discharge"] - ref["discharge"]) ** 2).sum(dim="time") / (
            (ref["discharge"] - ref_mean) ** 2
        ).sum(dim="time")
        nse = nse.clip(0, 1)
        nse.name = "NSE Score global"
        nse.attrs["units"] = "1"

        # Kling-Gupta Efficiency
        kge = 1 - np.sqrt(
            (xr.corr(ref["discharge"], mod["discharge"]) - 1) ** 2
            + (mod["discharge"].std() / ref["discharge"].std() - 1) ** 2
            + (mod_mean / ref_mean - 1) ** 2
        )
        kge = kge.clip(0, 1)
        kge.name = "KGE Score global"
        kge.attrs["units"] = "1"

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
        nse.to_netcdf(
            _path(f"{self.name}_{model.name}.nc"), group="MeanState/scalars", mode="a"
        )
        kge.to_netcdf(
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

        # time series plot
        page.addFigure(
            "Discharge",
            "discharge",
            "MNAME_global_discharge.png",
            longname="Discharge Time Series",
        )
        nyears = float(ref["time"].max() - ref["time"].min()) * 1e-9 / 3600 / 24 / 365
        fig, pax = plt.subplots(figsize=(2 / 8 * nyears, 2), tight_layout=True, dpi=200)
        ref["discharge"].plot(ax=pax, lw=1, color="k", label="USGS")
        mod["discharge"].plot(ax=pax, lw=1, color=model.color, label=model.name)
        pax.set_xlabel("")
        pax.legend()
        fig.savefig(_path(f"{model.name}_global_discharge.png"))
        plt.close()

        # scatter plot
        ref, mod = xr.align(ref, mod)
        page.addFigure(
            "Discharge",
            "scatter",
            "MNAME_global_scatter.png",
            longname="Discharge Scatter Plot",
        )
        nyears = float(ref["time"].max() - ref["time"].min()) * 1e-9 / 3600 / 24 / 365
        fig, pax = plt.subplots(figsize=(5, 5), tight_layout=True, dpi=200)
        pax.scatter(ref["discharge"], mod["discharge"], s=2, c=model.color)
        vmax = max(ref["discharge"].max(), mod["discharge"].max())
        pax.plot([0, vmax], [0, vmax], "--k")
        pax.set_xlabel(f"USGS discharge [{ref['discharge'].attrs['units']}]")
        pax.set_ylabel(f"{model.name} discharge [{mod['discharge'].attrs['units']}]")
        fig.savefig(_path(f"{model.name}_global_scatter.png"))
        plt.close()

    def compositePlots(self):
        """Render plots only once per master process."""
        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        page.addFigure("Site Information", "site", "site.png")
        plot_usgs_site(self.sitecode, os.path.join(self.output_path, "site.png"))
