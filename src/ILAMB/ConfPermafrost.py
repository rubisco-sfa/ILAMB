from copy import deepcopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from netCDF4 import Dataset

from ILAMB import ilamblib as il
from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable

FIGURE_DPI = 100


def permafrost_extent_slater2013(
    tsl: Variable, dmax: float = 3.5, Teps: float = 273.15
) -> Variable:
    """Return the estimated permafrost extent.

    From Slater2013, "If soil at a depth within 3.5 m of the surface (based on the lower
    boundary of a model's soil layers) maintains a temperature of 0°C or less for the
    present and prior year, it is considered to contain permafrost."

    Parameters
    ----------
    tsl
        The soil temperatures in [K].
    dmax
        The maximum depth to consider in [m].
    Teps
        The temperature threshold to use to indicate permafrost [K].

    """
    tsl = tsl.trim(d=[0, dmax])
    # Only use whole years
    begin = np.argmin(tsl.time[:11] % 365)
    end = begin + int(tsl.time[begin:].size / 12.0) * 12
    # First we compute the maximum annual temperature for all depth/lat/lon and check if
    # it is below the temperature threshold.
    ext = tsl.data[begin:end].reshape((-1, 12) + tsl.data.shape[-3:]).max(axis=1) < Teps
    # If the difference is 0, then consecutive entries are the same and we have a
    # repeated year. If the ext itself is also 1, then we have a repeated year of
    # permafrost. If this is true at all in the time and depth dimension, we flag this
    # gridcell as permafrosted.
    ext = ((np.diff(ext, axis=0) == 0) * (ext[1:] == 1)).any(axis=(0, 1))
    # Now mask out the un-permarosted areas
    ext = np.ma.masked_values(ext, False).astype(float)
    ext = Variable(
        name="permafrost_extent",
        unit="1",
        data=ext,
        lat=tsl.lat,
        lat_bnds=tsl.lat_bnds,
        lon=tsl.lon,
        lon_bnds=tsl.lon_bnds,
    )
    return ext


def add_continent_shading(ax):
    """Add shading to the matplotlib axis."""
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", edgecolor="face", facecolor="0.875"
        ),
        zorder=-1,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m", edgecolor="face", facecolor="0.750"
        ),
        zorder=-1,
    )
    ax.set_global()


def plot_extent(var: Variable, filename: str) -> None:
    var.data = np.ma.masked_values(var.data, 0)
    fig, ax = plt.subplots(
        figsize=(10, 12),
        dpi=FIGURE_DPI,
        subplot_kw={
            "projection": ccrs.Orthographic(central_latitude=90, central_longitude=180)
        },
        tight_layout=True,
    )
    vals = np.unique(var.data.compressed())
    if vals.size == 1:
        lbls = ["           Permafrost Extent"]
    elif vals.size == 2:
        lbls = ["Continuous Permafrost", "Discontinuous Permafrost"]
    else:
        raise ValueError()
    norm_bins = np.sort(vals) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    pcm = ax.pcolormesh(
        np.hstack([var.lon_bnds[:, 0], var.lon_bnds[-1, -1]]),
        np.hstack([var.lat_bnds[:, 0], var.lat_bnds[-1, -1]]),
        var.data,
        cmap=plt.get_cmap("Blues_r", 4),
        norm=colors.BoundaryNorm(norm_bins, vals.size, clip=True),
        transform=ccrs.PlateCarree(),
    )
    add_continent_shading(ax)
    cb = fig.colorbar(pcm, location="bottom", pad=0.07)
    cb.set_ticks(vals, labels=lbls)
    cb.ax.tick_params(rotation=90, labelsize=18)
    fig.savefig(filename, dpi="figure")
    plt.close()


def plot_bias(var: Variable, filename: str) -> None:
    nm = 2
    fig, ax = plt.subplots(
        figsize=(10, 12),
        dpi=FIGURE_DPI,
        subplot_kw={
            "projection": ccrs.Orthographic(central_latitude=+90, central_longitude=180)
        },
        tight_layout=True,
    )
    cm = plt.get_cmap("bwr", 2 * nm + 1)
    norm_bins = np.sort(list(range(-nm, nm + 1))) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = colors.BoundaryNorm(norm_bins, 2 * nm + 1, clip=True)
    pcm = ax.pcolormesh(
        np.hstack([var.lon_bnds[:, 0], var.lon_bnds[-1, -1]]),
        np.hstack([var.lat_bnds[:, 0], var.lat_bnds[-1, -1]]),
        var.data,
        cmap=cm,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    add_continent_shading(ax)
    cb = fig.colorbar(pcm, location="bottom", pad=0.07)
    cb.set_ticks(
        list(range(-nm, nm + 1)),
        labels=[
            "Overlap Continuous",
            "   Overlap Discontinuous",
            "Excess Area",
            "Missed Discontinuous",
            "Missed Continuous",
        ],
    )
    cb.ax.tick_params(rotation=90, labelsize=18)
    fig.savefig(filename, dpi="figure")
    plt.close()


class ConfPermafrost(Confrontation):
    def __init__(self, **keywords):
        # Ugly, but this is how we call the Confrontation constructor
        super(ConfPermafrost, self).__init__(**keywords)

        # Now we overwrite some things which are different here
        self.layout
        self.regions = ["global"]
        self.layout.regions = self.regions
        self.weight = {
            "Missed Continuous Score": 2.0,
            "Missed Discontinuous Score": 1.0,
            "Excess Score": 3.0,
        }
        for page in self.layout.pages:
            page.setMetricPriority(
                [
                    "Total Area",
                    "Overlap Continuous Area",
                    "Overlap Discontinuous Area",
                    "Missed Continuous Area",
                    "Missed Discontinuous Area",
                    "Excess Area",
                    "Missed Continuous Score",
                    "Missed Discontinuous Score",
                    "Excess Score",
                    "Overall Score",
                ]
            )

    def stageData(self, m):
        """Return the observation and model permafrost extent."""
        obs = Variable(filename=self.source, variable_name="permafrost_extent")
        if obs.temporal:
            obs = obs.integrateInTime(mean=True)
        obs.name = "permafrost_extent"
        y0 = float(self.keywords.get("y0", 1985.0))
        yf = float(self.keywords.get("yf", 2005.0))
        dmax = float(self.keywords.get("dmax", 3.5))
        Teps = float(self.keywords.get("Teps", 273.15))
        tsl = m.extractTimeSeries(
            "tsl", alt_vars = ["TSOI"], initial_time=(y0 - 1850) * 365, final_time=(yf - 1850) * 365
        )
        tsl = tsl.trim(lat=[max(obs.lat.min(), tsl.lat.min()), 90])

        # estimate extent from the soil temperatures
        mod = permafrost_extent_slater2013(tsl, dmax=dmax, Teps=Teps)

        # mask out where modeled land fraction is less than 30%
        with np.errstate(under="ignore"):
            mod.data.mask += (
                tsl.area.data / il.CellAreas(None, None, tsl.lat_bnds, tsl.lon_bnds)
            ) < 0.3

        # mask out reference values which are marked as glaciers
        mod.data.mask += (
            Variable(
                unit="1",
                data=((~obs.data.mask) * (obs.data.data == 0)),
                lat=obs.lat,
                lat_bnds=obs.lat_bnds,
                lon=obs.lon,
                lon_bnds=obs.lon_bnds,
            )
            .interpolate(lat=mod.lat, lon=mod.lon)
            .data.data
        )
        obs.data.mask += obs.data.data == 0
        return obs, mod

    def confront(self, m):
        """."""

        def _area_bias(bias: Variable, flag: int) -> Variable:
            var = deepcopy(bias)
            var.data = var.data == flag
            return var.integrateInSpace().convert("1e6 km2").data

        obs, mod = self.stageData(m)

        # compose the grids and then just use standard numpy arrays for comparisons
        lat, lon, lat_bnds, lon_bnds = il._composeGrids(obs, mod)
        obs_c = obs.interpolate(lat=lat, lon=lon, lat_bnds=lat_bnds, lon_bnds=lon_bnds)
        mod_c = mod.interpolate(lat=lat, lon=lon, lat_bnds=lat_bnds, lon_bnds=lon_bnds)
        mask = obs_c.data.mask * mod_c.data.mask
        obs_c = obs_c.data.data
        mod_c = mod_c.data.data

        # build up the 'bias' array
        data = np.zeros(obs_c.shape)
        data[...] = np.nan
        data[(obs_c == 0) * (mod_c > 0)] = 0  # excess
        nm = 2
        for i in range(1, nm + 1):
            data[(obs_c == i) * (mod_c > 0)] = i - (nm + 1)  # missed
            data[(obs_c == i) * (mod_c < 1)] = (nm + 1) - i  # intersection
        bias = Variable(
            name="bias",
            unit="1",
            data=np.ma.masked_array(data, mask=mask),
            lat=lat,
            lon=lon,
        )

        # scalar areas
        area_obs = obs.integrateInSpace().convert("1e6 km2")
        area_mod = mod.integrateInSpace().convert("1e6 km2")
        area_obs.name = "Total Area global"
        area_mod.name = "Total Area global"
        area_excess = _area_bias(bias, 0)

        # scores
        both = {}
        missed = {}
        score_missed = {}
        area_both = 0.0
        for ptype, pflag in zip(["d", "c"], [1, 2]):
            both[ptype] = _area_bias(bias, -pflag)
            missed[ptype] = _area_bias(bias, pflag)
            with np.errstate(all="ignore"):
                score_missed[ptype] = both[ptype] / (both[ptype] + missed[ptype])
            area_both += both[ptype]
        score_excess = both[ptype] / area_mod.data

        with Dataset(
            "%s/%s_%s.nc" % (self.output_path, self.name, m.name), mode="w"
        ) as results:
            results.setncatts(
                {
                    "name": m.name,
                    "color": m.color,
                    "complete": 0,
                    "weight": self.cweight,
                }
            )
            for var in [
                mod,
                bias,
                area_mod,
                Variable(
                    name="Overlap Continuous Area global",
                    unit="1e6 km2",
                    data=both["c"],
                ),
                Variable(
                    name="Overlap Discontinuous Area global",
                    unit="1e6 km2",
                    data=both["d"],
                ),
                Variable(
                    name="Missed Continuous Area global",
                    unit="1e6 km2",
                    data=missed["c"],
                ),
                Variable(
                    name="Missed Discontinuous Area global",
                    unit="1e6 km2",
                    data=missed["d"],
                ),
                Variable(
                    name="Excess Area global",
                    unit="1e6 km2",
                    data=area_excess,
                ),
                Variable(
                    name="Missed Continuous Score global",
                    unit="1",
                    data=score_missed["c"],
                ),
                Variable(
                    name="Missed Discontinuous Score global",
                    unit="1",
                    data=score_missed["d"],
                ),
                Variable(
                    name="Excess Score global",
                    unit="1",
                    data=score_excess,
                ),
            ]:
                var.toNetCDF4(results, group="MeanState")
            results.setncattr("complete", 1)

        if not self.master:
            return

        with Dataset(
            f"{self.output_path}/{self.name}_Benchmark.nc", mode="w"
        ) as results:
            results.setncatts(
                {
                    "name": "Benchmark",
                    "color": np.asarray([0.5, 0.5, 0.5]),
                    "weight": self.cweight,
                    "complete": 0,
                }
            )
            area_obs.toNetCDF4(results, group="MeanState")
            results.setncattr("complete", 1)

    def modelPlots(self, m):
        # Add the figures to the html output page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        page.addFigure(
            "Temporally integrated period mean",
            "benchmark_timeint",
            "Benchmark_global_timeint.png",
            side="BENCHMARK EXTENT",
            legend=False,
        )
        page.addFigure(
            "Temporally integrated period mean",
            "timeint",
            "MNAME_global_timeint.png",
            side="MODEL EXTENT",
            legend=False,
        )
        page.addFigure(
            "Temporally integrated period mean",
            "bias",
            "MNAME_global_bias.png",
            side="BIAS",
            legend=False,
        )

        # model extent
        fname = f"{self.output_path}/{self.name}_{m.name}.nc"
        mod = Variable(
            filename=fname, variable_name="permafrost_extent", groupname="MeanState"
        )
        plot_extent(mod, f"{self.output_path}/{m.name}_global_timeint.png")

        # model bias
        bias = Variable(filename=fname, variable_name="bias", groupname="MeanState")
        plot_bias(bias, f"{self.output_path}/{m.name}_global_bias.png")

        if not self.master:
            return

        # obs extent
        obs = Variable(filename=self.source, variable_name="permafrost_extent")
        if obs.temporal:
            obs = obs.integrateInTime(mean=True)
        plot_extent(obs, f"{self.output_path}/Benchmark_global_timeint.png")

    def compositePlots(self):
        pass
