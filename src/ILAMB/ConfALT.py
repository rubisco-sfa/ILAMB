from copy import deepcopy
from typing import Union

import numpy as np

import ILAMB.ilamblib as il
from ILAMB.Confrontation import Confrontation
from ILAMB.constants import bnd_months, dpm_noleap
from ILAMB.Variable import Variable


def annual_mean(var: Variable) -> Variable:
    """Return the annual mean for each year in the input variable."""
    years = np.unique(np.floor(var.time / 365)) * 365
    years = np.asarray([years, np.hstack([years[1:], years.max() + 365])]).T
    return var.coarsenInTime(years)


def seasonal_mean(var: Variable, season: str) -> Variable:
    """Return the weighted seasonal mean of the given variable and season"""
    # compute seasonal weights
    assert season in ["DJF", "JJA"]  # limited implementation
    start_of_season = 0
    month = 1
    if season == "DJF":
        month = 12
    elif season == "JJA":
        month = 6
    start_of_season = bnd_months[month - 1]
    weights = np.roll(dpm_noleap, -(month - 1))
    weights[len(season) :] = 0
    # only use complete seasons, can span the calendar year break
    begin = np.argmin(np.abs((var.time_bnds[:12, 0] % 365) - start_of_season))
    end = begin + int(var.time[begin:].size / 12.0) * 12
    data = var.data[begin:end].reshape((-1, 12) + var.data.shape[1:])
    tbnd = var.time_bnds[begin:end].reshape((-1, 12, 2))[:, : len(season), :]
    tbnd = np.array([tbnd[:, 0, 0], tbnd[:, -1, -1]]).T
    return Variable(
        name=var.name + season,
        unit=var.unit,
        data=(data * weights[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1)
        / weights.sum(),
        time=tbnd.mean(axis=1),
        time_bnds=tbnd,
        lat=var.lat,
        lat_bnds=var.lat_bnds,
        lon=var.lon,
        lon_bnds=var.lon_bnds,
    )


def effective_snow_depth(snd: Variable) -> Variable:
    """Return the effective snow depth.

    Snow has a big impact on the soil temperatures and presence or absence of permafrost
    in the northern high latitudes. The effective snow depth describes the insulation of
    snow over the cold period. It is an integral value such that the mean snow depth in
    each month is weighted by its duration.

    """
    # cold season starts in October (day 274)
    begin = np.argmin(np.abs((snd.time[:12] % 365) - (274 + 5)))
    end = begin + int(snd.time[begin:].size / 12.0) * 12
    data = snd.data[begin:end].reshape((-1, 12) + snd.data.shape[1:])
    tbnd = snd.time_bnds[begin:end].reshape((-1, 12, 2))[:, [0, -1], [0, 1]]
    # weights
    M = 6
    m = np.array(range(1, M + 1))
    weights = np.hstack([(M + 1 - m) / m.sum(), np.zeros(6)])
    # return effective snow depth, an annual quantity starting in October
    return Variable(
        name="snd_eff",
        unit=snd.unit,
        data=(data * weights[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1),
        time=tbnd.mean(axis=1),
        time_bnds=tbnd,
        lat=snd.lat,
        lat_bnds=snd.lat_bnds,
        lon=snd.lon,
        lon_bnds=snd.lon_bnds,
    )


def season_offset(
    tsl: Variable, tas: Variable, season: str, depth: float = 0.2
) -> Variable:
    """
    The seasonal offset is defined as the difference between the mean soil temperature
    at 0.2 m and the mean air temperature for the period from December to February
    """
    assert season in ["DJF", "JJA"]
    level = np.argmin(np.abs(tsl.depth - depth))
    offset = seasonal_mean(
        tsl.integrateInDepth(
            z0=tsl.depth_bnds[level, 0], zf=tsl.depth_bnds[level, 1], mean=True
        ),
        season,
    )
    ts = seasonal_mean(tas, season).convert(tsl.unit)
    offset.data -= ts.data
    offset.name = f"offset{season}"
    return offset


def thermal_offset(magst: Variable, magt: Variable) -> Variable:
    """Return the thermal offset.

    The thermal offset is the temperature difference between the annual mean soil
    temperature at 0.2 m (mean annual ground surface temperature, MAGST) and the annual
    mean soil temperature at the top of the permafrost (mean annual ground temperature,
    MAGT)."""
    magt.convert(magst.unit)
    offset = deepcopy(magst)
    offset.data -= magt.data
    offset.name = "offsetthermal"
    return offset


def mean_annual_ground_surface_temperature(
    tsl: Variable, depth: float = 0.2
) -> Variable:
    """
    Also known as MAGST in Burke2020.
    """
    level = np.argmin(np.abs(tsl.depth - depth))
    magst = annual_mean(
        tsl.integrateInDepth(
            z0=tsl.depth_bnds[level, 0], zf=tsl.depth_bnds[level, 1], mean=True
        )
    )
    magst.name = "magst"
    return magst


def mean_annual_air_temperature(tas: Variable) -> Variable:
    """
    Also known as MAAT in Burke2020.
    """
    maat = annual_mean(tas)
    maat.name = "maat"
    return maat


def active_layer_thickness(tsl: Variable, Teps: float = 273.15) -> Variable:
    """Return the estimated activate layer thickness [m].

    The depth to where the maximum annual temperature is above Teps.

    Parameters
    ----------
    tsl
        The soil temperatures in [K].
    Teps
        The temperature threshold to use to indicate activity [K].

    """
    # Only use whole years
    begin = np.argmin(tsl.time[:11] % 365)
    end = begin + int(tsl.time[begin:].size / 12.0) * 12
    tsl.time = tsl.time[begin:end]
    tsl.time_bnds = tsl.time_bnds[begin:end]
    tsl.data = tsl.data[begin:end]

    # Compute the annual max soil temperature at every depth/lat/lon. Note that this
    # approach will not return a single month where you have a maximum nor will it catch
    # places where you may have a few frozen layers encased in a larger active region.
    tmax = tsl.data.reshape((-1, 12) + tsl.data.shape[1:]).max(axis=1)

    # Compute the depth cell heights and extent to the year/lat/lon dimensions
    alt = tsl.depth_bnds[:, 1] - tsl.depth_bnds[:, 0]
    alt = np.ones(tmax.shape[0])[:, np.newaxis] * alt
    for i in range(tmax.ndim - 2):
        alt = alt[..., np.newaxis] * np.ones(tmax.shape[2 + i])

    # The active layer thickness is the sum of these depths where the max temperature is
    # greater than a threshold
    alt = np.ma.masked_array(alt, mask=tmax.mask + (tmax < Teps))
    alt = alt.sum(axis=1)

    # Write out variable
    tb = tsl.time_bnds.reshape((-1, 12, 2))
    tb = np.asarray([tb[..., 0].min(axis=1), tb[..., 1].max(axis=1)]).T
    alt = Variable(
        name="alt",
        data=alt,
        unit="m",
        time=tb.mean(axis=1),
        time_bnds=tb,
        lat=tsl.lat,
        lon=tsl.lon,
        ndata=tsl.ndata,
    )
    return alt


def thawed_fraction(
    tsl: Variable, Teps: float = 273.15, max_depth: float = 2.0
) -> Variable:
    """Return the annual mean thawed fraction.

    Harp et al. (2016) defined the annual mean thawed fraction for permafrost soils. By
    default we assume 2 [m] of dept to consider which is relatively shallow but enables
    models with shallower soil depths to be included consistently within the analysis.
    """
    frac = deepcopy(tsl)
    frac.data = tsl.data > Teps
    frac = annual_mean(frac.integrateInDepth(z0=0, zf=max_depth, mean=True))
    frac.name = "thawed_fraction"
    frac.unit = "1"
    return frac


def mean_annual_ground_temperature(
    tsl: Variable, alt: Union[Variable, None] = None
) -> Variable:
    """Return the mean annual temperature at the base of the permafrost."""
    if alt is None:
        alt = active_layer_thickness(tsl)
    mean = annual_mean(tsl)
    magt = deepcopy(alt)
    ind = mean.depth_bnds[:, 0].searchsorted(alt.data.data) - 1
    # there must be a better way
    for i in range(alt.data.shape[0]):
        for j in range(alt.data.shape[1]):
            for k in range(alt.data.shape[2]):
                magt.data[i, j, k] = mean.data[i, ind[i, j, k], j, k]
    magt.name = "magt"
    magt.unit = mean.unit
    return magt


class ConfALT(Confrontation):
    def stageData(self, m):
        """Return the observation and model permafrost extent."""
        Teps = float(self.keywords.get("Teps", 273.15))
        obs = Variable(
            filename=self.source,
            variable_name=self.variable,
            alternate_vars=self.alternate_vars,
            t0=None if len(self.study_limits) != 2 else self.study_limits[0],
            tf=None if len(self.study_limits) != 2 else self.study_limits[1],
        )
        obs.data *= self.scale_factor
        self.pruneRegions(obs)

        # estimate the active layer thickness from the soil temperatures
        tsl = m.extractTimeSeries(
            "tsl",
            alt_vars=self.alternate_vars,
            expression=self.derived,
            initial_time=obs.time_bnds[0, 0],
            final_time=obs.time_bnds[-1, 1],
            lats=None if obs.spatial else obs.lat,
            lons=None if obs.spatial else obs.lon,
        )
        if tsl.spatial:
            tsl = tsl.trim(lat=[obs.lat.min(), 90])
        mod = active_layer_thickness(tsl, Teps=Teps)

        # ensure that these are now comparable
        obs, mod = il.MakeComparable(
            obs,
            mod,
            mask_ref=True,
            clip_ref=True,
            extents=self.extents,
            logstring=f"[{self.longname}][{m.name}]",
        )
        mod.convert(obs.unit)
        return obs, mod

    def determinePlotLimits(self):
        super().determinePlotLimits()
