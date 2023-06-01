"""."""
import os

import numpy as np
import xarray as xr
from typing import Tuple
import datetime

from .Confrontation import Confrontation
from .Variable import Variable


def coupling_index(control: xr.DataArray, respond: xr.DataArray) -> xr.DataArray:
    """Return the coupling index in units of the responding variable.

    Parameters
    ----------
    control
        The controlling variable
    respond
        The responding variable, the index will be in the units of
        this variable.

    """
    control, response = xr.align(control, respond, join="override")
    control = control.sel(time=control["time.season"] == "JJA")
    respond = respond.sel(time=respond["time.season"] == "JJA")
    cov = xr.cov(control, respond, dim="time")
    std = control.std(dim="time")
    cov.load()
    std.load()
    coupling = cov/std
    coupling.attrs = {"long_name": "Coupling index", "units": respond.attrs["units"]}
    coupling.load()
    return coupling


def max_time_bounds(*dss):
    """."""
    t0 = []
    tf = []
    for ds in dss:
        if "time" not in ds:
            continue
        time = ds["time"].attrs["bounds"] if "bounds" in ds["time"].attrs else "time"
        time = ds[time] if time in ds else ds["time"]
        t0.append(time.min())
        tf.append(time.max())
    t0 = max(t0)
    tf = min(tf)
    return t0, tf


class ConfTerrestrialCoupling(Confrontation):
    def __init__(self, **keywords):
        super(ConfTerrestrialCoupling, self).__init__(**keywords)
        for srcname in ["mrsos_source", "hfss_source"]:
            src = self.keywords.get(srcname, None)
            if src is None:
                continue
            self.keywords[srcname] = os.path.join(os.environ["ILAMB_ROOT"], src)

    def stageData(self, m):

        # read in reference data and find maximal overlap
        mrsos_obs = xr.open_dataset(
            self.keywords.get("mrsos_source", self.source), chunks=dict(time=1800)
        )
        hfss_obs = xr.open_dataset(
            self.keywords.get("hfss_source", self.source), chunks=dict(time=1800)
        )
        tmin, tmax = max_time_bounds(mrsos_obs, hfss_obs)
        if len(mrsos_obs['time']) != len(hfss_obs['time']):            
            mrsos_obs = mrsos_obs.sel(time=slice(tmin, tmax))
            hfss_obs = hfss_obs.sel(time=slice(tmin, tmax))
        ci_obs = coupling_index(mrsos_obs["mrsos"], hfss_obs["hfss"])

        # for backwards compatibility, now convert to ILAMB object
        tbnds = np.asarray(
            [
                [
                    ((tmin.dt.year - 1850) * 365) + tmin.dt.dayofyear,
                    ((tmax.dt.year - 1850) * 365) + tmax.dt.dayofyear,
                ]
            ],
            dtype=float,
        )
        ndata = len(ci_obs['SITE']) if 'SITE' in ci_obs.dims else None
        data = np.ma.masked_invalid(ci_obs.to_numpy())
        if len(data.mask)==1: data.mask = np.zeros_like(data)
        data = data.reshape((1,)+data.shape)
        obs = Variable(
            name="ci",
            data=data,
            unit=ci_obs.attrs["units"],
            ndata=ndata,
            lat=mrsos_obs["lat"].to_numpy(),
            lon=mrsos_obs["lon"].to_numpy(),
            time=tbnds.mean(axis=1),
            time_bnds=tbnds,
        )
        print(obs)
        
        # load the model result
        mod_ds = {}
        units = {}
        lat = xr.DataArray(obs.lat, dims="SITE")
        lon = xr.DataArray(obs.lon, dims="SITE")
        for vname in ["mrsos", "hfss"]:
            var = xr.open_mfdataset(m.variables[vname])
            cal = var["time"].values[0].__class__
            t0 = tmin.dt.day
            tf = tmax.dt.day
            if cal.__name__ == "Datetime360Day":
                if t0 == 31:
                    t0 = 30
                if tf == 31:
                    tf = 30
            units[vname] = var[vname].attrs["units"]
            var = var.sel(
                time=slice(
                    cal(tmin.dt.year, tmin.dt.month, t0),
                    cal(tmax.dt.year, tmax.dt.month, tf),
                ),
            )
            var = var.sel(lat=lat, lon=lon, method="nearest")
            var.load()
            mod_ds[vname] = (["time", "SITE"], var[vname].data)
        mod_ds = xr.Dataset(
            data_vars=mod_ds,
            coords={"time": var["time"].data, "SITE": var["SITE"].data},
        )
        for var, unit in units.items():
            mod_ds[var].attrs["units"] = unit
        ci_mod = coupling_index(mod_ds["mrsos"], mod_ds["hfss"])
        data = np.ma.masked_invalid(ci_mod.to_numpy())
        mod = Variable(
            name="ci",
            data=data.reshape((1,) + data.shape),
            unit=ci_mod.attrs["units"],
            lat=lat.to_numpy(),
            lon=lon.to_numpy(),
            ndata=len(lat),
            time=tbnds.mean(axis=1),
            time_bnds=tbnds,
        )
        return obs, mod
