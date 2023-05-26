"""."""
from copy import deepcopy

import numpy as np
import xarray as xr

from .Confrontation import Confrontation
from .Variable import Variable


def compute_coupling_index(dset: xr.Dataset, control: str, respond: str) -> xr.Dataset:
    """Return the dataset with the coupling index added.

    Parameters
    ----------
    dset
        The dataset containing DataArray's labeled `control` and `respond`
    control
        The name of the controlling variable
    respond
        The name of the responding variable, the index will be in the units of
        this variable.
    """
    assert control in dset
    assert respond in dset
    ssn = dset.sel(time=dset["time.season"] == "JJA")
    dset["ci"] = xr.cov(ssn[control], ssn[respond], dim="time") / ssn[control].std(
        dim="time"
    )
    dset["ci"].attrs["units"] = dset[respond].attrs["units"]
    return dset


class ConfTerrestrialCoupling(Confrontation):
    def stageData(self, m):
        # eventually we need mrsos_source and hfls_source
        obs_ds = xr.open_dataset(self.source)
        obs_ds = compute_coupling_index(obs_ds, "mrsos", "hfls")

        # for backwards compatibility, now convert to ILAMB object
        tmin = obs_ds["time"].min()
        tmax = obs_ds["time"].max()
        tbnds = np.asarray(
            [
                [
                    ((tmin.dt.year - 1850) * 365) + tmin.dt.dayofyear,
                    ((tmax.dt.year - 1850) * 365) + tmax.dt.dayofyear,
                ]
            ],
            dtype=float,
        )
        obs = Variable(
            name="ci",
            data=np.ma.masked_array(
                obs_ds["ci"].to_numpy(), mask=np.zeros_like(obs_ds["ci"])
            ).reshape((1, -1)),
            unit=obs_ds["ci"].attrs["units"],
            ndata=len(obs_ds["SITE"]),
            lat=obs_ds["lat"].to_numpy(),
            lon=obs_ds["lon"].to_numpy(),
            time=tbnds.mean(axis=1),
            time_bnds=tbnds,
        )

        # load the model result
        mod_ds = {}
        units = {}
        lat = xr.DataArray(obs.lat, dims="SITE")
        lon = xr.DataArray(obs.lon, dims="SITE")
        for vname in ["mrsos", "hfls"]:
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
        mod_ds = compute_coupling_index(mod_ds, "mrsos", "hfls")
        data = np.ma.masked_invalid(mod_ds["ci"].to_numpy())
        mod = Variable(
            name="ci",
            data=data.reshape((1,) + data.shape),
            unit=mod_ds["ci"].attrs["units"],
            lat=lat.to_numpy(),
            lon=lon.to_numpy(),
            ndata=len(lat),
            time=tbnds.mean(axis=1),
            time_bnds=tbnds,
        )
        return obs, mod
