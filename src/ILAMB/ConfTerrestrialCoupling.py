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

        # just for testing
        mod = deepcopy(obs)
        mod.data += obs.data.max() * 0.1 * (np.random.rand(*mod.data.shape) - 0.5)

        return obs, mod
