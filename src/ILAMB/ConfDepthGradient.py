import numpy as np

from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable


def compute_depth_gradient(
    var: Variable, depth_min: float = 200, depth_max: float = 1000
):
    """Return the depth gradient of the input variable.

    Parameters
    ----------
    var
        the input variable.
    depth_min
        the minimum depth to consider.
    depth_max
        the maximum depth to consider.

    """
    mean = var.integrateInTime(mean=True)
    mean.trim(d=[depth_min, depth_max])
    shp = mean.data.shape
    slope = -np.polyfit(mean.depth, mean.data.data.reshape((shp[0], -1)), deg=1)[0, :]
    slope = slope.reshape((1,) + shp[-2:])
    slope = np.ma.masked_array(slope, mask=mean.data.mask[0, ...])
    tbnds = np.asarray([[var.time_bnds.min(), var.time_bnds.max()]])
    return Variable(
        name=f"{var.name}_depth_gradient",
        unit=f"{var.unit} m-1",
        data=slope,
        time=tbnds.mean(axis=1),
        time_bnds=tbnds,
        lat=mean.lat,
        lat_bnds=mean.lat_bnds,
        lon=mean.lon,
        lon_bnds=mean.lon_bnds,
    )


class ConfDepthGradient(Confrontation):
    def stageData(self, m):
        obs, mod = super(ConfDepthGradient, self).stageData(m)
        obs = compute_depth_gradient(obs)
        mod = compute_depth_gradient(mod)
        return obs, mod
