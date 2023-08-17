import numpy as np

import ILAMB.ilamblib as il
from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable


def active_layer_thickness(tsl: Variable, Teps: float = 273.15) -> Variable:
    """Return the estimated activate layer thickness [m].

    Parameters
    ----------
    tsl
        The soil temperatures in [K].
    Teps
        The temperature threshold to use to indicate activity [K].

    """
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
            logstring="[%s][%s]" % (self.longname, m.name),
        )
        mod.convert(obs.unit)
        return obs, mod
