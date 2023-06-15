"""."""
import os

import numpy as np

from . import ilamblib as il
from .Confrontation import Confrontation
from .Variable import Variable


def variable_ratio(num: Variable, den: Variable):
    """."""
    den.convert(num.unit)
    years = np.asarray([num.time_bnds[::12, 0], num.time_bnds[11::12, 1]]).T
    num = num.coarsenInTime(years)
    den = den.coarsenInTime(years)
    with np.errstate(under="ignore"):
        ratio = num.data / den.data
    ratio = Variable(
        name="ratio",
        unit="1",
        data=ratio,
        time=num.time,
        time_bnds=num.time_bnds,
        lat=num.lat,
        lat_bnds=num.lat_bnds,
        lon=num.lon,
        lon_bnds=num.lon_bnds,
    )
    return ratio


class ConfRatio(Confrontation):
    """.
    * over_regions, a list of regions overwhich to compute averages, could be basins here
    * mask_lower_percentile = 0.05, would create a mask of the lower end of the denominator and intepolate the mask to the model
    """

    def __init__(self, **keywords):
        required_keys = [
            "numerator_source",
            "numerator_variable",
            "denominator_source",
            "denominator_variable",
        ]
        if set(required_keys).difference(keywords):
            msg = f"This confrontation requires a different set of keywords: {','.join(required_keys)}"
            raise il.MisplacedData(msg)
        for key in ["numerator_source", "denominator_source"]:
            keywords[key] = os.path.join(os.environ["ILAMB_ROOT"], keywords[key])
        keywords["source"] = keywords["numerator_source"]
        keywords["skip_cycle"] = True
        keywords["skip_sd"] = True
        keywords["rmse_score_basis"] = "series"
        self.source = keywords["numerator_source"]
        super().__init__(**keywords)

    def stageData(self, m):
        """."""
        # construct observation ratios
        kwargs = self.keywords
        num = Variable(
            filename=kwargs["numerator_source"],
            variable_name=kwargs["numerator_variable"],
        )
        den = Variable(
            filename=kwargs["denominator_source"],
            variable_name=kwargs["denominator_variable"],
            t0=num.time_bnds.min(),
            tf=num.time_bnds.max(),
        )
        num.trim(t=[den.time_bnds.min(), den.time_bnds.max()])
        obs = variable_ratio(num, den)

        # constructor model ratios
        num = m.extractTimeSeries(
            kwargs["numerator_variable"],
            initial_time=obs.time_bnds[0, 0],
            final_time=obs.time_bnds[-1, 1],
            lats=None if obs.spatial else obs.lat,
            lons=None if obs.spatial else obs.lon,
        )
        den = m.extractTimeSeries(
            kwargs["denominator_variable"],
            initial_time=obs.time_bnds[0, 0],
            final_time=obs.time_bnds[-1, 1],
            lats=None if obs.spatial else obs.lat,
            lons=None if obs.spatial else obs.lon,
        )
        mod = variable_ratio(num, den)
        return obs, mod
