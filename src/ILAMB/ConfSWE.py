import numpy as np
from cf_units import Unit

from ILAMB.Confrontation import Confrontation


class ConfSWE(Confrontation):
    def stageData(self, m):
        obs, mod = super(ConfSWE, self).stageData(m)

        def _transform(var):
            var.trim(lat=[20.0, 90.0])
            var.convert("m")
            vmin = Unit("m").convert(1e-3, var.unit)  # nothing under a [mm]
            var.data.mask += (var.data < vmin).all(axis=0)[np.newaxis, ...]
            with np.errstate(all="ignore"):
                var.data = np.log10(var.data)
            var.unit = f"{Unit(var.unit).log(10)}"
            return var

        obs = _transform(obs)
        mod = _transform(mod)

        return obs, mod
