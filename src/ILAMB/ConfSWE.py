import numpy as np

from ILAMB.Confrontation import Confrontation


class ConfSWE(Confrontation):
    def stageData(self, m):
        # same as regular, but we subtract off the minimum
        obs, mod = super(ConfSWE, self).stageData(m)
        omin = obs.data.min(axis=0)
        mmin = mod.data.min(axis=0)
        obs.data -= omin[np.newaxis, ...]
        mod.data -= mmin[np.newaxis, ...]
        return obs, mod
