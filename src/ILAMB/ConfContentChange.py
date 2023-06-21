"""blah"""
import numpy as np

from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable


def _insert_dummy_time(var):
    """."""
    var.data = var.data.reshape((1,) + var.data.shape)
    var.time = np.asarray([0.5])
    var.time_bnds = np.asarray([[0.0, 1.0]])
    var.temporal = True
    return var


class ConfContentChange(Confrontation):
    """."""

    def stageData(self, m):
        """."""
        obs = Variable(filename=self.source, variable_name=self.variable)
        mod = m.extractTimeSeries(
            self.variable,
            alt_vars=self.alternate_vars,
            expression=self.derived,
            initial_time=(1994 - 1850) * 365,
            final_time=(2008 - 1850) * 365,
        )

        # we do not have a good way to difference objects, so here we do a manual hack
        mod.data = mod.data[-1] - mod.data[0]
        mod.time = None
        mod.time_bnds = None
        mod.temporal = None

        # make model comparable to reference
        mod = mod.integrateInDepth(z0=0, zf=3000)
        mod.convert(obs.unit)
        mod.name = obs.name

        obs = _insert_dummy_time(obs)
        mod = _insert_dummy_time(mod)

        return obs, mod
