from .Confrontation import Confrontation
from .Variable import Variable
from . import ilamblib as il
import numpy as np

class ConfBurntArea(Confrontation):
    """
    Burnt area is sometimes expressed as a total per month, and
    sometimes as the mean rate in the month. This specialized code
    detects the difference and converts.
    """
    def stageData(self,m):
        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars,
                       t0 = None if len(self.study_limits) != 2 else self.study_limits[0],
                       tf = None if len(self.study_limits) != 2 else self.study_limits[1])
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars     = self.alternate_vars,
                                  expression   = self.derived,
                                  initial_time = obs.time_bnds[ 0,0],
                                  final_time   = obs.time_bnds[-1,1])
        try:
            mod.convert(obs.unit)
        except:
            mod.convert("d-1")
            dt = mod.time_bnds[:,1]-mod.time_bnds[:,0]
            for i in range(mod.data.ndim-1): dt = np.expand_dims(dt,axis=-1)
            mod.data *= dt
            mod.unit  = "1"
        obs,mod = il.MakeComparable(obs,mod,
                                    mask_ref  = True,
                                    clip_ref  = True,
                                    extents   = self.extents,
                                    logstring = "[%s][%s]" % (self.longname,m.name))
        return obs,mod
