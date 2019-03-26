import numpy as np
from parcels.particle import Variable, ScipyParticle, JITParticle
from parcels.tools.loggers import warning_once

__all__ = ['Generic3D', 'Akashiwo3D']


class Generic3D(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=0.)
    diameter = Variable('diameter', dtype=np.float32, initial=None)


class Akashiwo3D(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=0.)
    diameter = Variable('diameter', dtype=np.float32, initial=None)
    v_swim = Variable('v_swim', dtype=np.float32, initial=0.)
    dir_x = Variable('dir_x', dtype=np.float32, initial=0.)
    dir_y = Variable('dir_y', dtype=np.float32, initial=0.)
    dir_z = Variable('dir_z', dtype=np.float32, initial=0.)
    B = Variable('B', dtype=np.float32, initial=2.0)