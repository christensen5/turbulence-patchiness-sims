import numpy as np
from parcels.particle import Variable, ScipyParticle, JITParticle
from parcels.tools.loggers import warning_once

__all__ = ['Generic3D', 'Akashiwo3D', 'Akashiwo3D_verbose']


class Generic3D(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=0., to_write=False)
    diameter = Variable('diameter', dtype=np.float32, initial=None)#, to_write='once')


class Akashiwo3D(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=0., to_write=False)
    diameter = Variable('diameter', dtype=np.float32, initial=None)#, to_write='once')
    v_swim = Variable('v_swim', dtype=np.float32, initial=0.)#, to_write='once')
    dir_x = Variable('dir_x', dtype=np.float32, initial=0.)
    dir_y = Variable('dir_y', dtype=np.float32, initial=0.)
    dir_z = Variable('dir_z', dtype=np.float32, initial=0.)
    B = Variable('B', dtype=np.float32, initial=None)#, to_write='once')


class Akashiwo3D_verbose(JITParticle):
    temp = Variable('temp', dtype=np.float32, initial=0., to_write=False)
    diameter = Variable('diameter', dtype=np.float32, initial=None, to_write='once')
    v_swim = Variable('v_swim', dtype=np.float32, initial=0., to_write='once')
    dir_x = Variable('dir_x', dtype=np.float32, initial=0.)
    dir_y = Variable('dir_y', dtype=np.float32, initial=0.)
    dir_z = Variable('dir_z', dtype=np.float32, initial=0.)
    B = Variable('B', dtype=np.float32, initial=None, to_write='once')
    u = Variable('u', dtype=np.float32, initial=0.)
    v = Variable('v', dtype=np.float32, initial=0.)
    w = Variable('w', dtype=np.float32, initial=0.)
    vort_x = Variable('vort_x', dtype=np.float32, initial=0.)
    vort_y = Variable('vort_y', dtype=np.float32, initial=0.)
    vort_z = Variable('vort_z', dtype=np.float32, initial=0.)
