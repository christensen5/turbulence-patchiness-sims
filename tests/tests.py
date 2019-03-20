"""Collection of testing scripts for turbulance-patchiness-sims model."""
import matplotlib
matplotlib.use('TkAgg')
from firedrake import *
from parcels import *
import unittest
import numpy as np
from tqdm import tqdm
import math

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt

class OfflineTPSTestCase(unittest.TestCase):
    def setUp(self):
        self.lon = np.linspace(0, 10, 11, dtype=np.float32)
        self.lat = np.linspace(0, 10, 11, dtype=np.float32)
        self.depth = np.linspace(0, 10, 11, dtype=np.float32)
        self.grid = RectilinearZGrid(lon=self.lon, lat=self.lat, depth=self.depth, time=np.zeros(1), mesh='flat')
        Ufield_init = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        Vfield_init = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        # self.fieldset = FieldSet(U=Ufield_init, V=Vfield_init)


    def test_kernel_Gyr_EE_3D_advection(self):
        """Test that the the advection portion of the 3D EE Gyrotaxis kernel is doing what it's expected to.
        We test by loading a basic flow in the X-direction and check the position of n particles
        (with random initial orientations and zero swim-speed) at each timestep relative to their expected position.
        Note that this is not a test of the accuracy of the advection scheme, just a test that it is actually advecting
        particles in the expected manner."""
        print("\nTesting EE Gyrotaxis kernel 3D (advection)")
        n = 10
        v_swim = 0.
        B = 2.
        self.Ufield = Field(name='U', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.Vfield = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.Wfield = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.fieldset = FieldSet(U=self.Ufield, V=self.Vfield, W=self.Wfield)
        self.particleset = ParticleSet.from_line(fieldset=self.fieldset, pclass=,
                                                 start=(0.01, 0.0), finish=(0.0, 10.0), size=n,
                                                 depth=np.linspace(0.0, 10, n))