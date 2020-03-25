"""Collection of testing scripts for turbulance-patchiness-sims model."""
import matplotlib
matplotlib.use('TkAgg')
from datetime import timedelta
from simulations import *
from parcels import *
import unittest
import numpy as np
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
        self.k = (0., 0., 1.)

    def test_kernel_top_bottom_boundary(self):
        """Test that the custom TopBottomBoundary kernel is correctly dealing with particles at the boundaries of the
        flow. We test by placing 4 particles to test the following cases:
        0) Above surface but no diameter. Should print "zero diameter" warning and place particle 1 cell below surface.
        1) Above surface with diameter. Should place particle 0.5 diameters below surface.
        2) Below floor. Should print "particle deleted" warning and delete particle.
        3) Out-of-bounds. Should print "particle deleted" warning and delete particle.
        """
        print("Testing TopBottom boundary.")
        self.kernel = AdvectionRK4_3D
        self.dt = 1
        self.runtime = 1
        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W})
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Generic3D,
                                                 lon=[5, 5, 5, 50],
                                                 lat=[5, 5, 5, 50],
                                                 depth=[-10, -10, 15, 5])
        self.particleset[1].diameter = 2

        self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                 recovery={ErrorCode.ErrorOutOfBounds: TopBottomBoundary})

        # Check particles have been correctly repositioned/deleted.
        p = self.particleset
        self.assertAlmostEqual(p[0].lon, 5, 6)
        self.assertAlmostEqual(p[0].lat, 5, 6)
        self.assertAlmostEqual(p[0].depth, 0.5, 6)
        self.assertAlmostEqual(p[1].lon, 5, 6)
        self.assertAlmostEqual(p[1].lat, 5, 6)
        self.assertAlmostEqual(p[1].depth, p[1].diameter/2., 6)
        self.assertEqual(p.size, 2)

    def test_kernel_Gyr_EE_3D_advection(self):
        """Test that the the advection portion of the 3D EE Gyrotaxis kernel is doing what it's expected to.
        We test by loading a basic flow in the X-direction and check the position of n particles
        (with random initial orientations and zero swim-speed) at each timestep relative to their expected position.
        Note that this is not a test of the accuracy of the advection scheme, just a test that it is actually advecting
        particles in the expected manner."""
        print("\nTesting EE Gyrotaxis kernel 3D (advection)")
        self.kernel = GyrotaxisEE_3D_withTemp
        self.dt = 0.1
        self.runtime = 0.1
        n = 10
        v_swim = 0.
        B = 2.
        self.U = Field(name='U', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                           grid=self.grid,
                           allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                           grid=self.grid,
                           allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                           grid=self.grid,
                           allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                                       'vort_X': self.vort_X,
                                                                       'vort_Y': self.vort_Y,
                                                                       'vort_Z': self.vort_Z,
                                                                       'Temp': self.Temp})
        self.particleset = ParticleSet.from_line(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                 start=(0.01, 0.0), finish=(0.0, 10.0), size=n,
                                                 depth=np.linspace(0.0, 10, n))

        for p in self.particleset:
            p.B = B
            p.v_swim = v_swim
            dir = rand_unit_vect_3D()
            p.dir_x = dir[0]
            p.dir_y = dir[1]
            p.dir_z = dir[2]
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.u, p.v, p.w = self.fieldset.UVW[0, p.depth, p.lat, p.lon]
            p.check_xu, p.check_yv, p.check_zw = ([p.lon, p.u], [p.lat, p.v], [p.depth, p.w])

        for steps in range(5):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
            for p in self.particleset:
                # Check that particle has moved by (v_flow * dt) in each direction.
                self.assertAlmostEqual(p.lon, p.check_xu[0] + p.check_xu[1] * self.dt, 6)
                self.assertAlmostEqual(p.lat, p.check_yv[0] + p.check_yv[1] * self.dt, 6)
                self.assertAlmostEqual(p.depth, p.check_zw[0] + p.check_zw[1] * self.dt, 6)
                # print("Particle %d at position (%f, %f, %f) sees velocities (%f, %f, %f)." % (p.id, p.lon, p.lat, p.depth, p.u, p.v, p.w))
                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.u, p.v, p.w = self.fieldset.UVW[0, p.depth, p.lat, p.lon]
                p.check_xu, p.check_yv, p.check_zw = ([p.lon, p.u], [p.lat, p.v], [p.depth, p.w])

        # #Plot particle trajectories (for diagnosing issues).
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_title("3D Particle Trajectories")
        # ax.set_xlabel("X - Longitude")
        # ax.set_ylabel("Y - Latitude")
        # ax.set_zlabel("Z - Depth")
        # colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        # for p in self.particleset:
        #     plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id])
        # plt.show()
        # plt.close()

    def test_kernel_Gyr_EE_3D_reorientation(self):
        """Test that the the Gyrotactic reorientation portion of the  3D EE Gyrotaxis kernel
        is doing what it's expected to. We test by loading a cubic mesh with a zero flow, and check the position and
        orientation of n particles at each timestep, relative to their expected
        positions and orientations. Note that this is not a test of the accuracy of the scheme, just a test that it is
        actually reorienting and 'swimming' particles in the expected manner."""
        print("\nTesting EE Gyrotaxis kernel 3D (reorientation)")
        self.kernel = GyrotaxisEE_3D_withTemp
        self.dt = 1
        self.runtime = 1
        n = 50
        v_swim = 0.1
        B = 2.
        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})
        lon = 5 + np.random.rand(n)
        lat = 5 + np.random.rand(n)
        depth = np.repeat(7.0, n)
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                 lon=lon, lat=lat, depth=depth)

        for p in self.particleset:
            p.B = B
            p.v_swim = v_swim
            offset_x = (np.random.rand(1)[0] * 0.1) - 0.05
            offset_y = (np.random.rand(1)[0] * 0.1) - 0.05
            dir = np.array((offset_x, offset_y, 1.0))
            dir = dir / np.linalg.norm(dir)
            p.dir_x = dir[0]
            p.dir_y = dir[1]
            p.dir_z = dir[2]

            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])
            p.check_x, p.check_y, p.check_z = (p.lon, p.lat, p.depth)
            p.check_dir_x, p.check_dir_y, p.check_dir_z = (p.dir_x, p.dir_y, p.dir_z)

        for steps in range(10):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
            for p in self.particleset:
                # Check that observed change of direction "diff" matches expected change "dp" (using dot product).
                if steps > 0:  # can't/don't check first step!
                    p_n = np.array((p.dx[-1], p.dy[-1], p.dz[-1]))
                    p_n1 = np.array((p.dir_x, p.dir_y, p.dir_z))
                    diff = (p_n1 - p_n) / self.dt
                    dp = 0.5 * (1 / p.B) * (self.k - (np.dot(self.k, p_n) * p_n))
                    self.assertAlmostEqual(np.dot(diff, dp), np.linalg.norm(diff) * np.linalg.norm(dp), 3)
                # Check that particles swim a distance (v_swim * dt) in their direction of orientation.
                check_pos = np.array((p.check_x + (p.dir_x * p.v_swim * self.dt),
                                      p.check_y + (p.dir_y * p.v_swim * self.dt),
                                      p.check_z + (p.dir_z * p.v_swim * self.dt)))
                self.assertAlmostEqual(p.lon, check_pos[0], 6)
                self.assertAlmostEqual(p.lat, check_pos[1], 6)
                self.assertAlmostEqual(p.depth, check_pos[2], 6)

                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)
                p.check_x, p.check_y, p.check_z = (p.lon, p.lat, p.depth)
                p.check_dir_x, p.check_dir_y, p.check_dir_z = (p.dir_x, p.dir_y, p.dir_z)

        #Plot particle trajectories (for diagnosing issues).
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title("3D Particle Trajectories")
        ax.set_xlabel("X - Longitude")
        ax.set_ylabel("Y - Latitude")
        ax.set_zlabel("Z - Depth")
        colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        m = 1
        for p in self.particleset:
            plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
            ax.quiver(p.x[::m], p.y[::m], p.z[::m],
                      p.dx[::m], p.dy[::m], p.dz[::m],
                      length=0.1, color='k')  # facing
        plt.show()
        plt.close()

    def test_kernel_Gyr_EE_3D_tumble_rot(self):
        """Test that the the Gyrotactic tumble portion of the particlespot 3D EE Gyrotaxis kernel is doing what it's
        expected to. We test first on a uniform-flow with vorticities set as a non-zero constant. We set
        particle swimming speed to zero (since we only care about vorticity-driven reorientation, and set B=0 so that
        particles do not re-orient towards the vertical.

        In the constant vorticity case, particles should maintain their orientation relative to
        each other while drifting in the flow.

        We test by checking the orientation of n particles at each timestep, relative to their expected orientations."""
        print("\nTesting EE Gyrotaxis kernel 3D (tumble_rot)")
        self.kernel = GyrotaxisEE_3D_withTemp
        self.dt = 0.1
        self.runtime = 0.1
        self.lon = np.linspace(0, 50, 51, dtype=np.float32)
        self.lat = np.linspace(0, 50, 51, dtype=np.float32)
        self.depth = np.linspace(0, 50, 51, dtype=np.float32)
        self.grid = RectilinearZGrid(lon=self.lon, lat=self.lat, depth=self.depth, time=np.zeros(1), mesh='flat')
        n = 10

        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})

        lon = np.linspace(4.0, 6.0, n)
        lat = np.ones(n) * 5
        depth = np.linspace(4.0, 6.0, n)
        # Set up rotational case.
        dir_rot = rand_unit_vect_3D()
        dir_x_rot = np.repeat(dir_rot[0], n)
        dir_y_rot = np.repeat(dir_rot[1], n)
        dir_z_rot = np.repeat(dir_rot[2], n)
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                 lon=lon, lat=lat, depth=depth,
                                                 dir_x=dir_x_rot, dir_y=dir_y_rot, dir_z=dir_z_rot)

        # Testing rotational case.
        for p in self.particleset:
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])
            p.check_dir_x, p.check_dir_y, p.check_dir_z = (p.dir_x, p.dir_y, p.dir_z)

        for steps_rot in range(50):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

            dir_x = self.particleset[0].dir_x
            dir_y = self.particleset[0].dir_y
            dir_z = self.particleset[0].dir_z

            for p in self.particleset:
                # Check particles have retained constant orientation relative to one another.
                self.assertAlmostEqual(p.dir_x, dir_x,
                                       msg='Particle %d x-direction failure at step %d.' % (p.id, steps_rot))
                self.assertAlmostEqual(p.dir_y, dir_y,
                                       msg='Particle %d y-direction failure at step %d.' % (p.id, steps_rot))
                self.assertAlmostEqual(p.dir_z, dir_z,
                                       msg='Particle %d z-direction failure at step %d.' % (p.id, steps_rot))

                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

        # # Plot particle trajectories (for diagnosing issues).
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_title("3D Particle Trajectories")
        # ax.set_xlabel("X - Longitude")
        # ax.set_ylabel("Y - Latitude")
        # ax.set_zlabel("Z - Depth")
        # colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        # m = 10
        # for p in self.particleset:
        #     plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
        #     ax.quiver(p.x[::m], p.y[::m], p.z[::m],
        #               p.dx[::m], p.dy[::m], p.dz[::m],
        #               length=0.5, color='k')
        # plt.show()
        # plt.close()

    def test_kernel_Gyr_EE_3D_tumble_irrot(self):
        """Test that the the Gyrotactic tumble portion of the 3D EE Gyrotaxis kernel is doing what it's
        expected to. We test second on an uniform flow with vorticity set to zero everywhere.
        We set particle swimming speed to zero (since we only care about vorticity-driven reorientation, and set
        B=0 so that particles do not re-orient towards the vertical.

        In the irrotational case, since vorticity is zero everywhere, particles should keep their initial orientation
        at all times.

        We test by checking the orientation of n particles at each timestep, relative to their expected orientations."""
        print("\nTesting EE Gyrotaxis kernel 3D (tumble_irrot)")
        self.kernel = GyrotaxisEE_3D_withTemp
        self.dt = 1.
        self.runtime = 1.
        self.lon = np.linspace(0, 50, 51, dtype=np.float32)
        self.lat = np.linspace(0, 50, 51, dtype=np.float32)
        self.depth = np.linspace(0, 50, 51, dtype=np.float32)
        self.grid = RectilinearZGrid(lon=self.lon, lat=self.lat, depth=self.depth, time=np.zeros(1), mesh='flat')
        n = 10

        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})

        lon = np.linspace(3.0, 7.0, n)
        lat = np.linspace(3.0, 7.0, n)
        depth = np.linspace(3.0, 7.0, n)
        # Set up rotational case.
        dir_rot = rand_unit_vect_3D()
        dir_x_rot = np.repeat(dir_rot[0], n)
        dir_y_rot = np.repeat(dir_rot[1], n)
        dir_z_rot = np.repeat(dir_rot[2], n)
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                       lon=lon, lat=lat, depth=depth)
        for p in self.particleset:
            p.v_swim = 0
            p.B = 0
            dir_init = rand_unit_vect_3D()
            p.dir_x = dir_init[0]
            p.dir_y = dir_init[1]
            p.dir_z = dir_init[2]

            # Testing rotational case.
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])

        for steps_irrot in range(40):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

            for p in self.particleset:
                if steps_irrot > 0:  # don't/can't check first step.
                    olddir = np.array((p.dx[-1], p.dy[-1], p.dz[-1]))
                    newdir = np.array((p.dir_x, p.dir_y, p.dir_z))
                    vort_p = (self.fieldset.vort_X[0, p.depth, p.lat, p.lon],
                              self.fieldset.vort_Y[0, p.depth, p.lat, p.lon],
                              self.fieldset.vort_Z[0, p.depth, p.lat, p.lon])
                    dp_exp = 0.5 * np.cross(vort_p, olddir)
                    dp = (newdir - olddir)/self.dt
                    # Check that particle orientation change matches expected theoretical value.
                    self.assertAlmostEqual(np.dot(dp_exp, dp), np.linalg.norm(dp_exp) * np.linalg.norm(dp), 4,
                                           msg='\nParticle %d orientation change at step %d does not match expected value.' % (
                                               p.id, steps_irrot))
                    # Check that particle orientation change is sufficiently close to zero (tolerance is somewhat lax
                    # in this case since the Firedrake curl field is not precisely zero).
                    self.assertTrue(np.allclose(olddir, newdir, 1e-1),
                                    msg='\nParticle %d orientation changed between steps %d-%d: p(%3.1f) = (%f, %f, %f) and p(%3.1f) = (%f, %f, %f).' % (
                                        p.id, steps_irrot-1, steps_irrot,
                                        self.dt * (steps_irrot), p.dx[-1], p.dy[-1], p.dz[-1],
                                        self.dt * (steps_irrot+1), p.dir_x, p.dir_y, p.dir_z))

                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

        # # Plot particle trajectories (for diagnosing issues).
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_title("3D Particle Trajectories")
        # ax.set_xlabel("X - Longitude")
        # ax.set_ylabel("Y - Latitude")
        # ax.set_zlabel("Z - Depth")
        # colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        # m = 10
        # for p in self.particleset:
        #     plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
        #     ax.quiver(p.x[::m], p.y[::m], p.z[::m],
        #               p.dx[::m], p.dy[::m], p.dz[::m],
        #               length=0.5, color='k')
        # plt.show()
        # plt.close()

    def test_kernel_Gyr_RK4_3D_reorientation(self):
        """
        Test that the the Gyrotactic reorientation portion of the 3D RK4 Gyrotaxis kernel
        is doing what it's expected to. We test with a zero flow, and check the position and
        orientation of n particles (with random initial orientations) at each timestep, relative to their expected
        positions and orientations. Note that this is not a test of the accuracy of the scheme, just a test that it is
        actually reorienting and 'swimming' particles in the expected manner.
        """
        print("\nTesting RK4 Gyrotaxis kernel 3D (reorientation)")
        self.kernel = GyrotaxisRK4_3D_withTemp
        n = 50
        self.dt = 1
        self.runtime = 1

        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})

        lon = 5 + np.random.rand(n)
        lat = 5 + np.random.rand(n)
        depth = np.repeat(7.0, n)
        v_swim = np.repeat(0.1, n)
        B = np.repeat(2, n)

        self.particleset_down = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                      lon=lon, lat=lat, depth=depth, v_swim=v_swim, B=B)
        for p in self.particleset_down:
            offset_x = (np.random.rand(1)[0] * 0.5) - 0.25
            offset_y = (np.random.rand(1)[0] * 0.5) - 0.25
            dir = np.array((offset_x, offset_y, 1.))
            dir = dir / np.linalg.norm(dir)
            p.dir_x = dir[0]
            p.dir_y = dir[1]
            p.dir_z = dir[2]

        self.particleset_up = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                    lon=lon, lat=lat, depth=depth, v_swim=v_swim, B=np.repeat(0, n))
        for p in self.particleset_up:
            dir = rand_unit_vect_3D()
            p.dir_x = dir[0]
            p.dir_y = dir[1]
            p.dir_z = dir[2]

        for p in self.particleset_down:
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])
        for p in self.particleset_up:
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])


        for steps in range(30):
            self.particleset_down.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
            self.particleset_up.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
            for p in self.particleset_down:
                # Check that observed change of direction "diff" matches expected change "dp" (using dot product).
                if steps > 0:  # can't/don't check first step!
                    p_n = np.array((p.dx[-1], p.dy[-1], p.dz[-1]))
                    p_n1 = np.array((p.dir_x, p.dir_y, p.dir_z))
                    diff = (p_n1 - p_n) / self.dt
                    dp = 0.5 * (1/p.B) * (self.k - (np.dot(self.k, p_n) * p_n))
                    self.assertAlmostEqual(np.dot(diff, dp), np.linalg.norm(diff)*np.linalg.norm(dp), 3)
                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

            for p in self.particleset_up:
                # Check that particles with B=0 swim a distance (v_swim * dt) per step in the direction of orientation.
                if steps > 0:  #can't/don't check first step!
                    x_n = np.array((p.x[-1], p.y[-1], p.z[-1]))
                    x_n1 = np.array((p.lon, p.lat, p.depth))
                    p_n = np.array((p.dx[-1], p.dy[-1], p.dz[-1]))
                    self.assertTrue(np.allclose(x_n + (p.v_swim * self.dt * p_n), x_n1, 1e-05))
                p.x.append(p.lon)
                p.y.append(p.lat)
                p.dz.append(p.dir_z)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

        # Plot particle trajectories (for diagnosing issues).
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title("3D Particle Trajectories")
        ax.set_xlabel("X - Longitude")
        ax.set_ylabel("Y - Latitude")
        ax.set_zlabel("Z - Depth")
        colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        m = 10
        for p in self.particleset_down:
            plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
            ax.quiver(p.x[::m], p.y[::m], p.z[::m],
                      p.dx[::m], p.dy[::m], p.dz[::m],
                      length=0.5, color='k')
        plt.show()
        plt.close()

    def test_kernel_Gyr_RK4_3D_tumble_rot(self):
        """Test that the the Gyrotactic tumble portion of the 3D RK4 Gyrotaxis kernel is doing what it's
        expected to. We test first on a We test first on a uniform-flow with vorticities set as a non-zero constant.
        We set particle swimming speed to zero (since we only care about vorticity-driven reorientation, and set B=0 so
        that particles do not re-orient towards the vertical.

        In the rotational case, vorticity is uniform, so particles should maintain their orientation relative to
        each other while drifting in the flow.

        We test by checking the orientation of n particles at each timestep, relative to their expected orientations."""
        print("\nTesting RK4 Gyrotaxis kernel 3D (tumble_rot)")
        self.kernel = GyrotaxisRK4_3D_withTemp
        self.dt = 0.1
        self.runtime = 0.1
        self.lon = np.linspace(0, 50, 51, dtype=np.float32)
        self.lat = np.linspace(0, 50, 51, dtype=np.float32)
        self.depth = np.linspace(0, 50, 51, dtype=np.float32)
        self.grid = RectilinearZGrid(lon=self.lon, lat=self.lat, depth=self.depth, time=np.zeros(1), mesh='flat')
        n = 10

        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})

        lon = np.linspace(4.0, 6.0, n)
        lat = np.ones(n) * 5
        depth = np.linspace(4.0, 6.0, n)
        # Set up rotational case.
        dir_rot = rand_unit_vect_3D()
        dir_x_rot = np.repeat(dir_rot[0], n)
        dir_y_rot = np.repeat(dir_rot[1], n)
        dir_z_rot = np.repeat(dir_rot[2], n)
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                 lon=lon, lat=lat, depth=depth,
                                                 dir_x=dir_x_rot, dir_y=dir_y_rot, dir_z=dir_z_rot)

        # Testing rotational case.
        for p in self.particleset:
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])
            p.check_dir_x, p.check_dir_y, p.check_dir_z = (p.dir_x, p.dir_y, p.dir_z)

        for steps_rot in range(50):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

            dir_x = self.particleset[0].dir_x
            dir_y = self.particleset[0].dir_y
            dir_z = self.particleset[0].dir_z

            for p in self.particleset:
                # Check particles have retained constant orientation relative to one another.
                self.assertAlmostEqual(p.dir_x, dir_x,
                                       msg='Particle %d x-direction failure at step %d.' % (p.id, steps_rot))
                self.assertAlmostEqual(p.dir_y, dir_y,
                                       msg='Particle %d y-direction failure at step %d.' % (p.id, steps_rot))
                self.assertAlmostEqual(p.dir_z, dir_z,
                                       msg='Particle %d z-direction failure at step %d.' % (p.id, steps_rot))

                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

        # # Plot particle trajectories (for diagnosing issues).
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_title("3D Particle Trajectories")
        # ax.set_xlabel("X - Longitude")
        # ax.set_ylabel("Y - Latitude")
        # ax.set_zlabel("Z - Depth")
        # colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        # m = 10
        # for p in self.particleset:
        #     plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
        #     ax.quiver(p.x[::m], p.y[::m], p.z[::m],
        #               p.dx[::m], p.dy[::m], p.dz[::m],
        #               length=0.5, color='k')
        # plt.show()
        # plt.close()

    def test_kernel_Gyr_RK4_3D_tumble_irrot(self):
        """Test that the the Gyrotactic tumble portion of the 3D RK4 Gyrotaxis kernel is doing what it's
        expected to. We test second on an uniform flow with vorticity set to zero everywhere.
        We set particle swimming speed to zero (since we only care about vorticity-driven reorientation, and set
        B=0 so that particles do not re-orient towards the vertical.

        In the irrotational case, since vorticity is zero everywhere, particles should keep their initial orientation
        at all times.

        We test by checking the orientation of n particles at each timestep, relative to their expected orientations."""
        print("\nTesting RK4 Gyrotaxis kernel 3D (tumble_irrot)")
        self.kernel = GyrotaxisRK4_3D_withTemp
        self.dt = 1.
        self.runtime = 1.
        self.lon = np.linspace(0, 50, 51, dtype=np.float32)
        self.lat = np.linspace(0, 50, 51, dtype=np.float32)
        self.depth = np.linspace(0, 50, 51, dtype=np.float32)
        self.grid = RectilinearZGrid(lon=self.lon, lat=self.lat, depth=self.depth, time=np.zeros(1), mesh='flat')
        n = 10

        self.U = Field(name='U', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.V = Field(name='V', data=np.ones([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.W = Field(name='W', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]), grid=self.grid,
                       allow_time_extrapolation=True)
        self.vort_X = Field(name='vort_X', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Y = Field(name='vort_Y', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.vort_Z = Field(name='vort_Z', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                            grid=self.grid,
                            allow_time_extrapolation=True)
        self.Temp = Field(name='Temp', data=np.zeros([len(self.depth), len(self.lat), len(self.lon)]),
                          grid=self.grid,
                          allow_time_extrapolation=True)

        self.fieldset = FieldSet(U=self.U, V=self.V, fields={'W': self.W,
                                                             'vort_X': self.vort_X,
                                                             'vort_Y': self.vort_Y,
                                                             'vort_Z': self.vort_Z,
                                                             'Temp': self.Temp})

        lon = np.linspace(3.0, 7.0, n)
        lat = np.linspace(3.0, 7.0, n)
        depth = np.linspace(3.0, 7.0, n)
        # Set up rotational case.
        dir_rot = rand_unit_vect_3D()
        dir_x_rot = np.repeat(dir_rot[0], n)
        dir_y_rot = np.repeat(dir_rot[1], n)
        dir_z_rot = np.repeat(dir_rot[2], n)
        self.particleset = ParticleSet.from_list(fieldset=self.fieldset, pclass=Akashiwo3D,
                                                 lon=lon, lat=lat, depth=depth)
        for p in self.particleset:
            p.v_swim = 0
            p.B = 0
            dir_init = rand_unit_vect_3D()
            p.dir_x = dir_init[0]
            p.dir_y = dir_init[1]
            p.dir_z = dir_init[2]

            # Testing rotational case.
            p.x, p.y, p.z = ([p.lon], [p.lat], [p.depth])
            p.dx, p.dy, p.dz = ([p.dir_x], [p.dir_y], [p.dir_z])

        for steps_irrot in range(40):
            self.particleset.execute(self.kernel, runtime=self.runtime, dt=self.dt,
                                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

            for p in self.particleset:
                if steps_irrot > 0:  # don't/can't check first step.
                    olddir = np.array((p.dx[-1], p.dy[-1], p.dz[-1]))
                    newdir = np.array((p.dir_x, p.dir_y, p.dir_z))
                    vort_p = (self.fieldset.vort_X[0, p.depth, p.lat, p.lon],
                              self.fieldset.vort_Y[0, p.depth, p.lat, p.lon],
                              self.fieldset.vort_Z[0, p.depth, p.lat, p.lon])
                    dp_exp = 0.5 * np.cross(vort_p, olddir)
                    dp = (newdir - olddir) / self.dt
                    # Check that particle orientation change matches expected theoretical value.
                    self.assertAlmostEqual(np.dot(dp_exp, dp), np.linalg.norm(dp_exp) * np.linalg.norm(dp), 4,
                                           msg='\nParticle %d orientation change at step %d does not match expected value.' % (
                                               p.id, steps_irrot))
                    # Check that particle orientation change is sufficiently close to zero (tolerance is somewhat lax
                    # in this case since the Firedrake curl field is not precisely zero).
                    self.assertTrue(np.allclose(olddir, newdir, 1e-1),
                                    msg='\nParticle %d orientation changed between steps %d-%d: p(%3.1f) = (%f, %f, %f) and p(%3.1f) = (%f, %f, %f).' % (
                                        p.id, steps_irrot - 1, steps_irrot,
                                        self.dt * (steps_irrot), p.dx[-1], p.dy[-1], p.dz[-1],
                                        self.dt * (steps_irrot + 1), p.dir_x, p.dir_y, p.dir_z))

                p.x.append(p.lon)
                p.y.append(p.lat)
                p.z.append(p.depth)
                p.dx.append(p.dir_x)
                p.dy.append(p.dir_y)
                p.dz.append(p.dir_z)

        # # Plot particle trajectories (for diagnosing issues).
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_title("3D Particle Trajectories")
        # ax.set_xlabel("X - Longitude")
        # ax.set_ylabel("Y - Latitude")
        # ax.set_zlabel("Z - Depth")
        # colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b', 'k', 'k']
        # m = 10
        # for p in self.particleset:
        #     plot = ax.plot(p.x, p.y, p.z, 'o-', linewidth=1, markersize=2, color=colors[p.id % 10])
        #     ax.quiver(p.x[::m], p.y[::m], p.z[::m],
        #               p.dx[::m], p.dy[::m], p.dz[::m],
        #               length=0.5, color='k')
        # plt.show()
        # plt.close()

def suite():
    suite = unittest.TestSuite()
    # Test utils and subroutines.
    # suite.addTest(OfflineTPSTestCase('test_kernel_top_bottom_boundary'))
    # suite.addTest(OfflineTPSTestCase('test_kernel_Gyr_EE_3D_advection'))
    suite.addTest(OfflineTPSTestCase('test_kernel_Gyr_EE_3D_reorientation'))
    # suite.addTest((OfflineTPSTestCase('test_kernel_Gyr_EE_3D_tumble_rot')))
    # suite.addTest((OfflineTPSTestCase('test_kernel_Gyr_EE_3D_tumble_irrot')))

    suite.addTest(OfflineTPSTestCase('test_kernel_Gyr_RK4_3D_reorientation'))
    # suite.addTest(OfflineTPSTestCase('test_kernel_Gyr_RK4_3D_tumble_rot'))
    # suite.addTest(OfflineTPSTestCase('test_kernel_Gyr_RK4_3D_tumble_irrot'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
