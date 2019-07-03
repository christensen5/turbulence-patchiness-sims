from parcels import *
import netCDF4
from datetime import timedelta
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from simulations import extract_timestamps

def compute_vorticities(filenames):
    timestamps = extract_timestamps(filenames)
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
    mesh = 'flat'
    interp_method = {}
    for v in variables:
        interp_method[v] = 'cgrid_velocity'
    fieldset_vel = FieldSet.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions,
                                                mesh=mesh, timestamps=timestamps, interp_method=interp_method)
    # # Make fieldset periodic.
    # fieldset_vel.add_constant('halo_west', fieldset_vel.U.grid.lon[0])
    # fieldset_vel.add_constant('halo_east', fieldset_vel.U.grid.lon[-1])
    # fieldset_vel.add_constant('halo_south', fieldset_vel.U.grid.lat[0])
    # fieldset_vel.add_constant('halo_north', fieldset_vel.U.grid.lat[-1])
    # fieldset_vel.add_periodic_halo(zonal=True, meridional=True)
    fieldset_vel.check_complete()

    nc = netCDF4.Dataset(filenames)

    xb = nc.variables['xb'][:]
    yb = nc.variables['yb'][:]
    zc = nc.variables['zc'][:]

    dx = xb[1] - xb[0]
    dy = yb[1] - yb[0]
    dz = zc[1] - zc[0]

    # Velocities at f-points (according to https://www.nemo-ocean.eu/doc/node19.html), i.e. (x_b, y_b, z_c) gridpoints.
    shape = (len(zc), len(yb), len(xb))
    uf = np.zeros(shape)  # z, y, x
    vf = np.zeros(shape)
    wf = np.zeros(shape)
    for i in tqdm(range(1, len(zc)-1)):  # ignore first and last entries since they correspond to out-of-bounds cells.
        for j in range(1, len(yb)-1):
            for k in range(1, len(xb)-1):
                # ub = 0.5 * (fieldset_vel.U.data[0, k, j+1, i+1] + fieldset_vel.U.data[0, k+1, j+1, i+1])
                # ub_check = fieldset_vel.U.eval(timestamps[0], k+1, j+1, i+1)
                # vb = 0.5 * (fieldset_vel.V.data[0, k, j+1, i+1] + fieldset_vel.V.data[0, k+1, j+1, i+1])
                # vb_check = fieldset_vel.V.eval(timestamps[0], k+1, j+1, i+1)
                # wb = 0.5 * (fieldset_vel.W.data[0, k, j+1, i+1] + fieldset_vel.W.data[0, k+1, j+1, i+1])
                # wb_check = fieldset_vel.W.eval(timestamps[0], k+1, j+1, i+1)
                utmp, vtmp, wtmp = fieldset_vel.UVW.eval(timestamps[0], zc[i], yb[j], xb[k])
                uf[i, j, k] = utmp
                vf[i, j, k] = vtmp
                wf[i, j, k] = wtmp

    for array in (uf, vf, wf):
        array[0, :, :]  = array[1, :, :]
        array[-1, :, :] = array[-2, :, :]

        array[:, 0, :]  = array[:, 1, :]
        array[:, -1, :] = array[:, -2, :]

        array[:, :, 0]  = array[:, :, 1]
        array[:, :, -1] = array[:, :, -2]

    du_dy, du_dz = np.gradient(uf, dx=dx, dy=dy, dz=dz, axis=(1, 0))
    dv_dx, dv_dz = np.gradient(vf, dx=dx, dy=dy, dz=dz, axis=(2, 0))
    dw_dx, dw_dy = np.gradient(wf, dx=dx, dy=dy, dz=dz, axis=(2, 1))

    vort_x = dw_dy - dv_dz
    vort_y = du_dz - dw_dx
    vort_z = dv_dx - du_dy


def compute_vorticities_fast(filenames):
    timestamps = extract_timestamps(filenames)
    variables = {'U': 'u', 'V': 'v', 'W': 'w'}
    dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
    mesh = 'flat'
    interp_method = {}
    for v in variables:
        interp_method[v] = 'cgrid_velocity'
    fieldset_vel = FieldSet.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions,
                                                mesh=mesh, timestamps=timestamps, interp_method=interp_method)
    # # Make fieldset periodic.
    # fieldset_vel.add_constant('halo_west', fieldset_vel.U.grid.lon[0])
    # fieldset_vel.add_constant('halo_east', fieldset_vel.U.grid.lon[-1])
    # fieldset_vel.add_constant('halo_south', fieldset_vel.U.grid.lat[0])
    # fieldset_vel.add_constant('halo_north', fieldset_vel.U.grid.lat[-1])
    # fieldset_vel.add_periodic_halo(zonal=True, meridional=True)
    fieldset_vel.check_complete()

    nc = netCDF4.Dataset(filenames)

    xb = nc.variables['xb'][:]  # in metres
    yb = nc.variables['yb'][:]
    zb = nc.variables['zb'][:]

    dx = xb[1] - xb[0]
    dy = yb[1] - yb[0]
    dz = zb[1] - zb[0]

    scale_fact = 1200  # 5120./3  # Convert m/s to cells/s

    xsi = 0.
    eta = 0.
    zeta = 0.5

    U0 = nc.variables['u'][:][:, :, :-1] * scale_fact
    # U1 = nc.variables['u'][:][:, :, 1:] * scale_fact
    V0 = nc.variables['v'][:][:, :-1, :] * scale_fact
    # V1 = nc.variables['v'][:][:, 1:, :] * scale_fact
    W0 = nc.variables['w'][:][:-1, :, :] * scale_fact
    W1 = nc.variables['w'][:][1:, :, :] * scale_fact

    # Velocities at f-points (according to https://www.nemo-ocean.eu/doc/node19.html), i.e. (x_b, y_b, z_c) gridpoints.
    uf = (1-xsi) * U0# + xsi * U1
    vf = (1-eta) * V0# + eta * V1
    wf = (1-zeta) * W0 + zeta * W1

    # Return to original shape (we lose one slice in the interpolation between e.g. U_i and U_i+1)
    uf = np.concatenate((uf, np.reshape(uf[:, :, -1], (362, 722, 1))), 2)
    vf = np.concatenate((vf, np.reshape(vf[:, -1, :], (362, 1, 722))), 1)
    wf = np.concatenate((wf, np.reshape(wf[-1, :, :], (1, 722, 722))), 0)

    # Convert back to m/s from cells/s
    uf = uf / scale_fact
    vf = vf / scale_fact
    wf = wf / scale_fact

    # Compute gradients
    du_dz, du_dy, du_dx = np.gradient(uf, dz, dy, dx)
    dv_dz, dv_dy, dv_dx = np.gradient(vf, dz, dy, dx)
    dw_dz, dw_dy, dw_dx = np.gradient(wf, dz, dy, dx)

    vort_x = dw_dy - dv_dz
    vort_y = du_dz - dw_dx
    vort_z = dv_dx - du_dy



def DeleteParticle(particle, fieldset, time):  # delete particles who run out of bounds.
    print("Particle %d deleted at (%f, %f, %f)" % (particle.id, particle.lon, particle.lat, particle.depth))
    particle.delete()


def TopBottomBoundary(particle, fieldset, time):  # delete particles who run out of bounds.
    if particle.depth < 0:
        particle.depth = particle.diameter/2 if particle.diameter is not None else 0.1
    if particle.depth > fieldset.U.grid.depth[-1]:
        print("Particle %d deleted at (%f, %f, %f)" % (particle.id, particle.lon, particle.lat, particle.depth))
        particle.delete()


def periodicBC(particle, fieldset, time):
    # longitudinal/X boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west
    # latitudinal/Y boundary
    if particle.lat < fieldset.halo_south:
        particle.lat += fieldset.halo_north - fieldset.halo_south
    elif particle.lat > fieldset.halo_north:
        particle.lat -= fieldset.halo_north - fieldset.halo_south

np.random.seed(1234)

# nc_example = netCDF4.Dataset("/home/alexander/parcels/parcels/examples/GlobCurrent_example_data/20020101000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")

os.chdir("/media/alexander/AKC Passport 2TB/Maarten/sim022")

t0 = time.time()
compute_vorticities_fast("F0000168n.nc.022")
t1 = time.time()

print("\n %f s" % (t1-t0))

# nc1 = netCDF4.Dataset("F0000168n.nc.022")
# nc_movv = netCDF4.Dataset("movv_u.nc.021")

#==============================================
# Compute first dimension of vorticity to check against movv file.
# compute_vorticities(nc0)

#==============================================
# Random experimenting
# nc_u = netCDF4.Dataset("movv_u.nc.021")
# nc_v = netCDF4.Dataset("movv_v.nc.021")
# nc_w = netCDF4.Dataset("movv_w.nc.021")
#
# u0 = nc_u.variables['u'][0]
# v0 = nc_v.variables['v'][0]
# w0 = nc_w.variables['w'][0]


#=========================================================
# Getting nc file into Parcels FieldSet
# filenames_ex = "/home/alexander/parcels/parcels/examples/GlobCurrent_example_data/20*.nc"
# variables_ex = {'U': 'eastward_eulerian_current_velocity',
#              'V': 'northward_eulerian_current_velocity'}
# dimensions_ex = {'lat': 'lat',
#               'lon': 'lon',
#               'time': 'time'}
# fieldset_ex = FieldSet.from_netcdf(filenames_ex, variables_ex, dimensions_ex)

# filestring = "F*n.nc.021"
# filenames = filestring
# #filenames = {'U': filestring, 'V': filestring, 'W': filestring, 'time': filestring, 'Temp': filestring}
# variables = {'U': 'u',
#              'V': 'v',
#              'W': 'w',
#              'Temp': 't01'}
# dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
# mesh = 'flat'
# timestamps = extract_timestamps(filestring) #np.linspace(0.25, 10, 40)
# fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps)
# fieldset.U.set_scaling_factor(5120./3)
# fieldset.V.set_scaling_factor(5120./3)
# fieldset.W.set_scaling_factor(5120./3)
# fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
# fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
# fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
# fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
# fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# fieldset.lons = (fieldset.U.lon[0], fieldset.U.lon[-1])
# fieldset.lats = (fieldset.U.lat[0], fieldset.U.lat[-1])
# fieldset.depths = (fieldset.U.depth[0], fieldset.U.depth[-1])
# mygrid = RectilinearZGrid(lon=np.arange(0, 265), lat=np.arange(0, 265), depth=np.arange(0, 512), time=np.zeros(1), mesh='flat')
# mydata = np.ones((mygrid.ydim, mygrid.xdim))
#mydata[15:-15, 15:-15] = 1
# pfield = Field(name='uniform_initial_dist', data=mydata, grid=mygrid)
#pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, lon=list(np.linspace(100, 200, 100)), lat=list(np.linspace(100, 200, 100)), depth=np.repeat(155, 100))
#
# num_particles = 1000
# pset = ParticleSet.from_field(fieldset=fieldset,
#                               pclass=ScipyParticle,
#                               start_field=pfield,
#                               depth=np.random.rand(num_particles) * 350,#fieldset.U.depth[-1],
#                               size=num_particles)
#
# pset.execute(AdvectionRK4_3D + pset.Kernel(periodicBC),
#              runtime=timedelta(seconds=3),
#              dt=timedelta(seconds=0.25),
#              recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
#              output_file=pset.ParticleFile(name="/home/alexander/Desktop/temp_maarten/particle_test_period",
#                                            outputdt=timedelta(seconds=0.25))
#              )
#
# print("There were %d particles left at endtime.\n" % pset.size)

# nc0.close()
# nc1.close()
# nc_example.close()
# nc_movv.close()

# print("Finished.")