from parcels import *
import netCDF4
from datetime import timedelta
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from timestamp_extractor import extract_timestamps

def compute_vorticities(nc):
    print("REMEMBER TO MAKE IT WRITE THE UPDATED NC FILE AT THE END\n")
    dx = 0.15
    dy = 0.15
    dz = 0.3
    shape = nc.variables['u'].shape
    # nc.variables["du_dy"] = np.zeros(shape)
    # nc.variables["du_dz"] = np.zeros(shape)
    # nc.variables["dv_dx"] = np.zeros(shape)
    nc.variables["dv_dz"] = np.zeros((250, shape[2]))#shape[0], shape[2]))
    # nc.variables["dw_dx"] = np.zeros(shape)
    nc.variables["dw_dy"] = np.zeros((250, shape[2]))#shape[0], shape[2]))
    nc.variables["vort_u"] = np.zeros((250, shape[2]))#shape[0], shape[2]))

    for y_ind in tqdm(range(1, shape[1] - 1)):
        for z_ind in range(1, 250-1):#shape[0]-1)):
            for x_ind in range(1, shape[2]-1):
                dv_dz = (nc.variables["v"][z_ind + 1, y_ind, x_ind] - nc.variables["v"][z_ind - 1, y_ind, x_ind]) / (2*dz)
                #nc.variables["dv_dz"][z_ind, x_ind] = dv_dz
                dw_dy = (nc.variables["w"][z_ind, y_ind + 1, x_ind] - nc.variables["w"][z_ind, y_ind - 1, x_ind]) / (2*dy)
                #nc.variables["dw_dy"][z_ind, x_ind] = dw_dy
                vort_u = dw_dy - dv_dz
                nc.variables["vort_u"][z_ind, x_ind] = vort_u

        plt.imshow(nc.variables['vort_u'], cmap="coolwarm", origin="lower", vmin=-0.0341726, vmax=0.0407044)
        plt.savefig("/home/alexander/Desktop/temp_maarten/vort_u" + "%03.0f" % y_ind + ".png")


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

os.chdir("/media/alexander/DATA/Ubuntu/Maarten/sim021_b")
# nc0 = netCDF4.Dataset("F0000291n.nc.021")
# nc1 = netCDF4.Dataset("F0010239n.nc.021")
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

filestring = "F*n.nc.021"
filenames = filestring
#filenames = {'U': filestring, 'V': filestring, 'W': filestring, 'time': filestring, 'Temp': filestring}
variables = {'U': 'u',
             'V': 'v',
             'W': 'w',
             'Temp': 't01'}
dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
mesh = 'flat'
timestamps = extract_timestamps(filestring) #np.linspace(0.25, 10, 40)
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps)
fieldset.U.set_scaling_factor(5120./3)
fieldset.V.set_scaling_factor(5120./3)
fieldset.W.set_scaling_factor(5120./3)
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# fieldset.lons = (fieldset.U.lon[0], fieldset.U.lon[-1])
# fieldset.lats = (fieldset.U.lat[0], fieldset.U.lat[-1])
# fieldset.depths = (fieldset.U.depth[0], fieldset.U.depth[-1])
mygrid = RectilinearZGrid(lon=np.arange(0, 265), lat=np.arange(0, 265), depth=np.arange(0, 512), time=np.zeros(1), mesh='flat')
mydata = np.ones((mygrid.ydim, mygrid.xdim))
#mydata[15:-15, 15:-15] = 1
pfield = Field(name='uniform_initial_dist', data=mydata, grid=mygrid)
#pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, lon=list(np.linspace(100, 200, 100)), lat=list(np.linspace(100, 200, 100)), depth=np.repeat(155, 100))

num_particles = 1000
pset = ParticleSet.from_field(fieldset=fieldset,
                              pclass=ScipyParticle,
                              start_field=pfield,
                              depth=np.random.rand(num_particles) * 350,#fieldset.U.depth[-1],
                              size=num_particles)

pset.execute(AdvectionRK4_3D + pset.Kernel(periodicBC),
             runtime=timedelta(seconds=3),
             dt=timedelta(seconds=0.25),
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
             output_file=pset.ParticleFile(name="/home/alexander/Desktop/temp_maarten/particle_test_period",
                                           outputdt=timedelta(seconds=0.25))
             )

print("There were %d particles left at endtime.\n" % pset.size)

# nc0.close()
# nc1.close()
# nc_example.close()
# nc_movv.close()

# print("Finished.")