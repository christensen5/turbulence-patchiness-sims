from simulations import *
from parcels import *
import os
import vg
import numpy as np
from datetime import timedelta

# np.random.seed(1234)

class SampleParticle(JITParticle):
    u = Variable("u", initial=-100)
    v = Variable("v", initial=-100)
    w = Variable("w", initial=-100)
    theta = Variable("theta", initial=-1)

# Set simulation parameters
os.chdir("/media/alexander/AKC Passport 2TB/0-60/")
filenames = "F*n.nc_vort.123"
savepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/velocities_around_particles_0-60s_1.0B_10um.nc"
# scale_fact = 1200 #5120./3

dt = timedelta(seconds=0)

motile = True
verbose = False
time_periodic = False  # WARNING!!! DOESNT SEEM TO WORK WHEN TRUE

# Set up parcels objects.
timestamps = extract_timestamps(filenames)
variables = {'U': 'u', 'V': 'v', 'W': 'w'}
# if motile:
#     variables["vort_X"] = 'vort_x'
#     variables["vort_Y"] = 'vort_y'
#     variables["vort_Z"] = 'vort_z'
dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
mesh = 'flat'
interp_method = {}
for v in variables:
    if v in ['U', 'V', 'W']:
        interp_method[v] = 'cgrid_velocity'
    elif v in ['vort_X', 'vort_Y', 'vort_Z']:
        interp_method[v] = 'linear'
    else:
        interp_method[v] = 'cgrid_tracer'

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps, time_periodic=time_periodic, interp_method=interp_method, field_chunksize=False)

# # Implement field scaling.
# logger.warning_once("Scaling factor set to %f - ensure this is correct." % scale_fact)
# fieldset.U.set_scaling_factor(scale_fact)
# fieldset.V.set_scaling_factor(scale_fact)
# fieldset.W.set_scaling_factor(scale_fact)

# Make fieldset periodic.
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# Generate particle set from previous simulation.
loadpath = os.path.dirname(savepath)
nc = netCDF4.Dataset("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot.nc")
lons = nc.variables["lon"][:].flatten()
lats = nc.variables["lat"][:].flatten()
deps = nc.variables["z"][:].flatten()
times = nc.variables["time"][:].flatten()
assert not np.any(np.isnan(lons)), "NaNs in previous simulation."
up = np.tile(np.array([0., 0., 1]), (lons.size, 1))
thetas = vg.angle(up, np.vstack((nc.variables["dir_x"][:].flatten(), nc.variables["dir_y"][:].flatten(), nc.variables["dir_z"][:].flatten())).transpose())
nc.close()

# Initiate particleset & kernels
pset = ParticleSet.from_list(fieldset, SampleParticle, lon=lons, lat=lats, depth=deps, time=times, theta=thetas)
kernels = pset.Kernel(Track_velocities)

# Run simulation
pset.execute(kernels, dt=0, output_file=ParticleFile(name=savepath), verbose_progress=True)

print("\nDone.\n")

