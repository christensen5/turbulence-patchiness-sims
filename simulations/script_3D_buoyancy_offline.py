from simulations import *
from parcels import *
import os
import numpy as np
from datetime import timedelta

# np.random.seed(1234)

# Set simulation parameters
os.chdir("/media/alexander/AKC Passport 2TB/0-30/")
filenames = "F*n.nc_vort.123"
savepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-60s_0.01dt_0.1sdt_initunif_dead.nc2"
scale_fact = 1200 #5120./3
num_particles = 100000
runtime = timedelta(seconds=60.0)
dt = timedelta(seconds=0.01)
outputdt = timedelta(seconds=0.1)
B = 1.0
motile = False
verbose = False
time_periodic = False  # WARNING!!! DOESNT SEEM TO WORK WHEN TRUE

# Set up parcels objects.
timestamps = extract_timestamps(filenames)
variables = {'U': 'u', 'V': 'v', 'W': 'w', 'Temp': 't01'}
if motile:
    variables["vort_X"] = 'vort_x'
    variables["vort_Y"] = 'vort_y'
    variables["vort_Z"] = 'vort_z'
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

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps, time_periodic=time_periodic, interp_method=interp_method)

# Implement field scaling.
logger.warning_once("Scaling factor set to %f - ensure this is correct." %scale_fact)
fieldset.U.set_scaling_factor(scale_fact)
fieldset.V.set_scaling_factor(scale_fact)
fieldset.W.set_scaling_factor(scale_fact)

# Make fieldset periodic.
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# Generate initial particle density field
pfield_grid = RectilinearZGrid(lon=np.arange(0, 720), lat=np.arange(0, 720), depth=np.arange(0, 360), time=np.zeros(1), mesh='flat')

pfield_uniform = uniform_init_field(pfield_grid)
pfield_conc = conc_init_field(pfield_grid)

# Initiate particleset & kernels
if verbose:
    pclass = Akashiwo3D_verbose
elif motile:
    pclass = Akashiwo3D
else:
    pclass = Generic3D
pset = ParticleSet.from_field(fieldset=fieldset,
                              pclass=pclass,
                              start_field=pfield_uniform,
                              depth=180 + (np.random.rand(num_particles) * 180),
                              size=num_particles)

# Initialise custom particle variables.
if motile:
    swim_init = scale_fact * 1e-5  # scale factor * swim speed in metres
for particle in pset:
    particle.diameter = scale_fact * np.random.uniform(0.000018, 0.000032)
    if motile:
        dir = rand_unit_vect_3D()
        particle.dir_x = dir[0]
        particle.dir_y = dir[1]
        particle.dir_z = dir[2]
        particle.v_swim = swim_init
        particle.B = B

if motile:
    kernels = pset.Kernel(GyrotaxisRK4_3D_withTemp) + pset.Kernel(periodicBC)
else:
    kernels = pset.Kernel(AdvectionRK4_3D_withTemp) + pset.Kernel(periodicBC)
if verbose:
    kernels += pset.Kernel(Track_velocities) + pset.Kernel(Track_vorticities)

# Run simulation
pset.execute(kernels,
             runtime=runtime,
             dt=dt,
             recovery={ErrorCode.ErrorOutOfBounds: TopBottomBoundary},
             output_file=pset.ParticleFile(name=savepath, outputdt=outputdt)
             )

print("\nFinished with %d particles left at endtime." % pset.size)

# print("\nReformatting output for animations.")
# reformat_for_animate(savepath)

print("\nDone.\n")

