from simulations import *
from parcels import *
import os
import numpy as np
from datetime import timedelta

np.random.seed(1234)

# Set simulation parameters
os.chdir("/media/alexander/DATA/Ubuntu/Maarten/sim021_b") #("/media/alexander/AKC Passport 2TB/Maarten/sim022/")
filenames = "F*n.nc.021"
savepath = "/home/alexander/Desktop/temp_maarten/10x0.01"#os.path.join(os.getcwd(), "/data/trajectories")
scale_fact = 1200 #5120./3
num_particles = 100
runtime = timedelta(seconds=0.1)
dt = timedelta(seconds=0.01)
outputdt = timedelta(seconds=0.01)
motile = True

# Set up parcels objects.
timestamps = extract_timestamps(filenames)
variables = {'U': 'u', 'V': 'v', 'W': 'w', 'Temp': 't01'}
if motile:
    variables["Vort_X"] = 'vort_x'
    variables["Vort_Y"] = 'vort_y'
    variables["Vort_Z"] = 'vort_z'
dimensions = {'lon': 'Nx', 'lat': 'Ny', 'depth': 'Nz'}
mesh = 'flat'
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps)

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
pclass = Akashiwo3D
pset = ParticleSet.from_field(fieldset=fieldset,
                              pclass=pclass,
                              start_field=pfield_uniform,
                              depth=np.random.rand(num_particles) * 180,
                              size=num_particles)

# Initialise custom particle variables.
if motile:
    swim_init = swim_speed_dist(particleset.size, dist='$HOME/packages/turbulence-patchiness-sims/simulations/util/swim_speed_distribution.csv')
for particle in particleset:
    particle.diameter = np.random.uniform(0.018, 0.032)
    if motile:
        dir = rand_unit_vect_3D()
        particle.dir_x = dir[0]
        particle.dir_y = dir[1]
        particle.dir_z = dir[2]
        particle.v_swim = swim_init[particle.id]

kernels = pset.Kernel(AdvectionRK4_3D_withTemp) + pset.Kernel(periodicBC)

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

