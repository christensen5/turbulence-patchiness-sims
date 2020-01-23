from simulations import *
from parcels import *
import os
import sys
import numpy as np
from datetime import timedelta
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD

timer.root = timer.Timer('Main')
timer.init = timer.Timer('Init', parent=timer.root)
timer.args = timer.Timer('Args', parent=timer.init)

# np.random.seed(1234)

# Set simulation parameters
num_particles = 100000
motile = False

filenames = "/media/alexander/AKC Passport 2TB/Maarten/sim021_b/F*n.nc.021"
#filenames = "/rds/general/user/akc17/home/WORK/sim022_vort/F*n.nc_vort.022"
savepath = "/home/alexander/Desktop/temp_results/MPI_trials/trajectories_" + str(num_particles) + "p_dead_MPIprofile"
# savepath = os.path.join(os.getcwd(), "trajectories_" + str(num_particles) + "p_30s_0.01dt_0.1sdt_initunif_dead_MPIprofile")

scale_fact = 5120./3  #1200
runtime = timedelta(seconds=1) #30
dt = timedelta(seconds=0.01)
outputdt = timedelta(seconds=0.1)
B = 2

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

timer.args.stop()
timer.fieldset = timer.Timer('FieldSet', parent=timer.init)

# chunksize = [50, 100, 500, 1000, 2000, 'auto'][int(sys.argv[1])-1]
chunksize = '1000'

# Initiate memory monitoring
ncores = comm.Get_size()
rank = comm.Get_rank()
pid = os.getpid()
memlog_savepath = "/home/alexander/Desktop/temp_results/MPI_trials/memuseage_%dp_%dchunksize_%d.%dprocs.npy" % (num_particles, chunksize, ncores, rank)
subprocess.Popen("python3.5 /home/alexander/Documents/turbulence-patchiness-sims/simulations/util/memory_monitor.py %s %s" % (pid, memlog_savepath), shell=True)

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps, interp_method=interp_method, field_chunksize=chunksize)

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

timer.fieldset.stop()
timer.psetargs = timer.Timer('ParticleSetArgs', parent=timer.init)
# Generate initial particle density field
pfield_grid_021 = RectilinearZGrid(lon=np.arange(0, 256), lat=np.arange(0, 256), depth=np.arange(0, 512), time=np.zeros(1), mesh='flat')
# pfield_grid = RectilinearZGrid(lon=np.arange(0, 720), lat=np.arange(0, 720), depth=np.arange(0, 360), time=np.zeros(1), mesh='flat')

pfield_uniform = uniform_init_field(pfield_grid_021)
# pfield_conc = conc_init_field(pfield_grid)

# Initiate particleset & kernels
pclass = Akashiwo3D if motile else Generic3D
timer.psetargs.stop()
timer.psetinit = timer.Timer('ParticleSetInit', parent=timer.init)
pset = ParticleSet.from_field(fieldset=fieldset,
                              pclass=pclass,
                              start_field=pfield_uniform,
                              depth=np.random.rand(num_particles)*180, #90+np.random.rand(num_particles)*60,
                              size=num_particles)
# Initialise custom particle variables.
if motile:
    swim_init = scale_fact * swim_speed_dist(pset.size, dist='/rds/general/user/akc17/home/packages/turbulence-patchiness-sims/simulations/util/swim_speed_distribution.csv')
# In MPI mode, we cannot index swim_init with particle.id since each process only gets a subset of the whole pset;
# instead we use a counter variable 'i'.
i = 0
for particle in pset:
    particle.diameter = scale_fact * np.random.uniform(0.000018, 0.000032)
    if motile:
        dir = rand_unit_vect_3D()
        particle.dir_x = dir[0]
        particle.dir_y = dir[1]
        particle.dir_z = dir[2]
        particle.v_swim = swim_init[i]
        particle.B = B
        i += 1

if motile:
    kernels = pset.Kernel(GyrotaxisRK4_3D_withTemp) + pset.Kernel(periodicBC)
else:
    kernels = pset.Kernel(AdvectionRK4_3D_withTemp) + pset.Kernel(periodicBC)
timer.psetinit.stop()
timer.init.stop()

timer.psetrun = timer.Timer('ParticleSetRun', parent=timer.root)
# Run simulation
pset.execute(kernels,
             runtime=runtime,
             dt=dt,
             recovery={ErrorCode.ErrorOutOfBounds: TopBottomBoundary},
             output_file=pset.ParticleFile(name=savepath, outputdt=outputdt)
             )
timer.psetrun.stop()
print("\nFinished with %d particles left at endtime." % pset.size)

# print("\nReformatting output for animations.")
# reformat_for_animate(savepath)

print("\nDone.\n")

timer.root.stop()
timer.root.print_tree()