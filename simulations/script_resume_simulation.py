from simulations import *
from parcels import *
import os
import numpy as np
from datetime import timedelta
import netCDF4

# np.random.seed(1234)

# Open checkpoint file
nc_chk = netCDF4.Dataset("/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot/trajectories_100000p_0-30s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot.nc")


# Load preset simulation parameters
num_particles = nc_chk.variables["lon"].shape[0]

# Set manual simulation parameters
starttime = 0
motile = True
verbose = False
scale_fact = 1200
os.chdir("/media/alexander/AKC Passport 2TB/30-60/")
filenames = "F*n.nc_vort.123"
savepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot/trajectories_100000p_30-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot.nc"
runtime = timedelta(seconds=30)
dt = timedelta(seconds=0.01)
outputdt = timedelta(seconds=0.1)

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

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh,
                                timestamps=timestamps, interp_method=interp_method, field_chunksize=False)
# Implement field scaling.
logger.warning_once("Scaling factor set to %f - ensure this is correct." % scale_fact)
fieldset.U.set_scaling_factor(scale_fact)
fieldset.V.set_scaling_factor(scale_fact)
fieldset.W.set_scaling_factor(scale_fact)

# Make fieldset periodic.
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# Re-initiate particleset
time_index = 300
if verbose:
    pclass = Akashiwo3D_verbose
elif motile:
    pclass = Akashiwo3D
else:
    pclass = Generic3D
lon_init = nc_chk.variables["lon"][:, time_index]
lat_init = nc_chk.variables["lat"][:, time_index]
dep_init = nc_chk.variables["z"][:, time_index]
# temp_init = nc_chk.variables["temp"][:]
diam_init = nc_chk.variables["diameter"][:, time_index]
if motile:
    v_swim_init = nc_chk.variables["v_swim"][:, time_index]
    dir_x_init = nc_chk.variables["dir_x"][:, time_index]
    dir_y_init = nc_chk.variables["dir_y"][:, time_index]
    dir_z_init = nc_chk.variables["dir_z"][:, time_index]
    B_init = nc_chk.variables["B"][:, time_index]

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon_init, lat=lat_init, depth=dep_init,
                                 time=starttime, diameter=diam_init, v_swim=v_swim_init,
                                 dir_x=dir_x_init, dir_y=dir_y_init, dir_z=dir_z_init,
                                 B=B_init)
else:
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon_init, lat=lat_init, depth=dep_init,
                                 time=starttime, diameter=diam_init)

nc_chk.close()

# Initialise kernels
if motile:
    kernels = pset.Kernel(GyrotaxisRK4_3D_withTemp) + pset.Kernel(periodicBC)
else:
    kernels = pset.Kernel(AdvectionRK4_3D_withTemp) + pset.Kernel(periodicBC)

# Run simulation
pset.execute(kernels,
             runtime=runtime,
             dt=dt,
             recovery={ErrorCode.ErrorOutOfBounds: TopBottomBoundary},
             output_file=pset.ParticleFile(name=savepath, outputdt=outputdt)
             )

print("\nFinished with %d particles left at endtime." % pset.size)





