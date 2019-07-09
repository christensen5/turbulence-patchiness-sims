from simulations import *
from parcels import *
import os
import numpy as np
from datetime import timedelta
import netCDF4

np.random.seed(1234)

# Open checkpoint file
nc_chk = netCDF4.Dataset("/rds/general/user/akc17/ephemeral/checkpoint_100000p_30s_0.01dt_0.05sdt_initunif_mot.nc")

# Load preset simulation parameters
num_particles = nc_chk.variables["lon"].shape[0]

# Set manual simulation parameters
starttime = 25.55
motile = True
scale_fact = 1200
filenames = "/rds/general/user/akc17/home/WORK/sim022_vort/F*n.nc_vort.022"
savepath = os.path.join(os.getcwd(), "trajectories_" + str(num_particles) + "p_30s_0.01dt_0.05sdt_initunif_mot_RESUME")
endtime = 30.0
dt = timedelta(seconds=0.01)
outputdt = timedelta(seconds=0.05)

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

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, mesh=mesh, timestamps=timestamps, interp_method=interp_method)
# Implement field scaling.
logger.warning_once("Scaling factor set to %f - ensure this is correct." % scale_fact)
fieldset.U.set_scaling_factor(scale_fact)
fieldset.V.set_scaling_factor(scale_fact)
fieldset.W.set_scaling_factor(scale_fact)
fieldset.vort_X.set_scaling_factor(scale_fact)
fieldset.vort_Y.set_scaling_factor(scale_fact)
fieldset.vort_Z.set_scaling_factor(scale_fact)

# Make fieldset periodic.
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
fieldset.add_periodic_halo(zonal=True, meridional=True, halosize=10)

# Re-initiate particleset
pclass = Akashiwo3D if motile else Generic3D
lon_init = nc_chk.variables["lon"][:, -1]
lat_init = nc_chk.variables["lat"][:, -1]
dep_init = nc_chk.variables["z"][:, -1]
temp_init = nc_chk.variables["temp"][:, -1]
diam_init = nc_chk.variables["diameter"][:, -1]
if motile:
    v_swim_init = nc_chk.variables["v_swim"][:, -1]
    dir_x_init = nc_chk.variables["dir_x"][:, -1]
    dir_y_init = nc_chk.variables["dir_y"][:, -1]
    dir_z_init = nc_chk.variables["dir_z"][:, -1]
    B_init = nc_chk.variables["B"][:, -1]

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon_init, lat=lat_init, depth=dep_init,
                                 time=starttime,
                                 temp=temp_init, diameter=diam_init, v_swim=v_swim_init,
                                 dir_x=dir_x_init, dir_y=dir_y_init, dir_z=dir_z_init,
                                 B=B_init)
else:
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=pclass,
                                 lon=lon_init, lat=lat_init, depth=dep_init,
                                 time=starttime,
                                 temp=temp_init, diameter=diam_init)

# Initialise kernels
if motile:
    kernels = pset.Kernel(GyrotaxisRK4_3D_withTemp) + pset.Kernel(periodicBC)
else:
    kernels = pset.Kernel(AdvectionRK4_3D_withTemp) + pset.Kernel(periodicBC)

# Run simulation
pset.execute(kernels,
             endtime=endtime,
             dt=dt,
             recovery={ErrorCode.ErrorOutOfBounds: TopBottomBoundary},
             output_file=pset.ParticleFile(name=savepath, outputdt=outputdt)
             )

print("\nFinished with %d particles left at endtime." % pset.size)





