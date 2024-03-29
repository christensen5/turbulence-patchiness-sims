import numpy as np
import netCDF4
from glob import glob
from tqdm import tqdm
import scipy.interpolate
import os
import csv
import time
from parcels import Field


__all__ = ['rand_unit_vect_3D', 'extract_timestamps', 'find_max_velocities', 'extract_vorticities_c_grid',
           'uniform_init_field', 'conc_init_field', 'swim_speed_dist', 'join_resumed_sims', 'checkclose_0_30_60']


def rand_unit_vect_3D():
    """ Generate a unit 3-vector with random direction."""
    xyz = np.random.normal(size=3)
    mag = sum(i**2 for i in xyz) ** .5
    return xyz / mag


def extract_timestamps(filepaths):
    """
    A basic method to open netCDF files wherein time is stored as a variable (rather than dimension) and to extract
    the timestamps into a numpy array.
    :param filepaths: String or list of strings representing the path(s) to the file(s).
    :return: timestamps: A numpy array containing the timestamps of each file.
    """
    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths

    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))

    timestamps = []
    for fp in paths:
        if not os.path.exists(fp):
            raise IOError("FieldSet file not found: %s" % str(fp))

        nc = netCDF4.Dataset(fp)
        t = nc.variables["time"][0]
        nc.close()

        if isinstance(t, list):
            t = t[0]
        if isinstance(t, np.ndarray):
            try:
                t.shape[1]
                t = t[0,0]
            except IndexError:
                t = np.float32(t)

        timestamps.append(t)


    return np.expand_dims(np.array(timestamps), axis=1)


def find_max_velocities(filepaths):
    """
    A basic method to open netCDF files containing velocity data and calculate the maximum velocities in the
    x, y and z direction, for use in computing the Courant number.
    :param filepaths: String or list of strings representing the path(s) to the file(s).
    :return: max_velocities: A 3D numpy array containing the maximum velocities in the x, y and z directions over the
    full set of data files.
    """
    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths

    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))

    max_velocities = np.zeros(3)

    for fp in tqdm(paths):
        if not os.path.exists(fp):
            raise IOError("FieldSet file not found: %s" % str(fp))

        nc = netCDF4.Dataset(fp)
        max_velocities[0] = max(np.max(abs(nc.variables["u"][:,:,:])), max_velocities[0])
        max_velocities[1] = max(np.max(abs(nc.variables["v"][:,:,:])), max_velocities[1])
        max_velocities[2] = max(np.max(abs(nc.variables["w"][:,:,:])), max_velocities[2])
        nc.close()

    return max_velocities


def extract_vorticities_OLD(filepaths):
    """
    A method to open netCDF files containing velocity data, calculate the vorticities in the
    x, y and z direction, and save the result as a new netCDF file.
    :param filepaths: String or list of strings representing the path(s) to the file(s).
    """
    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths

    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))

    for fp in tqdm(paths):
        if not os.path.exists(fp):
            raise IOError("FieldSet file not found: %s" % str(fp))

        nc = netCDF4.Dataset(fp)

        nc_new = netCDF4.Dataset(fp[:-4] + "_vort.022", "w", format="NETCDF4")
        nc_new.createDimension('Nz', nc.dimensions['Nz'].size)
        nc_new.createDimension('Ny', nc.dimensions['Ny'].size)
        nc_new.createDimension('Nx', nc.dimensions['Nx'].size)
        nc_new.createVariable('u', nc.variables["u"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('v', nc.variables["v"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('w', nc.variables["w"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('t01', nc.variables["t01"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('time', nc.variables["time"].dtype, ())
        nc_new.createVariable('vort_x', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_y', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_z', np.float32, ('Nz', 'Ny', 'Nx'))

        nc_new.set_auto_mask(False)

        u = nc.variables["u"][:]
        v = nc.variables["v"][:]
        w = nc.variables["w"][:]
        time = nc.variables["time"][:]
        t01 = nc.variables["t01"][:]

        du_dz, du_dy, du_dx = np.gradient(u)
        dv_dz, dv_dy, dv_dx = np.gradient(v)
        dw_dz, dw_dy, dw_dx = np.gradient(w)

        vort_x = dw_dy - dv_dz
        vort_y = du_dz - dw_dx
        vort_z = dv_dx - du_dy

        nc_new.variables['u'][:] = u
        nc_new.variables['v'][:] = v
        nc_new.variables['w'][:] = w
        nc_new.variables['time'][:] = time
        nc_new.variables['t01'][:] = t01,
        nc_new.variables['vort_x'][:] = vort_x
        nc_new.variables['vort_y'][:] = vort_y
        nc_new.variables['vort_z'][:] = vort_z

        nc_new.sync()

        nc.close()
        nc_new.close()


def extract_vorticities_c_grid(filepaths):
    """
    A method to open netCDF files containing velocity data stored on a C-grid, calculate the f-point vorticities in the
    x, y and z direction, and save the result as a new netCDF file.
    :param filepaths: String or list of strings representing the path(s) to the file(s).
    """
    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths

    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))

    for fp in tqdm(paths):
        if not os.path.exists(fp):
            raise IOError("FieldSet file not found: %s" % str(fp))

        nc = netCDF4.Dataset(fp)

        NxSize = nc.dimensions['Nx'].size
        NySize = nc.dimensions['Ny'].size
        NzSize = nc.dimensions['Nz'].size

        nc_new = netCDF4.Dataset(fp[:-4] + "_vort." + fp[-3:], "w", format="NETCDF4")
        nc_new.createDimension('Nz', nc.dimensions['Nz'].size)
        nc_new.createDimension('Ny', nc.dimensions['Ny'].size)
        nc_new.createDimension('Nx', nc.dimensions['Nx'].size)
        nc_new.createVariable('u', nc.variables["u"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('v', nc.variables["v"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('w', nc.variables["w"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('t01', nc.variables["t01"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('time', nc.variables["time"].dtype, ())
        nc_new.createVariable('p', nc.variables["p"].dtype, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_x', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_y', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_z', np.float32, ('Nz', 'Ny', 'Nx'))

        nc_new.set_auto_mask(False)

        u = nc.variables["u"][:]
        v = nc.variables["v"][:]
        w = nc.variables["w"][:]
        time = nc.variables["time"][:]
        t01 = nc.variables["t01"][:]
        p = nc.variables["p"][:]

        xb = nc.variables['xb'][:]  # in metres
        yb = nc.variables['yb'][:]
        zb = nc.variables['zb'][:]

        nc.close()

        dx = xb[1] - xb[0]
        dy = yb[1] - yb[0]
        dz = zb[1] - zb[0]

        scale_fact = 1200  # 5120./3  # Convert m/s to cells/s

        xsi = 0.
        eta = 0.
        zeta = 0.5

        U0 = u[:, :, :-1] * scale_fact
        # U1 = u[:, :, 1:] * scale_fact
        V0 = v[:, :-1, :] * scale_fact
        # V1 = v[:, 1:, :] * scale_fact
        W0 = w[:-1, :, :] * scale_fact
        W1 = w[1:, :, :] * scale_fact

        # Velocities at f-points (according to https://www.nemo-ocean.eu/doc/node19.html), i.e. (x_b, y_b, z_c) gridpoints.
        uf = (1 - xsi) * U0  # + xsi * U1
        vf = (1 - eta) * V0  # + eta * V1
        wf = (1 - zeta) * W0 + zeta * W1

        # Return to original shape (we lose one slice in the interpolation between e.g. U_i and U_i+1)
        uf = np.concatenate((uf, np.reshape(uf[:, :, -1], (NzSize,
                                                           NySize,
                                                           1))), 2)

        vf = np.concatenate((vf, np.reshape(vf[:, -1, :], (NzSize,
                                                           1,
                                                           NxSize))), 1)

        wf = np.concatenate((wf, np.reshape(wf[-1, :, :], (1,
                                                           NySize,
                                                           NxSize))), 0)

        # Convert back to m/s from cells/s
        uf = uf / scale_fact
        vf = vf / scale_fact
        wf = wf / scale_fact

        # Compute gradients at f-points
        du_dz, du_dy, du_dx = np.gradient(uf, dz, dy, dx)
        dv_dz, dv_dy, dv_dx = np.gradient(vf, dz, dy, dx)
        dw_dz, dw_dy, dw_dx = np.gradient(wf, dz, dy, dx)

        vort_x = dw_dy - dv_dz
        vort_y = du_dz - dw_dx
        vort_z = dv_dx - du_dy

        nc_new.variables['u'][:] = u
        nc_new.variables['v'][:] = v
        nc_new.variables['w'][:] = w
        nc_new.variables['time'][:] = time
        nc_new.variables['t01'][:] = t01
        nc_new.variables["p"][:] = p
        nc_new.variables['vort_x'][:] = vort_x
        nc_new.variables['vort_y'][:] = vort_y
        nc_new.variables['vort_z'][:] = vort_z

        nc_new.sync()

        nc_new.close()


def uniform_init_field(grid):
    """Produce a uniform distribution over the whole sim022 domain.

    :param grid: Grid on which to define distribution.

    :returns pfield: OceanParcels Field object containing the distribution.
    """
    uniform_data = np.ones((grid.ydim, grid.xdim))
    pfield = Field(name='uniform_initial_dist', data=uniform_data, grid=grid)

    return pfield


def conc_init_field(grid):
    """Produce a uniform distribution over the whole domain.

    :param grid: Grid on which to define distribution.

    :returns pfield: OceanParcels Field object containing the distribution.
    """
    conc_data = np.zeros((grid.ydim, grid.xdim))
    conc_data[150:210, 150:210] = np.ones((60, 60))
    conc_data[330:390, 330:390] = np.ones((60, 60))
    conc_data[510:570, 510:570] = np.ones((60, 60))
    pfield = Field(name='concentrated_initial_dist', data=conc_data, grid=grid)


    return pfield


def swim_speed_dist(num_particles, dist='simulations/util/swim_speed_distribution.csv'):
    """Produce a random swim speed for each particle based on the swim speed distribution for H. Akashiwo given in
    [Durham2013]"""
    # Import histogram (contains particle speed dist in um/s)
    with open(dist, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        bins = []
        counts = []
        for row in reader:
            bins.append(row[0])
            counts.append(row[1])

    # Generate PDF
    cum_counts = np.cumsum(counts)
    bin_width = 3
    x = cum_counts * bin_width
    y = bins
    pdf = scipy.interpolate.interp1d(x, bins)
    b = np.zeros(num_particles)
    for i in range(len(b)):
        u = np.random.uniform(x[0], x[-1])
        b[i] = pdf(u) / 1000000  # convert um -> m

    return b


def join_resumed_sims(filepaths, savepath):
    """
    A method to aggregate two or more time-contiguous simulations into a single netCDF file.
    :param filepaths: List of paths to files to aggregate.

    :param savepath: Path to save aggregated file.
    """
    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths
    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))

    # extract variable names from first simulation.
    nc_first = netCDF4.Dataset(paths[0])
    variables_dict = {key: None for key in nc_first.variables.keys()}  # this dict of variables will later be used to create a new netCDF dataset with the combined data from all sims

    for var in list(variables_dict.keys()):
        variables_dict[var] = nc_first.variables[var][:]  # first fill the dict with the data from the first sim

    for fp in tqdm(paths[1:]):
        if not os.path.exists(fp):
            raise IOError("NetCDF file not found: %s" % str(fp))

        nc = netCDF4.Dataset(fp)
        for var in list(variables_dict.keys()):
            # ignore variables introduced in later sims but absent from the first.
            try:
                nc.variables[var]
            except KeyError:
                print("NetCDF file %s did not contain a variable called %s" % (str(fp), var))
                continue
            if len(nc.variables[var].shape) == 1:  # ignore variables with values saved only at start of sim.
                continue
            else:
                #nc_new.variables[var][:] = np.concatenate((nc_new.variables[var][:], nc.variables[var][:]), axis=1)
                variables_dict[var] = np.concatenate((variables_dict[var], nc.variables[var][:][:, 1:]), axis=1)  # now append each later sim's data to the dict
        nc.close()

    nc_new = netCDF4.Dataset(savepath, "w", format="NETCDF4")
    nc_new.createDimension('obs', None)
    nc_new.createDimension('traj', None)
    nc_new.set_auto_mask(False)
    for var in list(variables_dict.keys()):
        nc_new.createVariable(var, nc_first.variables[var].dtype, nc_first.variables[var].dimensions, fill_value=nc_first.variables[var]._FillValue)
        nc_new.variables[var][:] = variables_dict[var]

    nc_first.close()
    nc_new.sync()
    nc_new.close()


def checkclose_0_30_60(sim030, sim060, sim3060, motile=False):
    """A function to ensure that two simulations (0-30s & 30-60s) joined by join_resumed_sims() have indeed been
    correctly joined together.
    :param sim030: Path to .nc file containing the 0-30s simulation.
    :param sim060: Path to .nc file containing the 0-60s simulation.
    :param sim3060: Path to .nc file containing the 30-60s simulation.

    :param motile: Boolean variable describing whether the simulations involve motile or non-motile particles.
    """

    # load sims.
    nc030 = netCDF4.Dataset(sim030)
    nc060 = netCDF4.Dataset(sim060)
    nc3060 = netCDF4.Dataset(sim3060)

    # define the simulation variables that will be checked.
    keys = ['lon', 'lat', 'z', 'dir_x', 'dir_y', 'dir_z'] if motile else ['lon', 'lat', 'z']
    for key in keys:
        try:
            assert (
                np.ma.allclose(nc030.variables[key][:], nc060.variables[key][:][:, 0:nc030.variables[key][:].shape[1]]))
        except AssertionError:
            print("Error with variable %s in 0-30s." % key)
        except KeyError:
            print("KeyError: %s is not a variable in either sim030 or sim060 or both." % key)
        else:
            print("No error with variable %s in 0-30s." % key)
        try:
            assert (
                np.ma.allclose(nc3060.variables[key][:], nc060.variables[key][:][:, -nc3060.variables[key][:].shape[1]:]))
        except AssertionError:
            print("Error with variable %s in 30-60s." % key)
        except KeyError:
            print("KeyError: %s is not a variable in either sim030 or sim3060 or both." % key)
        else:
            print("No error with variable %s in 30-60s." % key)

    nc030.close()
    nc060.close()
    nc3060.close()


if __name__ == "__main__":

    # nc = "/media/alexander/AKC Passport 2TB/F000090s.nc.123"
    # extract_vorticities_c_grid(nc)

    # input1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/10000p_15s_0.01dt_0.1sdt_initunif_mot_tloop.nc"
    # input2 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/10000p_15s_0.01dt_0.1sdt_initunif_mot_tloop_RESUME.nc"
    # output = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/10000p_30s_0.01dt_0.1sdt_initunif_mot_tloop_JOINED.nc"
    #
    # join_resumed_sims([input1, input2], output)

    checkclose_0_30_60(
        '/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-45s_0.01dt_0.1sdt_initunif_dead.nc',
        '/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-60s_0.01dt_0.1sdt_initunif_dead.nc',
        '/media/alexander/DATA/Ubuntu/Maarten/outputs/sim123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_30-60s_0.01dt_0.1sdt_initunif_dead.nc',
        True)


