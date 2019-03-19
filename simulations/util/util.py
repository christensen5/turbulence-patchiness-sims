import numpy as np
import netCDF4
from glob import glob
from tqdm import tqdm
import os
import time
from parcels import Field


__all__ = ['rand_unit_vect_3D', 'extract_timestamps', 'find_max_velocities', 'extract_vorticities', 'uniform_init_field', 'conc_init_field']


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


    return np.array(timestamps)


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


def extract_vorticities(filepaths):
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
        nc_new.createVariable('vort_u', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_v', np.float32, ('Nz', 'Ny', 'Nx'))
        nc_new.createVariable('vort_w', np.float32, ('Nz', 'Ny', 'Nx'))

        nc_new.set_auto_mask(False)

        u = nc.variables["u"][:]
        v = nc.variables["v"][:]
        w = nc.variables["w"][:]
        time = nc.variables["time"][:]
        t01 = nc.variables["t01"][:]

        du_dz, du_dy, du_dx = np.gradient(u)
        dv_dz, dv_dy, dv_dx = np.gradient(v)
        dw_dz, dw_dy, dw_dx = np.gradient(w)

        vort_u = dw_dy - dv_dz
        vort_v = du_dz - dw_dx
        vort_w = dv_dx - du_dy

        nc_new.variables['u'][:] = u
        nc_new.variables['v'][:] = v
        nc_new.variables['w'][:] = w
        nc_new.variables['time'][:] = time
        nc_new.variables['t01'][:] = t01,
        nc_new.variables['vort_u'][:] = vort_u
        nc_new.variables['vort_v'][:] = vort_v
        nc_new.variables['vort_w'][:] = vort_w

        nc_new.sync()

        nc.close()
        nc_new.close()


def uniform_init_field(grid):
    """Produce a uniform distribution over the whole domain.

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
