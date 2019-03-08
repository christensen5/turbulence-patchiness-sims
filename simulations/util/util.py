import numpy as np
import netCDF4
from glob import glob
from tqdm import tqdm
import os


__all__ = ['rand_unit_vect_3D', 'extract_timestamps', 'find_max_velocities']


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
