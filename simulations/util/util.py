import numpy as np
import netCDF4
from glob import glob
import os


__all__ = ['rand_unit_vect_3D', 'extract_timestamps']


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
                t = t[0]

        timestamps.append(t)


    return np.array(timestamps)