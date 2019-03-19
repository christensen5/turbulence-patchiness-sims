import netCDF4
import numpy as np
from tqdm import tqdm
from glob import glob
import os

all = ['reformat_for_animate', 'histogram_cell_velocities']


def reformat_for_animate(filepath):
    """
    A method to take the native parcels netcdf4 output file and reformat the particle position data to a matplotlib
    3D scatterplot-friendly format.

    :param filepath: path to the netCDF file containing particle trajectories WITHOUT THE .NC
    """

    particlefile = netCDF4.Dataset(str(filepath + ".nc"))

    num_frames = particlefile.variables["lon"].shape[1]

    lons = particlefile.variables["lon"][:][:, 0]
    lats = particlefile.variables["lat"][:][:, 0]
    deps = particlefile.variables["z"][:][:, 0]
    for f in tqdm(range(num_frames-1)):
        lons = np.vstack((lons, particlefile.variables["lon"][:][:, f + 1]))
        lats = np.vstack((lats, particlefile.variables["lat"][:][:, f + 1]))
        deps = np.vstack((deps, particlefile.variables["z"][:][:, f + 1]))

    np.save(filepath + '_lons.npy', lons)
    np.save(filepath + '_lats.npy', lats)
    np.save(filepath + '_deps.npy', deps)

    particlefile.close()

def histogram_cell_velocities(filepaths, n_bins):
    """
    A method to check the velocities in each cell of the fluid simulation and return a histogram of them, to help
    determining the Courant number.

    :param filepaths: String or list of strings representing the path(s) to the file(s).

    :return: H : array containing the values of the histogram. See density and weights for a description of the
    possible semantics.

    :return: bin_edges : array of dtype float containing the bin edges.
    """

    count = 0

    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths
    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))
    with tqdm(total=len(paths)) as pbar:
        for fp in paths:
            if not os.path.exists(fp):
                raise IOError("FieldSet file not found: %s" % str(fp))

            nc = netCDF4.Dataset(fp)

            vels = np.power(np.power(nc.variables["u"][:], 2) + np.power(nc.variables["v"][:], 2) + np.power(nc.variables["w"][:], 2), 0.5)
            if count==0:
                vmax = np.max(vels)

            nc.close()

            if count == 0:
                H, bin_edges = np.histogram(vels, bins=n_bins, range=(0, vmax))
            else:
                H += np.histogram(vels, bins=bin_edges)[0]

            count += 1
            pbar.update(1)

    return H, bin_edges

def find_cutoff(H, p):
...     cutoff = sum(H)*p
...     i = 0
...     diff = 0
...     diff = -1
...     while diff < 0:
...         diff = sum(H[:i])-cutoff
...         i+=1
...     return i
