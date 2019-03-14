import netCDF4
import numpy as np
from tqdm import tqdm

all = ['reformat_for_animate']


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

