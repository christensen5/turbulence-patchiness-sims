import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
from tqdm import tqdm
import time
import sys
import os

def calc_kde(data):
    return kde(data.T)

timestamps = np.linspace(0, 300, 31, dtype=int).tolist()
filepath = "/home/alexkc17/densities/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot"

# verify timestamp format
if isinstance(timestamps, int) or isinstance(timestamps, float):
    timestamps = [timestamps]

# load data
lons = np.load(os.path.join(filepath, "lons.npy"))
lats = np.load(os.path.join(filepath, "lats.npy"))
deps = np.load(os.path.join(filepath, "deps.npy"))

assert timestamps[-1] - (lons.shape[0]-1) < 1e-4  # rough check that timestamps span entire sim data

if timestamps == "firstlast":
    timestamps = [0, lons.shape[0]-1]

t0 = time.time()
# compute and plot densities
xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 720, 720, 360
density = np.zeros((50, 50, 25, len(timestamps)))  #np.zeros((100, 100, 50, len(timestamps)))
for t in range(len(timestamps)):
    x = lons[timestamps[t], ~np.isnan(lons[timestamps[t], :])]
    y = lats[timestamps[t], ~np.isnan(lats[timestamps[t], :])]
    z = deps[timestamps[t], ~np.isnan(deps[timestamps[t], :])]
    xyz = np.vstack([x, y, z])
    kde = stats.gaussian_kde(xyz)

    xi, yi, zi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:50j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

    # Multiprocessing
    cores = int(sys.argv[1])
    pool = multiprocessing.Pool(processes=cores)
    results = pool.map(calc_kde, np.array_split(coords.T, 2))
    density[:, :, :, t] = np.concatenate(results).reshape(xi.shape)
t1 = time.time()

np.save(filepath + '_density_' + str(cores) + 'cores.npy', density)
timefile = open(filepath + '_time_' + str(cores) + 'cores.txt', 'w')
timefile.write(str(t1-t0) + "seconds")