import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
from tqdm import tqdm
import time

def calc_kde(data):
    return kde(data.T)

timestamps = np.linspace(0, 300, 31, dtype=int).tolist()
filepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim"

# verify timestamp format
if isinstance(timestamps, int) or isinstance(timestamps, float):
    timestamps = [timestamps]

# load data
lons = np.load(str(filepath + "_lons.npy"))
lats = np.load(str(filepath + "_lats.npy"))
deps = np.load(str(filepath + "_deps.npy"))

assert timestamps[-1] - (lons.shape[0]-1) < 1e-4  # rough check that timestamps span entire sim data

if timestamps == "firstlast":
    timestamps = [0, lons.shape[0]-1]

# compute and plot densities
xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 720, 720, 360
nx, ny, nz = 72, 72, 36

density = np.zeros((nx, ny, nz, len(timestamps)))

N = len(timestamps)
with tqdm(total=N) as pbar:
    for t in range(N):
        x = lons[timestamps[t], ~np.isnan(lons[timestamps[t], :])]
        y = lats[timestamps[t], ~np.isnan(lats[timestamps[t], :])]
        z = deps[timestamps[t], ~np.isnan(deps[timestamps[t], :])]
        xyz = np.vstack([x, y, z])
        kde = stats.gaussian_kde(xyz)
        xi, yi, zi = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

        # Multiprocessing
        cores = 4
        pool = multiprocessing.Pool(processes=cores)
        results = pool.map(calc_kde, np.array_split(coords.T, 2))
        density[:, :, :, t] = np.concatenate(results).reshape(xi.shape)

        pbar.update(1)

np.save(filepath + '_density.npy', density)