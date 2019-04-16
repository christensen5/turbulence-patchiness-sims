import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
from tqdm import tqdm
import time

def calc_kde(data):
    return kde(data.T)

timestamps = np.linspace(0, 300, 31, dtype=int).tolist()
filepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/trajectories_10000p_30s_0.01dt_0.1sdt_initunif_dead"

# verify timestamp format
if isinstance(timestamps, int) or isinstance(timestamps, float):
    timestamps = [timestamps]

# load data
lons = np.load(str(filepath + "_lons.npy"))
lats = np.load(str(filepath + "_lats.npy"))
deps = np.load(str(filepath + "_deps.npy"))

if timestamps == "firstlast":
    timestamps = [0, lons.shape[0]-1]

# compute and plot densities
xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 720, 720, 360
density = np.zeros((100, 100, 50, len(timestamps)))
for t in tqdm(range(len(timestamps))):
    x = lons[timestamps[t], ~np.isnan(lons[timestamps[t], :])]
    y = lats[timestamps[t], ~np.isnan(lats[timestamps[t], :])]
    z = deps[timestamps[t], ~np.isnan(deps[timestamps[t], :])]
    xyz = np.vstack([x, y, z])
    kde = stats.gaussian_kde(xyz)

    xi, yi, zi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:50j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

    # Multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    results = pool.map(calc_kde, np.array_split(coords.T, 2))
    density[:, :, :, t] = np.concatenate(results).reshape(xi.shape)

np.save("/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/density_10000p_30s_0.01dt_0.1sdt_initunif_dead/density.npy", density)