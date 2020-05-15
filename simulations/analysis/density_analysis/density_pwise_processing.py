import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
from tqdm import tqdm
import os
import sys

timestamps = np.arange(0, 601, 120).tolist()
filepath = sys.argv[1]

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

density = np.zeros((lons.shape[1], len(timestamps)))

N = len(timestamps)
with tqdm(total=N) as pbar:
    for t in range(N):
        x = lons[timestamps[t], ~np.isnan(lons[timestamps[t], :])]
        y = lats[timestamps[t], ~np.isnan(lats[timestamps[t], :])]
        z = deps[timestamps[t], ~np.isnan(deps[timestamps[t], :])]
        xyz = np.vstack([x, y, z])
        kde = stats.gaussian_kde(xyz)
        density[:, t] = kde(xyz)

        pbar.update(1)

np.save(os.path.join(filepath, 'density_pwise.npy'), density)
