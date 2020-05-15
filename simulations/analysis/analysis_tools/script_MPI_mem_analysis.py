import numpy as np
from datetime import timedelta
from time import strptime
import netCDF4
from glob import glob
from tqdm import tqdm
import os
import sys
from copy import deepcopy
import matplotlib.pyplot as plt

"""
A script to load the memory usage and runtime data for MPI runs and plot diagnostics.
"""

filepaths = "/home/alexander/Desktop/temp_results/MPI_trials/?core/*"
paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths

# Initialise dictionaries.
ncores = {os.path.basename(fp).split("_")[3][0]+"c":[] for fp in paths}
memdict = {os.path.basename(fp).split("_")[2][:-9]:deepcopy(ncores) for fp in paths}
timedict = {os.path.basename(fp).split("_")[2][:-9]:deepcopy(ncores) for fp in paths}

for fp in paths:
    variable, nparticles, chunksize, nproc = os.path.basename(fp)[:-4].split("_")
    chunksize = chunksize[:-9]
    nproc = nproc[0]+"c"

    if variable == "memuseage":
        memdict[chunksize][nproc].append(np.load(fp).max())

    if variable == "timer":
        with open(fp, 'r') as data:
            lastline = data.read().splitlines()[-1]
            psetruntime = strptime(lastline[-14:], "%H:%M:%S.%f")
            timedict[chunksize][nproc].append(timedelta(hours=psetruntime.tm_hour,
                                                   minutes=psetruntime.tm_min,
                                                   seconds=psetruntime.tm_sec).total_seconds())

print("Memory useage and runtimes extracted.")

x = list(memdict.keys())
x.remove("auto")
x.remove("None")
x = sorted([int(i) for i in x])

mems_2c = [np.sum(memdict[str(chunksize)]['2c']) for chunksize in x]  # add mems
mems_4c = [np.sum(memdict[str(chunksize)]['4c']) for chunksize in x]  # add mems

times = [[np.mean(timedict[str(chunksize)][proc]) for proc in timedict[str(chunksize)].keys()] for chunksize in x]  # average time
times_2c = [np.mean(timedict[str(chunksize)]['2c']) for chunksize in x]  # average time
times_4c = [np.mean(timedict[str(chunksize)]['4c']) for chunksize in x]  # average time

plt.figure(figsize=(12, 8))
plt.suptitle("Memory Useage and Runtime Analysis in MPI-mode.")
ax1 = plt.subplot(211)
ax1.set_ylabel("Memory Useage [MB]")
ax1.plot(x, mems_2c, 'ob-', label="2c")
ax1.plot(x, mems_4c, '^b-', label="4c")
ax1.hlines(np.sum(memdict["auto"]["2c"]), x[0], x[-1], 'g', 'dashed', label="auto(2c)")
ax1.hlines(np.sum(memdict["auto"]["4c"]), x[0], x[-1], 'g', 'dotted', label="auto(4c)")
ax1.hlines(np.sum(memdict["None"]["2c"]), x[0], x[-1], 'r', 'dashed', label="None(2c)")
ax1.hlines(np.sum(memdict["None"]["4c"]), x[0], x[-1], 'r', 'dotted', label="None(4c)")
ax1.legend()
ax2 = plt.subplot(212)
ax2.set_xlabel("field_chunksize")
ax2.set_ylabel("ParticleSet runtime [s]")
ax2.plot(x, times_2c, 'om-', label="2-core pset runtime [s]")
ax2.plot(x, times_4c, '^m-', label="4-core pset runtime [s]")
ax2.hlines(np.mean(timedict["auto"]["2c"]), x[0], x[-1], 'g', 'dashed', label="auto(2c)")
ax2.hlines(np.mean(timedict["auto"]["4c"]), x[0], x[-1], 'g', 'dotted', label="auto(4c)")
ax2.hlines(np.mean(timedict["None"]["2c"]), x[0], x[-1], 'r', 'dashed', label="None(2c)")
ax2.hlines(np.mean(timedict["None"]["4c"]), x[0], x[-1], 'r', 'dotted', label="None(4c)")
ax2.legend()
plt.show()

