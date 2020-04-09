import subprocess
import sys
import os
from glob import glob
import numpy as np
from tqdm import tqdm

filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/100000p_30s_0.01dt_0.05sdt_initunif_dead"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim/"

# maxdep = np.max(np.load(os.path.join(filedir, "deps.npy")), 1).min()

print("Calling Voro++ via bash script.")
subprocess.check_call("./script_voronoi.sh '%s'" % (filedir), shell=True)
print("Voronoi cells computed. Concatenating Voro++ output files.")

os.chdir(os.path.join(filedir, "vor/out"))

files = sorted(glob("./*.vol"))

vols = np.genfromtxt(files[0])

for f in tqdm(files[1:]):
    vols = np.c_[vols, np.genfromtxt(f)]

np.save("../../vols.npy", vols)
