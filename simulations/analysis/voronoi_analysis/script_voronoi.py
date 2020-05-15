import subprocess
import sys
import os
from glob import glob
import numpy as np
from tqdm import tqdm

# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_1000um_initunif_mot"
# filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_1000um_initunif_mot"
filedir = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot"


# maxdep = np.max(np.load(os.path.join(filedir, "deps.npy")), 1).min()

print("Calling Voro++ via bash script.")
subprocess.check_call("./script_voronoi.sh '%s'" % (filedir), shell=True)
print("Voronoi cells computed. Concatenating Voro++ output files.")

os.chdir(os.path.join(filedir, "vor/out"))

files = sorted(glob("./*.vol"))

# voro++ will (very rarely) skip a particle that is almost exactly on top of to another one. If this happens,
# we shorten the particleset to avoid issues with stacking in the later for-loop.
outfile_sizes = []
for f in files:
    outfile_sizes.append(np.genfromtxt(f).shape[0])
numparticles = min(outfile_sizes)
print("(removed %d particles)." %(max(outfile_sizes)-numparticles))
vols = np.genfromtxt(files[0])[0:numparticles, 0]
deps = np.genfromtxt(files[0])[0:numparticles, 1]
for f in tqdm(files[1:]):
    vols = np.c_[vols, np.genfromtxt(f)[0:numparticles, 0]]
    deps = np.c_[deps, np.genfromtxt(f)[0:numparticles, 1]]

np.save("../../vols_v.npy", vols)
np.save("../../vols_d.npy", deps)
