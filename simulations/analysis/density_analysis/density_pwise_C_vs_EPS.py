import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns
import os, sys

# ======================================================================================================================
# LOAD DATA

# specify paths to simulation output files
filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead"
filepath_v10_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot"
filepath_v10_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot"
filepath_v10_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot"
filepath_v100_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot"
filepath_v100_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot"
filepath_v100_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot"
filepath_v500_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
filepath_v500_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot"
filepath_v500_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot"

# load relevant files
data_dead = {'density': np.load(os.path.join(filepath_dead, 'density_pwise.npy')),
             'eps': np.load(os.path.join(filepath_dead, 'eps.npy')),
             'V': 'dead', 'B': 'dead'}
data_v10_B1 = {'density': np.load(os.path.join(filepath_v10_B1, 'density_pwise.npy')),
               'eps': np.load(os.path.join(filepath_v10_B1, 'eps.npy')),
               'V': 10, 'B': 1.0}
data_v10_B3 = {'density': np.load(os.path.join(filepath_v10_B3, 'density_pwise.npy')),
               'eps': np.load(os.path.join(filepath_v10_B3, 'eps.npy')),
               'V': 10, 'B': 3.0}
data_v10_B5 = {'density': np.load(os.path.join(filepath_v10_B5, 'density_pwise.npy')),
               'eps': np.load(os.path.join(filepath_v10_B5, 'eps.npy')),
               'V': 10, 'B': 5.0}
data_v100_B1 = {'density': np.load(os.path.join(filepath_v100_B1, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v100_B1, 'eps.npy')),
                'V': 100, 'B': 1.0}
data_v100_B3 = {'density': np.load(os.path.join(filepath_v100_B3, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v100_B3, 'eps.npy')),
                'V': 100, 'B': 3.0}
data_v100_B5 = {'density': np.load(os.path.join(filepath_v100_B5, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v100_B5, 'eps.npy')),
                'V': 100, 'B': 5.0}
data_v500_B1 = {'density': np.load(os.path.join(filepath_v500_B1, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v500_B1, 'eps.npy')),
                'V': 500, 'B': 1.0}
data_v500_B3 = {'density': np.load(os.path.join(filepath_v500_B3, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v500_B3, 'eps.npy')),
                'V': 500, 'B': 3.0}
data_v500_B5 = {'density': np.load(os.path.join(filepath_v500_B5, 'density_pwise.npy')),
                'eps': np.load(os.path.join(filepath_v500_B5, 'eps.npy')),
                'V': 500, 'B': 5.0}

all_motile_data = [data_v10_B1, data_v10_B3, data_v10_B5,
                   data_v100_B1, data_v100_B3, data_v100_B5,
                   data_v500_B1, data_v500_B3, data_v500_B5]


# ======================================================================================================================
# SCATTER PLOT C vs EPS AT 6 TIMESNAPS.

# find max and min eps (for plot limits)
xmin = data_dead["eps"].min()
xmax = data_dead["eps"].max()
for simdata_dict in all_motile_data:
    mineps = np.amin(simdata_dict["eps"])
    maxeps = np.amax(simdata_dict["eps"])
    if mineps < xmin:
        xmin = mineps
    if maxeps > xmax:
        xmax = maxeps
xlims = [xmin, xmax]
# ylims = [-10, -6]


dens = data_v500_B1["density"][:, [0, 3]]
eps = data_v500_B1["eps"][:, [0, 30]]

# fig = plt.figure(figsize=(15, 9))
# ax = plt.subplot(121)
# ax.scatter(eps[:, 0], np.log10(dens[:, 0]), s=1, alpha=0.1)
# ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# ax.set_xlabel("Epsilon")
# ax.set_ylabel("C")
# ax = plt.subplot(122)
# ax.scatter(eps[:, 1], np.log10(dens[:, 1]), s=1, alpha=0.1)
# ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# ax.set_xlabel("Epsilon")
# plt.show()
#
# plt.hist(eps[:, 1], 100)
# plt.show()


fig = plt.figure(figsize=(15, 9))
g = (sns.jointplot(eps[:, 1], dens[:, 1], xlim=xlims,# ylim=ylims,
                   marginal_kws=dict(bins=100),
                   s=0.5, alpha=0.1)).set_axis_labels("epsilon", "C")
plt.show()

