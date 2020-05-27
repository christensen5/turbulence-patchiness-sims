"""
This script plots superimposed timeseries of the Q-statistic of all our simulations, at varying depth ranges.
The Q-statistic is computed using the volumes of the Voronoi tessellation of the particle positions at each timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
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
data_dead = {'vols': np.load(os.path.join(filepath_dead, 'vols_v.npy')),
             'depth': np.load(os.path.join(filepath_dead, 'vols_d.npy')),
             'V': 'dead', 'B': 'dead'}
data_v10_B1 = {'vols': np.load(os.path.join(filepath_v10_B1, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v10_B1, 'vols_d.npy')),
               'V': 10, 'B': 1.0}
data_v10_B3 = {'vols': np.load(os.path.join(filepath_v10_B3, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v10_B3, 'vols_d.npy')),
               'V': 10, 'B': 3.0}
data_v10_B5 = {'vols': np.load(os.path.join(filepath_v10_B5, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v10_B5, 'vols_d.npy')),
               'V': 10, 'B': 5.0}
data_v100_B1 = {'vols': np.load(os.path.join(filepath_v100_B1, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v100_B1, 'vols_d.npy')),
                'V': 100, 'B': 1.0}
data_v100_B3 = {'vols': np.load(os.path.join(filepath_v100_B3, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v100_B3, 'vols_d.npy')),
                'V': 100, 'B': 3.0}
data_v100_B5 = {'vols': np.load(os.path.join(filepath_v100_B5, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v100_B5, 'vols_d.npy')),
                'V': 100, 'B': 5.0}
data_v500_B1 = {'vols': np.load(os.path.join(filepath_v500_B1, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v500_B1, 'vols_d.npy')),
                'V': 500, 'B': 1.0}
data_v500_B3 = {'vols': np.load(os.path.join(filepath_v500_B3, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v500_B3, 'vols_d.npy')),
                'V': 500, 'B': 3.0}
data_v500_B5 = {'vols': np.load(os.path.join(filepath_v500_B5, 'vols_v.npy')),
                'depth': np.load(os.path.join(filepath_v500_B5, 'vols_d.npy')),
                'V': 500, 'B': 5.0}

data_dead['concs'] = np.reciprocal(data_dead['vols'])
data_v10_B1['concs'] = np.reciprocal(data_v10_B1['vols'])
data_v10_B3['concs'] = np.reciprocal(data_v10_B3['vols'])
data_v10_B5['concs'] = np.reciprocal(data_v10_B5['vols'])
data_v100_B1['concs'] = np.reciprocal(data_v100_B1['vols'])
data_v100_B3['concs'] = np.reciprocal(data_v100_B3['vols'])
data_v100_B5['concs'] = np.reciprocal(data_v100_B5['vols'])
data_v500_B1['concs'] = np.reciprocal(data_v500_B1['vols'])
data_v500_B3['concs'] = np.reciprocal(data_v500_B3['vols'])
data_v500_B5['concs'] = np.reciprocal(data_v500_B5['vols'])

all_motile_concentrations = [data_v10_B1, data_v10_B3, data_v10_B5,
                             data_v100_B1, data_v100_B3, data_v100_B5,
                             data_v500_B1, data_v500_B3, data_v500_B5]


# ======================================================================================================================
# Compute Q for each motile simulation.

f = 0.01

avg_func = np.mean
if not (avg_func.__name__ == 'mean' or avg_func.__name__ == 'median'):
    raise NotImplementedError("Q-analysis must use either mean or median, not %s." % avg_func.__name__)

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

# define depth ranges (in mm) (in ascending order from bottom to surface)
depth_slices = [[100, 170], [170, 240], [240, 305]]

timestamps = np.arange(0, 61, 1)#, 30]

for simdata_dict in tqdm(all_motile_concentrations):
    Q = {str(slice): [] for slice in depth_slices}  # Each sim will store the Q for each depth range.

    for t in timestamps:
        conc_dead = data_dead['concs'][:, t].flatten()
        conc_mot = simdata_dict['concs'][:, t].flatten()
        depths_dead = data_dead["depth"][:, t].flatten()
        depths_mot = simdata_dict["depth"][:, t].flatten()

        if nosurf:
            # remove surface particles
            conc_dead = conc_dead[depths_dead < 300]
            conc_mot = conc_mot[depths_mot < 300]
            depths_dead = depths_dead[depths_dead < 300]
            depths_mot = depths_mot[depths_mot < 300]

        for slice in depth_slices:
            # keep only particles in current slice
            conc_dead_d = conc_dead[np.logical_and(slice[0] < depths_dead, depths_dead < slice[1])]
            conc_mot_d = conc_mot[np.logical_and(slice[0] < depths_mot, depths_mot < slice[1])]

            Cm_d = avg_func(conc_dead_d)

            # keep only particles contained in patches, defined by f.
            Cdead_d = conc_dead_d[conc_dead_d.argsort()[-int(f * conc_dead_d.size):]]
            Cmot_d = conc_mot_d[conc_mot_d.argsort()[-int(f * conc_mot_d.size):]]

            Q_d = (avg_func(Cmot_d) - avg_func(Cdead_d)) / Cm_d
            Q[str(slice)].append(Q_d)

    for slice in depth_slices:
        Q[str(slice)] = np.array(Q[str(slice)])

    simdata_dict['Q'] = Q

# Q = np.log10(np.clip(Q, 1, None))


# ======================================================================================================================
# PLOT Q OVER TIME in a grid. Each row will contain plots from a different depth slice. Each column will contain plots
# from sims with different swim velocities. Each subplot has Q-values from different B-values superimposed.

fig = plt.figure(figsize=(24, 12))

colours = np.zeros((3, 3))
colours[1, :] = np.linspace(0, 1, 3)

plt.box(False)


# top depth slice ======================================================================================================
ax_v10_Dtop = plt.subplot(331)
l_v10_B1_Dtop = ax_v10_Dtop.plot(timestamps, data_v10_B1["Q"][str(depth_slices[-1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v10_B3_Dtop = ax_v10_Dtop.plot(timestamps, data_v10_B3["Q"][str(depth_slices[-1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v10_B5_Dtop = ax_v10_Dtop.plot(timestamps, data_v10_B5["Q"][str(depth_slices[-1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v10_B1_Dtop = plt.axhline(np.mean(data_v10_B1["Q"][str(depth_slices[-1])]), color=colours[:, 0], linestyle=':', linewidth=1, label="mean")
mean_v10_B3_Dtop = plt.axhline(np.mean(data_v10_B3["Q"][str(depth_slices[-1])]), color=colours[:, 1], linestyle=':', linewidth=1, label="mean")
mean_v10_B5_Dtop = plt.axhline(np.mean(data_v10_B5["Q"][str(depth_slices[-1])]), color=colours[:, 2], linestyle=':', linewidth=1, label="mean")
plt.axhline(0, color='k')
ax_v10_Dtop.set_title("v = 10um", fontsize=20)
ax_v10_Dtop.set_ylabel("Q", fontsize=25)
for tick in ax_v10_Dtop.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v10_Dtop.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax_v10_Dtop.legend(loc="lower left", ncol=2, columnspacing=1., bbox_to_anchor=[0, 1.15], fontsize=15)

ax_v100_Dtop = plt.subplot(332)
l_v100_B1_Dtop = ax_v100_Dtop.plot(timestamps, data_v100_B1["Q"][str(depth_slices[-1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v100_B3_Dtop = ax_v100_Dtop.plot(timestamps, data_v100_B3["Q"][str(depth_slices[-1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v100_B5_Dtop = ax_v100_Dtop.plot(timestamps, data_v100_B5["Q"][str(depth_slices[-1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v100_B1_Dtop = plt.axhline(np.mean(data_v100_B1["Q"][str(depth_slices[-1])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v100_B3_Dtop = plt.axhline(np.mean(data_v100_B3["Q"][str(depth_slices[-1])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v100_B5_Dtop = plt.axhline(np.mean(data_v100_B5["Q"][str(depth_slices[-1])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v100_Dtop.set_title("v = 100um", fontsize=20)
for tick in ax_v100_Dtop.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v100_Dtop.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

ax_v500_Dtop = plt.subplot(333)
l_v500_B1_Dtop = ax_v500_Dtop.plot(timestamps, data_v500_B1["Q"][str(depth_slices[-1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v500_B3_Dtop = ax_v500_Dtop.plot(timestamps, data_v500_B3["Q"][str(depth_slices[-1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v500_B5_Dtop = ax_v500_Dtop.plot(timestamps, data_v500_B5["Q"][str(depth_slices[-1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v500_B1_Dtop = plt.axhline(np.mean(data_v500_B1["Q"][str(depth_slices[-1])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v500_B3_Dtop = plt.axhline(np.mean(data_v500_B3["Q"][str(depth_slices[-1])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v500_B5_Dtop = plt.axhline(np.mean(data_v500_B5["Q"][str(depth_slices[-1])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v500_Dtop.set_title("v = 500um", fontsize=20)
ax_v500_Dtop_twin = ax_v500_Dtop.twinx()
ax_v500_Dtop_twin.set_yticks([], [])
ax_v500_Dtop_twin.set_ylabel("Shallow", fontsize=25, rotation=90)
for tick in ax_v500_Dtop.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v500_Dtop.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

# equalise ylims along rows (i.e. within depth slices)
if nosurf:  # don't equalise for shallow particles when nosurf=False
    row_ylim = (min([subplt.get_ylim()[0] for subplt in [ax_v10_Dtop, ax_v100_Dtop, ax_v500_Dtop]]),
                max([subplt.get_ylim()[1] for subplt in [ax_v10_Dtop, ax_v100_Dtop, ax_v500_Dtop]]))
    for subplt in [ax_v10_Dtop, ax_v100_Dtop, ax_v500_Dtop]:
        subplt.set_ylim(row_ylim)


# mid depth slice ======================================================================================================
ax_v10_Dmid = plt.subplot(334)
l_v10_B1_Dmid = ax_v10_Dmid.plot(timestamps, data_v10_B1["Q"][str(depth_slices[1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v10_B3_Dmid = ax_v10_Dmid.plot(timestamps, data_v10_B3["Q"][str(depth_slices[1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v10_B5_Dmid = ax_v10_Dmid.plot(timestamps, data_v10_B5["Q"][str(depth_slices[1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v10_B1_Dmid = plt.axhline(np.mean(data_v10_B1["Q"][str(depth_slices[1])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v10_B3_Dmid = plt.axhline(np.mean(data_v10_B3["Q"][str(depth_slices[1])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v10_B5_Dmid = plt.axhline(np.mean(data_v10_B5["Q"][str(depth_slices[1])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v10_Dmid.set_ylabel("Q", fontsize=25)
for tick in ax_v10_Dmid.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v10_Dmid.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

ax_v100_Dmid = plt.subplot(335)
l_v100_B1_Dmid = ax_v100_Dmid.plot(timestamps, data_v100_B1["Q"][str(depth_slices[1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v100_B3_Dmid = ax_v100_Dmid.plot(timestamps, data_v100_B3["Q"][str(depth_slices[1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v100_B5_Dmid = ax_v100_Dmid.plot(timestamps, data_v100_B5["Q"][str(depth_slices[1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v100_B1_Dmid = plt.axhline(np.mean(data_v100_B1["Q"][str(depth_slices[1])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v100_B3_Dmid = plt.axhline(np.mean(data_v100_B3["Q"][str(depth_slices[1])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v100_B5_Dmid = plt.axhline(np.mean(data_v100_B5["Q"][str(depth_slices[1])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
for tick in ax_v100_Dmid.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v100_Dmid.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

ax_v500_Dmid = plt.subplot(336)
l_v500_B1_Dmid = ax_v500_Dmid.plot(timestamps, data_v500_B1["Q"][str(depth_slices[1])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v500_B3_Dmid = ax_v500_Dmid.plot(timestamps, data_v500_B3["Q"][str(depth_slices[1])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v500_B5_Dmid = ax_v500_Dmid.plot(timestamps, data_v500_B5["Q"][str(depth_slices[1])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v500_B1_Dmid = plt.axhline(np.mean(data_v500_B1["Q"][str(depth_slices[1])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v500_B3_Dmid = plt.axhline(np.mean(data_v500_B3["Q"][str(depth_slices[1])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v500_B5_Dmid = plt.axhline(np.mean(data_v500_B5["Q"][str(depth_slices[1])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v500_Dmid_twin = ax_v500_Dmid.twinx()
ax_v500_Dmid_twin.set_yticks([], [])
ax_v500_Dmid_twin.set_ylabel("Mid", fontsize=25, rotation=90)
for tick in ax_v500_Dmid.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v500_Dmid.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

# equalise ylims along rows (i.e. within depth slices)
row_ylim = (min([subplt.get_ylim()[0] for subplt in [ax_v10_Dmid, ax_v100_Dmid, ax_v500_Dmid]]),
            max([subplt.get_ylim()[1] for subplt in [ax_v10_Dmid, ax_v100_Dmid, ax_v500_Dmid]]))
for subplt in [ax_v10_Dmid, ax_v100_Dmid, ax_v500_Dmid]:
    subplt.set_ylim(row_ylim)

# bottom depth slice ===================================================================================================
ax_v10_Dlow = plt.subplot(337)
l_v10_B1_Dlow = ax_v10_Dlow.plot(timestamps, data_v10_B1["Q"][str(depth_slices[0])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v10_B3_Dlow = ax_v10_Dlow.plot(timestamps, data_v10_B3["Q"][str(depth_slices[0])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v10_B5_Dlow = ax_v10_Dlow.plot(timestamps, data_v10_B5["Q"][str(depth_slices[0])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v10_B1_Dlow = plt.axhline(np.mean(data_v10_B1["Q"][str(depth_slices[0])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v10_B3_Dlow = plt.axhline(np.mean(data_v10_B3["Q"][str(depth_slices[0])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v10_B5_Dlow = plt.axhline(np.mean(data_v10_B5["Q"][str(depth_slices[0])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v10_Dlow.set_xlabel("Time", fontsize=25)
ax_v10_Dlow.set_ylabel("Q", fontsize=25)
for tick in ax_v10_Dlow.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v10_Dlow.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

ax_v100_Dlow = plt.subplot(338)
l_v100_B1_Dlow = ax_v100_Dlow.plot(timestamps, data_v100_B1["Q"][str(depth_slices[0])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v100_B3_Dlow = ax_v100_Dlow.plot(timestamps, data_v100_B3["Q"][str(depth_slices[0])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v100_B5_Dlow = ax_v100_Dlow.plot(timestamps, data_v100_B5["Q"][str(depth_slices[0])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v100_B1_Dlow = plt.axhline(np.mean(data_v100_B1["Q"][str(depth_slices[0])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v100_B3_Dlow = plt.axhline(np.mean(data_v100_B3["Q"][str(depth_slices[0])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v100_B5_Dlow = plt.axhline(np.mean(data_v100_B5["Q"][str(depth_slices[0])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v100_Dlow.set_xlabel("Time", fontsize=25)
for tick in ax_v100_Dlow.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v100_Dlow.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

ax_v500_Dlow = plt.subplot(339)
l_v500_B1_Dlow = ax_v500_Dlow.plot(timestamps, data_v500_B1["Q"][str(depth_slices[0])], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0')
l_v500_B3_Dlow = ax_v500_Dlow.plot(timestamps, data_v500_B3["Q"][str(depth_slices[0])], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0')
l_v500_B5_Dlow = ax_v500_Dlow.plot(timestamps, data_v500_B5["Q"][str(depth_slices[0])], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0')
mean_v500_B1_Dlow = plt.axhline(np.mean(data_v500_B1["Q"][str(depth_slices[0])]), color=colours[:, 0], linestyle=':', linewidth=1)
mean_v500_B3_Dlow = plt.axhline(np.mean(data_v500_B3["Q"][str(depth_slices[0])]), color=colours[:, 1], linestyle=':', linewidth=1)
mean_v500_B5_Dlow = plt.axhline(np.mean(data_v500_B5["Q"][str(depth_slices[0])]), color=colours[:, 2], linestyle=':', linewidth=1)
plt.axhline(0, color='k')
ax_v500_Dlow.set_xlabel("Time", fontsize=25)
ax_v500_Dlow_twin = ax_v500_Dlow.twinx()
ax_v500_Dlow_twin.set_yticks([], [])
ax_v500_Dlow_twin.set_ylabel("Deep", fontsize=25, rotation=90)
for tick in ax_v500_Dlow.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax_v500_Dlow.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

# equalise ylims along rows (i.e. within depth slices)
row_ylim = (min([subplt.get_ylim()[0] for subplt in [ax_v10_Dlow, ax_v100_Dlow, ax_v500_Dlow]]),
            max([subplt.get_ylim()[1] for subplt in [ax_v10_Dlow, ax_v100_Dlow, ax_v500_Dlow]]))
for subplt in [ax_v10_Dlow, ax_v100_Dlow, ax_v500_Dlow]:
    subplt.set_ylim(row_ylim)
    

if nosurf:
    st = fig.suptitle("Q statistic over time (excluding surface particles) (f=%0.2f)" % f, fontsize=25)
else:
    st = fig.suptitle("Q statistic over time (including surface particles) (f=%0.2f)" % f, fontsize=25)
fig.subplots_adjust(top=0.88)
# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/100000p_%0.2ff_Q_overtime_vs_Depth_%s%s.png" % (f, avg_func.__name__, surfstring),
            bbox_inches='tight')
