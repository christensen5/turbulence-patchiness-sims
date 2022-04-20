"""
This script plots superimposed timeseries of the Q-statistic of all our simulations, at varying depth ranges.
The Q-statistic is computed using the volumes of the Voronoi tessellation of the particle positions at each timestep.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from tqdm import tqdm
import os, sys

# ======================================================================================================================
# LOAD DATA

# specify paths to simulation output files
filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead"

filepath_v10_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_10um_initunif_mot"
filepath_v100_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_100um_initunif_mot"
filepath_v500_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_500um_initunif_mot"

# load relevant files
data_dead = {'vols': np.load(os.path.join(filepath_dead, 'vols_v.npy')),
             'depth': np.load(os.path.join(filepath_dead, 'vols_d.npy')),
             'V': 'dead', 'B': 'dead'}
data_v10_B0 = {'vols': np.load(os.path.join(filepath_v10_B0, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v10_B0, 'vols_d.npy')),
               'V': 10, 'B': 0}
data_v100_B0 = {'vols': np.load(os.path.join(filepath_v100_B0, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v100_B0, 'vols_d.npy')),
               'V': 100, 'B': 0}
data_v500_B0 = {'vols': np.load(os.path.join(filepath_v500_B0, 'vols_v.npy')),
               'depth': np.load(os.path.join(filepath_v500_B0, 'vols_d.npy')),
               'V': 500, 'B': 0}

data_dead['concs'] = np.reciprocal(data_dead['vols'])
data_v10_B0['concs'] = np.reciprocal(data_v10_B0['vols'])
data_v100_B0['concs'] = np.reciprocal(data_v100_B0['vols'])
data_v500_B0['concs'] = np.reciprocal(data_v500_B0['vols'])


all_motile_concentrations = [data_v10_B0, data_v100_B0, data_v500_B0]

# Set plot types
plot_3depthsx2slowVfastSims = True
plot_violins = False

# ======================================================================================================================
# Compute Q for each motile simulation.

f = 0.01

avg_func = np.median
if not (avg_func.__name__ == 'mean' or avg_func.__name__ == 'median'):
    raise NotImplementedError("Q-analysis must use either mean or median, not %s." % avg_func.__name__)

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

# define depth ranges (in mm) (in ascending order from bottom to surface)
depth_slices = [[100, 170], [170, 240], [240, 305]]
depth_slice_names = ["Deep", "Mid", "Shallow"]

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
# PLOT Q OVER TIME...

if plot_3depthsx2slowVfastSims:
    # ...in a grid. Each row will contain plots from a different depth slice. Left column will contain plots
    # from non-agile sims, right column will contain plots from agile sims.

    fig = plt.figure(figsize=(20, 16))

    # if nosurf:
    #     st = fig.suptitle("Q statistic over time (excluding surface layer) (f=%0.2f)" % f, fontsize=32)
    # else:
    #     st = fig.suptitle("Q statistic over time (including surface layer) (f=%0.2f)" % f, fontsize=32)

    colour = ['tab:blue', 'tab:orange', 'tab:green', 'c']
    colour_boring = ['tab:red', 'tab:purple'] #[[1, 0.5, 0.5], [1, 0, 0]]
    colour_agile = ['tab:blue', 'tab:orange'] #[[0.5, 0.5, 1], [0, 0, 1]]

    # top depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 321
    ax_shallow_boring = plt.subplot(plot_index)
    ax_shallow_boring.text(0.5, 1.1, 'No reorientation,\nSlow Swimmers',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_shallow_boring.transAxes, fontsize=25)
    ax_shallow_boring.text(-0.15, 0.9, 'a',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_boring.transAxes, fontsize=25, fontweight="bold")
    Qmeans_shallow_boring = []
    for sim, i in zip([data_v10_B0], range(len([data_v10_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_shallow_boring.set_ylim([-0.21, 0.11])
        #     ax_shallow_boring.set_yticks([-0.2, -0.1, 0, 0.1])
        if f == 0.01:
            ax_shallow_boring.set_ylim([-0.81, 1.61])
            ax_shallow_boring.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_shallow_boring.plot(timestamps, sim["Q"][str(depth_slices[-1])],
                                '-o', color=colour_boring[i], linewidth=2, markersize=3,
                                label=r'No reorientation, v = {:d}$\mu ms^{{-1}}$'.format(sim["V"]))
        Qmeans_shallow_boring.append(np.mean(sim["Q"][str(depth_slices[-1])]))
        # axis ticks fontsize handling
        for tick in ax_shallow_boring.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        for tick in ax_shallow_boring.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_shallow_boring.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_shallow_boring.set_ylabel(r"$Q_{{\mathrm{{Shallow}}}}$", fontsize=25)
    Q_meanline = plt.axhline(np.mean(Qmeans_shallow_boring), color='tab:gray', linestyle='--', linewidth=2)
    ax_shallow_boring.text(1.02, (np.mean(Qmeans_shallow_boring) - ax_shallow_boring.get_ylim()[0]) / (
                ax_shallow_boring.get_ylim()[1] - ax_shallow_boring.get_ylim()[0]), r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_shallow_boring)),
                           color='tab:gray', horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_boring.transAxes, fontsize=20)

    plot_index = 322
    ax_shallow_agile = plt.subplot(plot_index)
    ax_shallow_agile.text(0.5, 1.1, 'No reorientation,\nFast swimmers',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_shallow_agile.transAxes, fontsize=25)
    ax_shallow_agile.text(-0.15, 0.9, 'b',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_agile.transAxes, fontsize=25, fontweight="bold")
    Qmeans_shallow_agile = []
    for sim, i in zip([data_v500_B0], range(len([data_v500_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_shallow_agile.set_ylim([-0.21, 0.11])
        #     ax_shallow_agile.set_yticks([-0.2, -0.1, 0, 0.1])
        if f == 0.01:
            ax_shallow_agile.set_ylim([-0.81, 1.61])
            ax_shallow_agile.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_shallow_agile.plot(timestamps, sim["Q"][str(depth_slices[-1])],
                                       '-o', color=colour_agile[i], linewidth=2, markersize=3,
                                      label=r'No reorientation, v = {:d}$\mu ms^{{-1}}$'.format(sim["V"]))
        Qmeans_shallow_agile.append(np.mean(sim["Q"][str(depth_slices[-1])]))
        # axis ticks fontsize handling
        for tick in ax_shallow_agile.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_shallow_agile.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_shallow_agile.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_shallow_agile.set_ylabel(r"$Q_{{\mathrm{{Shallow}}}}$", fontsize=25)
    Q_meanline = plt.axhline(np.mean(Qmeans_shallow_agile), color='tab:gray', linestyle='--', linewidth=2)
    ax_shallow_agile.text(1.02, (np.mean(Qmeans_shallow_agile) - ax_shallow_agile.get_ylim()[0]) / (
            ax_shallow_agile.get_ylim()[1] - ax_shallow_agile.get_ylim()[0]),
                           r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_shallow_agile)),
                           color='tab:gray', horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_agile.transAxes, fontsize=20)


    # mid depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 323
    ax_mid_boring = plt.subplot(plot_index)
    ax_mid_boring.text(-0.15, 0.9, 'c',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_mid_boring.transAxes, fontsize=25, fontweight="bold")
    Qmeans_mid_boring = []
    for sim, i in zip([data_v10_B0], range(len([data_v10_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_mid_boring.set_ylim([-0.21, 0.11])
        #     ax_mid_boring.set_yticks([-0.2, -0.1, 0, 0.1])
        if f == 0.01:
            ax_mid_boring.set_ylim([-0.81, 1.61])
            ax_mid_boring.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_mid_boring.plot(timestamps, sim["Q"][str(depth_slices[1])],
                                '-o', color=colour_boring[i], linewidth=2, markersize=3)
        Qmeans_mid_boring.append(np.mean(sim["Q"][str(depth_slices[1])]))
        # axis ticks fontsize handling
        for tick in ax_mid_boring.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_mid_boring.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_mid_boring.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_mid_boring.set_ylabel(r"$Q_{{\mathrm{{Mid}}}}$", fontsize=25)
    Q_meanline = plt.axhline(np.mean(Qmeans_mid_boring), color='tab:gray', linestyle='--', linewidth=2)
    ax_mid_boring.text(1.02, (np.mean(Qmeans_mid_boring) - ax_mid_boring.get_ylim()[0]) / (
            ax_mid_boring.get_ylim()[1] - ax_mid_boring.get_ylim()[0]),
                           r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_mid_boring)),
                           color='tab:gray', horizontalalignment='left', verticalalignment='center',
                           transform=ax_mid_boring.transAxes, fontsize=20)

    plot_index = 324
    ax_mid_agile = plt.subplot(plot_index)
    ax_mid_agile.text(-0.15, 0.9, 'd',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_mid_agile.transAxes, fontsize=25, fontweight="bold")
    Qmeans_mid_agile = []
    for sim, i in zip([data_v500_B0], range(len([data_v500_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_mid_agile.set_ylim([-0.21, 0.11])
        #     ax_mid_agile.set_yticks([-0.2, -0.1, 0, 0.1])
        if f == 0.01:
            ax_mid_agile.set_ylim([-0.81, 1.61])
            ax_mid_agile.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_mid_agile.plot(timestamps, sim["Q"][str(depth_slices[1])],
                                   '-o', color=colour_agile[i], linewidth=2, markersize=3)
        Qmeans_mid_agile.append(np.mean(sim["Q"][str(depth_slices[1])]))
        # axis ticks fontsize handling
        for tick in ax_mid_agile.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_mid_agile.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_mid_agile.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_mid_agile.set_ylabel(r"$Q_{{\mathrm{{Mid}}}}$", fontsize=25)
    Q_meanline = plt.axhline(np.mean(Qmeans_mid_agile), color='tab:gray', linestyle='--', linewidth=2)
    ax_mid_agile.text(1.02, (np.mean(Qmeans_mid_agile) - ax_mid_agile.get_ylim()[0]) / (
            ax_mid_agile.get_ylim()[1] - ax_mid_agile.get_ylim()[0]),
                       r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_mid_agile)),
                       color='tab:gray', horizontalalignment='left', verticalalignment='center',
                       transform=ax_mid_agile.transAxes, fontsize=20)

    # Deep depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 325
    ax_deep_boring = plt.subplot(plot_index)
    ax_deep_boring.text(-0.15, 0.9, 'e',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_deep_boring.transAxes, fontsize=25, fontweight="bold")
    Qmeans_deep_boring = []
    for sim, i in zip([data_v10_B0], range(len([data_v10_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_deep_boring.set_ylim([-0.41, 0.41])
        #     ax_deep_boring.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        if f == 0.01:
            ax_deep_boring.set_ylim([-0.81, 1.61])
            ax_deep_boring.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_deep_boring.plot(timestamps, sim["Q"][str(depth_slices[0])],
                            '-o', color=colour_boring[i], linewidth=2, markersize=3)
        Qmeans_deep_boring.append(np.mean(sim["Q"][str(depth_slices[0])]))
        # axis ticks fontsize handling
        for tick in ax_deep_boring.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_deep_boring.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_deep_boring.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_deep_boring.set_ylabel(r"$Q_{{\mathrm{{Deep}}}}$", fontsize=25, rotation=90)
    Q_meanline = plt.axhline(np.mean(Qmeans_deep_boring), color='tab:gray', linestyle='--', linewidth=2)
    ax_deep_boring.text(1.02, (np.mean(Qmeans_deep_boring) - ax_deep_boring.get_ylim()[0]) / (
            ax_deep_boring.get_ylim()[1] - ax_deep_boring.get_ylim()[0]),
                           r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_deep_boring)),
                           color='tab:gray', horizontalalignment='left', verticalalignment='center',
                           transform=ax_deep_boring.transAxes, fontsize=20)

    plot_index = 326
    ax_deep_agile = plt.subplot(plot_index)
    ax_deep_agile.text(-0.15, 0.9, 'f',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_deep_agile.transAxes, fontsize=25, fontweight="bold")
    Qmeans_deep_agile = []
    for sim, i in zip([data_v500_B0], range(len([data_v500_B0]))):
        # set axis limits and tick locations
        # if f == 0.1:
        #     ax_deep_agile.set_ylim([-0.41, 0.41])
        #     ax_deep_agile.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        if f == 0.01:
            ax_deep_agile.set_ylim([-0.81, 1.61])
            ax_deep_agile.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_deep_agile.plot(timestamps, sim["Q"][str(depth_slices[0])],
                                    '-o', color=colour_agile[i], linewidth=2, markersize=3)
        Qmeans_deep_agile.append(np.mean(sim["Q"][str(depth_slices[0])]))
        # axis ticks fontsize handling
        for tick in ax_deep_agile.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_deep_agile.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_deep_agile.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_deep_agile.set_ylabel(r"$Q_{{\mathrm{{Deep}}}}$", fontsize=25)
    Q_meanline = plt.axhline(np.mean(Qmeans_deep_agile), color='tab:gray', linestyle='--', linewidth=2)
    ax_deep_agile.text(1.02, (np.mean(Qmeans_deep_agile) - ax_deep_agile.get_ylim()[0]) / (
            ax_deep_agile.get_ylim()[1] - ax_deep_agile.get_ylim()[0]),
                        r"$\overline{{Q}} =${:1.2f}".format(np.mean(Qmeans_deep_agile)),
                        color='tab:gray', horizontalalignment='left', verticalalignment='center',
                        transform=ax_deep_agile.transAxes, fontsize=20)

    # handles_boring, labels_boring = ax_deep_boring.get_legend_handles_labels()
    ax_shallow_boring.legend(loc="upper center", fontsize=20)
    ax_shallow_agile.legend(loc="upper center", fontsize=20)


    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # plt.show()
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/3x2slowVfastSims_0B/100000p_%0.2ff_0B_Q_overtime_%s%s.png" % (f, avg_func.__name__, surfstring))
    plt.clf()


# if plot_violins:
#     nonAgileSims = boring_motile_concentrations
#     agileSims = agile_motile_concentrations
#     nonAgileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])] for sim in nonAgileSims]), np.concatenate([sim['Q'][str(depth_slices[1])] for sim in nonAgileSims])]),
#                           np.concatenate([sim['Q'][str(depth_slices[0])] for sim in nonAgileSims])]  # deep nonagile
#     agileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])] for sim in agileSims]), np.concatenate([sim['Q'][str(depth_slices[1])] for sim in agileSims])]),
#                        np.concatenate([sim['Q'][str(depth_slices[0])] for sim in agileSims])]  # deep agile
#     # xlabels = ["Shallow+Mid\nSlow", "Shallow+Mid\nAgile", "Deep\nSlow", "Deep\nAgile"]
#     ylabels = ["Deep\nAgile", "Deep\nNon-Agile", "Shallow+Mid\nAgile", "Shallow+Mid\nNon-Agile"]
#
#     plt.clf()
#     fig = plt.figure(figsize=(20, 15))
#     ax = plt.gca()
#     nonAgileViolins = ax.violinplot(nonAgileViolinData, positions=[2.4, 1.2], vert=False, showextrema=False, showmeans=False) #4,2
#     agileViolins = ax.violinplot(agileViolinData, positions=[1.8, 0.6], vert=False, showextrema=False, showmeans=False) #3,1
#     for b in nonAgileViolins['bodies']:
#         b.set_facecolor('#d6604d')
#         b.set_alpha(1)
#     for b in agileViolins['bodies']:
#         b.set_facecolor('#4393c3')
#         b.set_alpha(1)
#     # means = [np.mean(x) for x in [nonAgileViolinData[0], agileViolinData[0], nonAgileViolinData[1], agileViolinData[1]]]
#     means = [np.mean(x) for x in [agileViolinData[1], nonAgileViolinData[1], agileViolinData[0], nonAgileViolinData[0]]]
#     ax.axvline(0, ymin=0, ymax=1, color='k', lw=1)
#     ax.vlines(means, ymin=[0.3, 0.9, 1.5, 2.1], ymax=[0.9, 1.5, 2.1, 2.7], colors='white', lw=2)
#     ax.text(means[0]+0.01, 0.6, r'$\overline{{Q}}={:1.2f}$'.format(means[0]), color='white', fontsize=22,
#             horizontalalignment="left", verticalalignment="top")
#     ax.text(means[1]+0.01, 1.2, r'$\overline{{Q}}={:1.2f}$'.format(means[1]), color='white', fontsize=22,
#             horizontalalignment="left", verticalalignment="top")
#     ax.text(means[2]-0.01, 1.8, r'$\overline{{Q}}={:1.2f}$'.format(means[2]), color='white', fontsize=22,
#             horizontalalignment="right", verticalalignment="center")
#     ax.text(means[3]-0.01, 2.4, r'$\overline{{Q}}={:1.2f}$'.format(means[3]), color='white', fontsize=22,
#             horizontalalignment="right", verticalalignment="center")
#     ax.set_yticks([0.6, 1.2, 1.8, 2.4])
#     ax.set_yticklabels(ylabels)
#     ax.tick_params(axis='y', which='both', length=0)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(26)
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(26)
#     # ax.set_ylabel("Sample Type", fontsize=26)
#     ax.set_xlabel("Q", fontsize=26)
#     legend_patches = [mpatches.Patch(color='#d6604d'), mpatches.Patch(color='#4393c3')]
#     legend_labels = ["Non-Agile", "Agile"]
#     ax.legend(legend_patches, legend_labels, fontsize=24, loc="center right")
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_color('black')
#
#     # plt.show()
#     fig.savefig(
#             "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/violins/100000p_%0.2ff_violin_%s%s.png" % (
#             f, avg_func.__name__, surfstring))