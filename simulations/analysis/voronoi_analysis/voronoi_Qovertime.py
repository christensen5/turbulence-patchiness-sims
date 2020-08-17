"""
This script plots superimposed timeseries of the Q-statistic of all our simulations. The Q-statistic here
is computed using the volumes of the Voronoi tessellation of the particle positions at each timestep.
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

# Set plot types
plot_3x1_superimposed = False
plot_3x3 = True


# ======================================================================================================================
# Compute Q for each motile simulation.

f = 0.1

avg_func = np.median
if not (avg_func.__name__ == 'mean' or avg_func.__name__ == 'median'):
    raise NotImplementedError("Q-analysis must use either mean or median, not %s." % avg_func.__name__)

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

timestamps = np.arange(0, 61, 1)#, 30]

for simdata_dict in tqdm(all_motile_concentrations):
    Q = []
    for t in timestamps:
        conc_dead = data_dead['concs'][:, t].flatten()
        conc_mot = simdata_dict['concs'][:, t].flatten()

        if nosurf:
            depths_dead = data_dead["depth"][:, t].flatten()
            depths_mot = simdata_dict["depth"][:, t].flatten()
            conc_dead = conc_dead[depths_dead < 300]
            conc_mot = conc_mot[depths_mot < 300]

        Cm_t = avg_func(conc_dead)

        # keep only particles contained in patches, defined by f.
        Cdead_t = conc_dead[conc_dead.argsort()[-int(f * conc_dead.size):]]
        Cmot_t = conc_mot[conc_mot.argsort()[-int(f * conc_mot.size):]]

        Q_t = (avg_func(Cmot_t) - avg_func(Cdead_t)) / Cm_t
        Q.append(Q_t)
    Q = np.array(Q)
    simdata_dict['Q'] = Q

# Q = np.log10(np.clip(Q, 1, None))


# ======================================================================================================================
# PLOT Q OVER TIME.
if plot_3x1_superimposed:
    fig = plt.figure(figsize=(15, 9))

    colours = np.zeros((3, 3))
    colours[1, :] = np.linspace(0, 1, 3)

    plt.box(False)
    ax_v10 = plt.subplot(311)
    l_v10_B1 = ax_v10.plot(timestamps, data_v10_B1["Q"], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0 v=10um')
    l_v10_B3 = ax_v10.plot(timestamps, data_v10_B3["Q"], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0 v=10um')
    l_v10_B5 = ax_v10.plot(timestamps, data_v10_B5["Q"], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0 v=10um')
    mean_v10_B1 = plt.axhline(np.mean(data_v10_B1["Q"]), color=colours[:, 0], linestyle=':', linewidth=1)
    mean_v10_B3 = plt.axhline(np.mean(data_v10_B3["Q"]), color=colours[:, 1], linestyle=':', linewidth=1)
    mean_v10_B5 = plt.axhline(np.mean(data_v10_B5["Q"]), color=colours[:, 2], linestyle=':', linewidth=1)
    plt.axhline(0, color='k')
    ax_v10.set_xlabel("Time", fontsize=25)
    ax_v10.set_ylabel("Q", fontsize=25)
    for tick in ax_v10.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    for tick in ax_v10.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    ax_v10.legend(loc="upper left", fontsize=15)
    ax_v100 = plt.subplot(312)
    l_v100_B1 = ax_v100.plot(timestamps, data_v100_B1["Q"], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0 v=100um')
    l_v100_B3 = ax_v100.plot(timestamps, data_v100_B3["Q"], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0 v=100um')
    l_v100_B5 = ax_v100.plot(timestamps, data_v100_B5["Q"], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0 v=100um')
    mean_v100_B1 = plt.axhline(np.mean(data_v100_B1["Q"]), color=colours[:, 0], linestyle=':', linewidth=1)
    mean_v100_B3 = plt.axhline(np.mean(data_v100_B3["Q"]), color=colours[:, 1], linestyle=':', linewidth=1)
    mean_v100_B5 = plt.axhline(np.mean(data_v100_B5["Q"]), color=colours[:, 2], linestyle=':', linewidth=1)
    plt.axhline(0, color='k')
    ax_v100.set_xlabel("Time", fontsize=25)
    ax_v100.set_ylabel("Q", fontsize=25)
    for tick in ax_v100.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    for tick in ax_v100.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    ax_v100.legend(loc="upper left", fontsize=15)
    ax_v500 = plt.subplot(313)
    l_v500_B1 = ax_v500.plot(timestamps, data_v500_B1["Q"], '-o', color=colours[:, 0], linewidth=1.5, markersize=3, label='B=1.0 v=500um')
    l_v500_B3 = ax_v500.plot(timestamps, data_v500_B3["Q"], '-o', color=colours[:, 1], linewidth=1.5, markersize=3, label='B=3.0 v=500um')
    l_v500_B5 = ax_v500.plot(timestamps, data_v500_B5["Q"], '-o', color=colours[:, 2], linewidth=1.5, markersize=3, label='B=5.0 v=500um')
    mean_v500_B1 = plt.axhline(np.mean(data_v500_B1["Q"]), color=colours[:, 0], linestyle=':', linewidth=1)
    mean_v500_B3 = plt.axhline(np.mean(data_v500_B3["Q"]), color=colours[:, 1], linestyle=':', linewidth=1)
    mean_v500_B5 = plt.axhline(np.mean(data_v500_B5["Q"]), color=colours[:, 2], linestyle=':', linewidth=1)
    plt.axhline(0, color='k')
    ax_v500.set_xlabel("Time", fontsize=25)
    ax_v500.set_ylabel("Q", fontsize=25)
    for tick in ax_v500.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    for tick in ax_v500.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    ax_v500.legend(loc="upper left", fontsize=15)

    if nosurf:
        st = fig.suptitle("Q statistic over time (excluding surface particles) (f=%0.2f)" % f, fontsize=25)
    else:
        st = fig.suptitle("Q statistic over time (including surface particles) (f=%0.2f)" % f, fontsize=25)
    # plt.show()
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime/3x1_superimposed/100000p_%0.2ff_Q_overtime_%s%s.png" % (f, avg_func.__name__, surfstring))


if plot_3x3:
    fig = plt.figure(figsize=(20, 16))

    if nosurf:
        st = fig.suptitle("Q statistic over time (excluding surface layer) (f=%0.2f)" % f, fontsize=32)
    else:
        st = fig.suptitle("Q statistic over time (including surface layer) (f=%0.2f)" % f, fontsize=32)

    colour = [0, 0.5, 0]

    plt.box(False)
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 330  # for indexing subplots
    for sim in all_motile_concentrations:
        plot_index += 1
        ax = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax.set_ylim([-0.21, 0.06])
            ax.set_yticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05])
        elif f == 0.01:
            ax.set_ylim([-0.41, 0.41])
            ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Q_line = ax.plot(timestamps, sim["Q"], '-o', color=colour, linewidth=2, markersize=3, label=r'B={:1.1f}s$^{{-1}}$ V={:d}um'.format(sim["B"], sim["V"]))
        Qmean = np.mean(sim["Q"])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax.text(0.04, 0.04, "("+plot_letter[plot_index % 10]+")",
                horizontalalignment='left', verticalalignment="bottom", transform=ax.transAxes, fontsize=18)
        ax.text(1.02, (Qmean - ax.get_ylim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]), "{:1.2f}".format(Qmean), color=colour,
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        # add B values above columns and vswim values alongside rows
        if plot_index % 10 < 4:
            ax.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$'.format(sim["B"]),
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=25)
        if plot_index % 10 in [1, 4, 7]:
            ax.text(-0.4, 0.5, 'Vswim = {:d}um'.format(sim["V"]),
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=25, rotation=90)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 6:
            ax.set_xlabel("Time", fontsize=25)
        if plot_index % 10 in [1, 4, 7]:
            ax.set_ylabel("Q", fontsize=25)

    fig.tight_layout(pad=0.7)
    fig.subplots_adjust(top=0.9)

    # plt.show()
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime/3x3/100000p_%0.2ff_Q_overtime_%s%s.png" % (f, avg_func.__name__, surfstring))
