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

agile_motile_concentrations = [data_v100_B3, data_v100_B1,
                               data_v500_B3, data_v500_B1]

boring_motile_concentrations = [data_v10_B5, data_v10_B3,
                                data_v100_B5, data_v100_B3]

agile_motile_representatives = [data_v500_B1, data_v500_B3]

boring_motile_representatives = [data_v10_B3, data_v100_B3]

agilest_motile_concentrations = [data_v500_B1, data_v500_B3]

boringest_motile_concentration = [data_v100_B3]

# Set plot types
plot_3x3allSims = False
plot_3depthsx2agilestSims = False
plot_3depthsx1boringSim = False
plot_3depthsx2boringVagileSims = True
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

if plot_3x3allSims:
    #  ...in a grid. Each row will contain plots from sims with different swim velocities. Each column will contain plots
    # from sims with different reorientation parameters.

    for i in range(len(depth_slices)):
        fig = plt.figure(figsize=(20, 16))
        plt.box(False)

        ax_v10_B1 = plt.subplot(331)
        ax_v10_B1.set_ylim([-0.8, 1.6])
        ax_v10_B1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v10_B1 = ax_v10_B1.plot(timestamps, data_v10_B1["Q"][str(depth_slices[i])], '-o', color='g', linewidth=1.5, markersize=3)
        Qmean = np.mean(data_v10_B1["Q"][str(depth_slices[i])])
        mean_v10_B1 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v10_B1.text(-0.2, 0.96, 'a', ha='left', va="top", transform=ax_v10_B1.transAxes, fontsize=20, fontweight="bold")
        ax_v10_B1.text(1.02, (Qmean - ax_v10_B1.get_ylim()[0]) / (ax_v10_B1.get_ylim()[1] - ax_v10_B1.get_ylim()[0]), "{:1.2f}".format(Qmean),
                color='g',
                horizontalalignment='left', verticalalignment='center', transform=ax_v10_B1.transAxes, fontsize=20)
        plt.axhline(0, color='k')
        ax_v10_B1.set_ylabel("Q", fontsize=25)
        ax_v10_B1.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$'.format(data_v10_B1["B"]),
                horizontalalignment='center', verticalalignment='center',
                transform=ax_v10_B1.transAxes, fontsize=25)
        ax_v10_B1.text(-0.3, 0.5, 'Vswim = {:d}um'.format(data_v10_B1["V"]),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_v10_B1.transAxes, fontsize=25, rotation=90)

        ax_v10_B3 = plt.subplot(332)
        ax_v10_B3.set_ylim([-0.8, 1.6])
        ax_v10_B3.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v10_B3 = ax_v10_B3.plot(timestamps, data_v10_B3["Q"][str(depth_slices[i])], '-o', color='g',
                                  linewidth=1.5, markersize=3)
        Qmean = np.mean(data_v10_B3["Q"][str(depth_slices[i])])
        mean_v10_B3 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v10_B3.text(-0.2, 0.96, 'b', ha='left', va="top", transform=ax_v10_B3.transAxes, fontsize=20, fontweight="bold")
        ax_v10_B3.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$'.format(data_v10_B3["B"]),
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax_v10_B3.transAxes, fontsize=25)
        ax_v10_B3.text(1.02, (Qmean - ax_v10_B3.get_ylim()[0]) / (ax_v10_B3.get_ylim()[1] - ax_v10_B3.get_ylim()[0]),
                       "{:1.2f}".format(Qmean),
                       color='g',
                       horizontalalignment='left', verticalalignment='center', transform=ax_v10_B3.transAxes,
                       fontsize=20)
        plt.axhline(0, color='k')

        ax_v10_B5 = plt.subplot(333)
        ax_v10_B5.set_ylim([-0.8, 1.6])
        ax_v10_B5.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v10_B5 = ax_v10_B5.plot(timestamps, data_v10_B5["Q"][str(depth_slices[i])], '-o', color='g',
                                  linewidth=1.5, markersize=3)
        Qmean = np.mean(data_v10_B5["Q"][str(depth_slices[i])])
        mean_v10_B5 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v10_B5.text(-0.2, 0.96, 'c', ha='left', va="top", transform=ax_v10_B5.transAxes, fontsize=20, fontweight="bold")
        ax_v10_B5.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$'.format(data_v10_B5["B"]),
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax_v10_B5.transAxes, fontsize=25)
        ax_v10_B5.text(1.02, (Qmean - ax_v10_B5.get_ylim()[0]) / (ax_v10_B5.get_ylim()[1] - ax_v10_B5.get_ylim()[0]),
                       "{:1.2f}".format(Qmean),
                       color='g',
                       horizontalalignment='left', verticalalignment='center', transform=ax_v10_B5.transAxes,
                       fontsize=20)
        plt.axhline(0, color='k')

        ax_v100_B1 = plt.subplot(334)
        ax_v100_B1.set_ylim([-0.8, 1.6])
        ax_v100_B1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v100_B1 = ax_v100_B1.plot(timestamps, data_v100_B1["Q"][str(depth_slices[i])], '-o', color='g', linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v100_B1["Q"][str(depth_slices[i])])
        mean_v100_B1 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v100_B1.text(-0.2, 0.96, 'd', ha='left', va="top", transform=ax_v100_B1.transAxes, fontsize=20, fontweight="bold")
        ax_v100_B1.text(1.02, (Qmean - ax_v100_B1.get_ylim()[0]) / (ax_v100_B1.get_ylim()[1] - ax_v100_B1.get_ylim()[0]),
                       "{:1.2f}".format(Qmean),
                       color='g',
                       horizontalalignment='left', verticalalignment='center', transform=ax_v100_B1.transAxes,
                       fontsize=20)
        plt.axhline(0, color='k')
        ax_v100_B1.set_ylabel("Q", fontsize=25)
        ax_v100_B1.text(-0.3, 0.5, 'Vswim = {:d}um'.format(data_v100_B1["V"]),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_v100_B1.transAxes, fontsize=25, rotation=90)

        ax_v100_B3 = plt.subplot(335)
        ax_v100_B3.set_ylim([-0.8, 1.6])
        ax_v100_B3.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v100_B3 = ax_v100_B3.plot(timestamps, data_v100_B3["Q"][str(depth_slices[i])], '-o', color='g',
                                    linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v100_B3["Q"][str(depth_slices[i])])
        mean_v100_B3 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v100_B3.text(-0.2, 0.96, 'e', ha='left', va="top", transform=ax_v100_B3.transAxes, fontsize=20, fontweight="bold")
        ax_v100_B3.text(1.02,
                        (Qmean - ax_v100_B3.get_ylim()[0]) / (ax_v100_B3.get_ylim()[1] - ax_v100_B3.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color='g',
                        horizontalalignment='left', verticalalignment='center', transform=ax_v100_B3.transAxes,
                        fontsize=20)
        plt.axhline(0, color='k')

        ax_v100_B5 = plt.subplot(336)
        ax_v100_B5.set_ylim([-0.8, 1.6])
        ax_v100_B5.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v100_B5 = ax_v100_B5.plot(timestamps, data_v100_B5["Q"][str(depth_slices[i])], '-o', color='g',
                                    linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v100_B5["Q"][str(depth_slices[i])])
        mean_v100_B5 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v100_B5.text(-0.2, 0.96, 'f', ha='left', va="top", transform=ax_v100_B5.transAxes, fontsize=20, fontweight="bold")
        ax_v100_B5.text(1.02,
                        (Qmean - ax_v100_B5.get_ylim()[0]) / (ax_v100_B5.get_ylim()[1] - ax_v100_B5.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color='g',
                        horizontalalignment='left', verticalalignment='center', transform=ax_v100_B5.transAxes,
                        fontsize=20)
        plt.axhline(0, color='k')

        ax_v500_B1 = plt.subplot(337)
        ax_v500_B1.set_ylim([-0.8, 1.6])
        ax_v500_B1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v500_B1 = ax_v500_B1.plot(timestamps, data_v500_B1["Q"][str(depth_slices[i])], '-o', color='g', linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v500_B1["Q"][str(depth_slices[i])])
        mean_v500_B1 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v500_B1.text(-0.2, 0.96, 'g', ha='left', va="top", transform=ax_v500_B1.transAxes, fontsize=20, fontweight="bold")
        ax_v500_B1.text(1.02,
                        (Qmean - ax_v500_B1.get_ylim()[0]) / (ax_v500_B1.get_ylim()[1] - ax_v500_B1.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color='g',
                        horizontalalignment='left', verticalalignment='center', transform=ax_v500_B1.transAxes,
                        fontsize=20)
        plt.axhline(0, color='k')
        ax_v500_B1.set_ylabel("Q", fontsize=25)
        ax_v500_B1.text(-0.3, 0.5, 'Vswim = {:d}um'.format(data_v500_B1["V"]),
                horizontalalignment='center', verticalalignment='center',
                transform=ax_v500_B1.transAxes, fontsize=25, rotation=90)

        ax_v500_B3 = plt.subplot(338)
        ax_v500_B3.set_ylim([-0.8, 1.6])
        ax_v500_B3.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v500_B3 = ax_v500_B3.plot(timestamps, data_v500_B3["Q"][str(depth_slices[i])], '-o', color='g',
                                    linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v500_B3["Q"][str(depth_slices[i])])
        mean_v500_B3 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v500_B3.text(-0.2, 0.96, 'h', ha='left', va="top", transform=ax_v500_B3.transAxes, fontsize=20, fontweight="bold")
        ax_v500_B3.text(1.02,
                        (Qmean - ax_v500_B3.get_ylim()[0]) / (ax_v500_B3.get_ylim()[1] - ax_v500_B3.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color='g',
                        horizontalalignment='left', verticalalignment='center', transform=ax_v500_B3.transAxes,
                        fontsize=20)
        plt.axhline(0, color='k')

        ax_v500_B5 = plt.subplot(339)
        ax_v500_B5.set_ylim([-0.8, 1.6])
        ax_v500_B5.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        l_v500_B5 = ax_v500_B5.plot(timestamps, data_v500_B5["Q"][str(depth_slices[i])], '-o', color='g',
                                    linewidth=1.5, markersize=3, label='B=1.0')
        Qmean = np.mean(data_v500_B5["Q"][str(depth_slices[i])])
        mean_v500_B5 = plt.axhline(Qmean, color='g', linestyle=':', linewidth=1, label="mean")
        ax_v500_B5.text(-0.2, 0.96, 'i', ha='left', va="top", transform=ax_v500_B5.transAxes, fontsize=20, fontweight="bold")
        ax_v500_B5.text(1.02,
                        (Qmean - ax_v500_B5.get_ylim()[0]) / (ax_v500_B5.get_ylim()[1] - ax_v500_B5.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color='g',
                        horizontalalignment='left', verticalalignment='center', transform=ax_v500_B5.transAxes,
                        fontsize=20)
        plt.axhline(0, color='k')

        for subplt in [ax_v10_B1, ax_v10_B3, ax_v10_B5,
                       ax_v100_B1, ax_v100_B3, ax_v100_B5,
                       ax_v500_B1, ax_v500_B3, ax_v500_B5]:
            for tick in subplt.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
            for tick in subplt.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)


        # st = fig.suptitle("Q statistic over time (%s region) (f=%0.2f)" % (depth_slice_names[i], f), fontsize=28)
        st = fig.suptitle("%s" % (depth_slice_names[i]), fontsize=28)
        fig.tight_layout(pad=0.7)
        fig.subplots_adjust(top=0.88)
        # plt.show()
        fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/3x3allSims/separated depth regions/100000p_%0.2ff_Q_overtime_vs_Depth_%s_%s%s.png" % (f, depth_slice_names[i], avg_func.__name__, surfstring),
                    bbox_inches='tight')


if plot_3depthsx2agilestSims:
    # ...in a grid. Each row will contain plots from a different depth slice. Each column will contain plots
    # from sims with different B values.

    fig = plt.figure(figsize=(20, 16))

    if nosurf:
        st = fig.suptitle("Q statistic over time (excluding surface layer) (f=%0.2f)" % f, fontsize=32)
    else:
        st = fig.suptitle("Q statistic over time (including surface layer) (f=%0.2f)" % f, fontsize=32)

    colour = [0, 0.5, 0]

    # top depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 320  # for indexing subplots
    for sim in agilest_motile_concentrations:
        plot_index += 1
        ax_shallow = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_shallow.set_ylim([-0.21, 0.11])
            ax_shallow.set_yticks([-0.2, -0.1, 0, 0.1])
        elif f == 0.01:
            ax_shallow.set_ylim([-0.61, 0.41])
            ax_shallow.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_shallow.plot(timestamps, sim["Q"][str(depth_slices[-1])],
                                '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[-1])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_shallow.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                horizontalalignment='left', verticalalignment="bottom", transform=ax_shallow.transAxes, fontsize=18)
        ax_shallow.text(1.02, (Qmean - ax_shallow.get_ylim()[0]) / (ax_shallow.get_ylim()[1] - ax_shallow.get_ylim()[0]), "{:1.2f}".format(Qmean),
                color=colour, horizontalalignment='left', verticalalignment='center', transform=ax_shallow.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_shallow.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        for tick in ax_shallow.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        # add B & V values above columns and depth region labels alongside rows
        ax_shallow.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$, v = {:d}$\mu ms^{{-1}}$'.format(sim["B"], sim["V"]),
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax_shallow.transAxes, fontsize=25)
        if plot_index % 10 == 2:
            ax_shallow.text(1.25, 0.5, 'Shallow',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax_shallow.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_shallow.set_xlabel("Time [s]", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_shallow.set_ylabel("Q", fontsize=25)

    # mid depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 322  # for indexing subplots
    for sim in agilest_motile_concentrations:
        plot_index += 1
        ax_mid = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_mid.set_ylim([-0.21, 0.11])
            ax_mid.set_yticks([-0.2, -0.1, 0, 0.1])
        elif f == 0.01:
            ax_mid.set_ylim([-0.61, 0.41])
            ax_mid.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_mid.plot(timestamps, sim["Q"][str(depth_slices[1])],
                                '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[1])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_mid.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                        horizontalalignment='left', verticalalignment="bottom", transform=ax_mid.transAxes,
                        fontsize=18)
        ax_mid.text(1.02,
                        (Qmean - ax_mid.get_ylim()[0]) / (ax_mid.get_ylim()[1] - ax_mid.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color=colour, horizontalalignment='left', verticalalignment='center',
                        transform=ax_mid.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_mid.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_mid.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # add depth region labels alongside rows
        if plot_index % 10 == 4:
            ax_mid.text(1.25, 0.5, 'Mid',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_mid.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_mid.set_xlabel("Time", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_mid.set_ylabel("Q", fontsize=25)

    # Deep depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 324  # for indexing subplots
    for sim in agilest_motile_concentrations:
        plot_index += 1
        ax_deep = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_deep.set_ylim([-0.41, 0.41])
            ax_deep.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        elif f == 0.01:
            ax_deep.set_ylim([-0.81, 1.61])
            ax_deep.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_deep.plot(timestamps, sim["Q"][str(depth_slices[0])],
                            '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[0])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_deep.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                    horizontalalignment='left', verticalalignment="bottom", transform=ax_deep.transAxes,
                    fontsize=18)
        ax_deep.text(1.02,
                    (Qmean - ax_deep.get_ylim()[0]) / (ax_deep.get_ylim()[1] - ax_deep.get_ylim()[0]),
                    "{:1.2f}".format(Qmean),
                    color=colour, horizontalalignment='left', verticalalignment='center',
                    transform=ax_deep.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_deep.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_deep.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # add depth region labels alongside rows
        if plot_index % 10 == 6:
            ax_deep.text(1.25, 0.5, 'Deep',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_deep.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        if plot_index % 10 > 4:
            ax_deep.set_xlabel("Time", fontsize=25)
        if plot_index % 10 in [1, 3, 5]:
            ax_deep.set_ylabel("Q", fontsize=25)


    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # plt.show()
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/3x2agilestSims/100000p_%0.2ff_Q_overtime_%s%s.png" % (f, avg_func.__name__, surfstring))
    plt.clf()


if plot_3depthsx1boringSim:
    # ...in a grid. Each row will contain plots from a different depth slice. Each column will contain plots
    # from sims with different B values.

    fig = plt.figure(figsize=(14, 16))

    if nosurf:
        st = fig.suptitle("Q statistic over time (excluding surface layer) (f=%0.2f)" % f, fontsize=32)
    else:
        st = fig.suptitle("Q statistic over time (including surface layer) (f=%0.2f)" % f, fontsize=32)

    colour = [0, 0.5, 0]

    # top depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 311  # for indexing subplots
    for sim in boringest_motile_concentration:
        ax_shallow = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_shallow.set_ylim([-0.21, 0.11])
            ax_shallow.set_yticks([-0.2, -0.1, 0, 0.1])
        elif f == 0.01:
            ax_shallow.set_ylim([-0.61, 0.41])
            ax_shallow.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_shallow.plot(timestamps, sim["Q"][str(depth_slices[-1])],
                                '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[-1])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_shallow.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                        horizontalalignment='left', verticalalignment="bottom", transform=ax_shallow.transAxes,
                        fontsize=18)
        ax_shallow.text(1.02,
                        (Qmean - ax_shallow.get_ylim()[0]) / (ax_shallow.get_ylim()[1] - ax_shallow.get_ylim()[0]),
                        "{:1.2f}".format(Qmean),
                        color=colour, horizontalalignment='left', verticalalignment='center',
                        transform=ax_shallow.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_shallow.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_shallow.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # add B values above columns and depth region labels alongside rows
        ax_shallow.text(0.5, 1.1, r'B = {:1.1f}s$^{{-1}}$, v = {:d}$\mu ms^{{-1}}$'.format(sim["B"], sim["V"]),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_shallow.transAxes, fontsize=25)
        ax_shallow.text(1.15, 0.5, 'Shallow',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_shallow.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        ax_shallow.set_ylabel("Q", fontsize=25)

    # mid depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 312  # for indexing subplots
    for sim in boringest_motile_concentration:
        ax_mid = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_mid.set_ylim([-0.21, 0.11])
            ax_mid.set_yticks([-0.2, -0.1, 0, 0.1])
        elif f == 0.01:
            ax_mid.set_ylim([-0.61, 0.41])
            ax_mid.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_mid.plot(timestamps, sim["Q"][str(depth_slices[1])],
                            '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[1])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_mid.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                    horizontalalignment='left', verticalalignment="bottom", transform=ax_mid.transAxes,
                    fontsize=18)
        ax_mid.text(1.02,
                    (Qmean - ax_mid.get_ylim()[0]) / (ax_mid.get_ylim()[1] - ax_mid.get_ylim()[0]),
                    "{:1.2f}".format(Qmean),
                    color=colour, horizontalalignment='left', verticalalignment='center',
                    transform=ax_mid.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_mid.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_mid.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # add depth region labels alongside rows
        ax_mid.text(1.15, 0.5, 'Mid',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_mid.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        ax_mid.set_ylabel("Q", fontsize=25)

    # Deep depth slice ======================================================================================================
    plot_letter = '_abcdefghi'  # for labelling subplots
    plot_index = 313  # for indexing subplots
    for sim in boringest_motile_concentration:
        ax_deep = plt.subplot(plot_index)
        # set axis limits and tick locations
        if f == 0.1:
            ax_deep.set_ylim([-0.41, 0.41])
            ax_deep.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        elif f == 0.01:
            ax_deep.set_ylim([-0.81, 1.61])
            ax_deep.set_yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        # plot Q timeseries and mean value
        plt.axhline(0, color='k')
        Qline = ax_deep.plot(timestamps, sim["Q"][str(depth_slices[0])],
                             '-o', color=colour, linewidth=2, markersize=3)
        Qmean = np.mean(sim["Q"][str(depth_slices[0])])
        Q_meanline = plt.axhline(Qmean, color=colour, linestyle='--', linewidth=2)
        # annotate each subplot with a letter (for labelling) and the Q mean value
        ax_deep.text(0.04, 0.04, "(" + plot_letter[plot_index % 10] + ")",
                     horizontalalignment='left', verticalalignment="bottom", transform=ax_deep.transAxes,
                     fontsize=18)
        ax_deep.text(1.02,
                     (Qmean - ax_deep.get_ylim()[0]) / (ax_deep.get_ylim()[1] - ax_deep.get_ylim()[0]),
                     "{:1.2f}".format(Qmean),
                     color=colour, horizontalalignment='left', verticalalignment='center',
                     transform=ax_deep.transAxes, fontsize=20)
        # axis ticks fontsize handling
        for tick in ax_deep.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in ax_deep.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        # add depth region labels alongside rows
        ax_deep.text(1.15, 0.5, 'Deep',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_deep.transAxes, fontsize=25, rotation=270)
        # restrict axis labels to left column and bottom row
        ax_deep.set_xlabel("Time [s]", fontsize=25)
        ax_deep.set_ylabel("Q", fontsize=25)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # plt.show()
    fig.savefig(
        "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/3x1boringSim/100000p_%0.2ff_Q_overtime_%s%s.png" % (
        f, avg_func.__name__, surfstring))


if plot_3depthsx2boringVagileSims:
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
    ax_shallow_boring.text(0.5, 1.1, 'Non-Agile',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_shallow_boring.transAxes, fontsize=25)
    ax_shallow_boring.text(-0.15, 0.9, 'a',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_boring.transAxes, fontsize=25, fontweight="bold")
    Qmeans_shallow_boring = []
    for sim, i in zip(boring_motile_representatives, range(len(boring_motile_representatives))):
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
                                label=r'B = {:1.1f}s, v = {:d}$\mu ms^{{-1}}$'.format(sim["B"],sim["V"]))
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
    ax_shallow_agile.text(0.5, 1.1, 'Agile',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax_shallow_agile.transAxes, fontsize=25)
    ax_shallow_agile.text(-0.15, 0.9, 'b',
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax_shallow_agile.transAxes, fontsize=25, fontweight="bold")
    Qmeans_shallow_agile = []
    for sim, i in zip(agile_motile_representatives, range(len(agile_motile_representatives))):
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
                                      label=r'B = {:1.1f}s, v = {:d}$\mu ms^{{-1}}$'.format(sim["B"], sim["V"]))
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
    for sim, i in zip(boring_motile_representatives, range(len(boring_motile_representatives))):
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
    for sim, i in zip(agile_motile_representatives, range(len(agile_motile_representatives))):
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
    for sim, i in zip(boring_motile_representatives, range(len(boring_motile_representatives))):
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
    for sim, i in zip(agile_motile_representatives, range(len(agile_motile_representatives))):
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
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/Qovertime_vs_Depth/3x2agileVboringSims/100000p_%0.2ff_Q_overtime_%s%s.png" % (f, avg_func.__name__, surfstring))
    plt.clf()


if plot_violins:
    nonAgileSims = boring_motile_concentrations
    agileSims = agile_motile_concentrations
    nonAgileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])] for sim in nonAgileSims]), np.concatenate([sim['Q'][str(depth_slices[1])] for sim in nonAgileSims])]),
                          np.concatenate([sim['Q'][str(depth_slices[0])] for sim in nonAgileSims])]  # deep nonagile
    agileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])] for sim in agileSims]), np.concatenate([sim['Q'][str(depth_slices[1])] for sim in agileSims])]),
                       np.concatenate([sim['Q'][str(depth_slices[0])] for sim in agileSims])]  # deep agile
    # xlabels = ["Shallow+Mid\nSlow", "Shallow+Mid\nAgile", "Deep\nSlow", "Deep\nAgile"]
    ylabels = ["Deep\nAgile", "Deep\nNon-Agile", "Shallow+Mid\nAgile", "Shallow+Mid\nNon-Agile"]

    plt.clf()
    fig = plt.figure(figsize=(20, 15))
    ax = plt.gca()
    nonAgileViolins = ax.violinplot(nonAgileViolinData, positions=[2.4, 1.2], vert=False, showextrema=False, showmeans=False) #4,2
    agileViolins = ax.violinplot(agileViolinData, positions=[1.8, 0.6], vert=False, showextrema=False, showmeans=False) #3,1
    for b in nonAgileViolins['bodies']:
        b.set_facecolor('#d6604d')
        b.set_alpha(1)
    for b in agileViolins['bodies']:
        b.set_facecolor('#4393c3')
        b.set_alpha(1)
    # means = [np.mean(x) for x in [nonAgileViolinData[0], agileViolinData[0], nonAgileViolinData[1], agileViolinData[1]]]
    means = [np.mean(x) for x in [agileViolinData[1], nonAgileViolinData[1], agileViolinData[0], nonAgileViolinData[0]]]
    ax.axvline(0, ymin=0, ymax=1, color='k', lw=1)
    ax.vlines(means, ymin=[0.3, 0.9, 1.5, 2.1], ymax=[0.9, 1.5, 2.1, 2.7], colors='white', lw=2)
    ax.text(means[0]+0.01, 0.6, r'$\overline{{Q}}={:1.2f}$'.format(means[0]), color='white', fontsize=22,
            horizontalalignment="left", verticalalignment="top")
    ax.text(means[1]+0.01, 1.2, r'$\overline{{Q}}={:1.2f}$'.format(means[1]), color='white', fontsize=22,
            horizontalalignment="left", verticalalignment="top")
    ax.text(means[2]-0.01, 1.8, r'$\overline{{Q}}={:1.2f}$'.format(means[2]), color='white', fontsize=22,
            horizontalalignment="right", verticalalignment="center")
    ax.text(means[3]-0.01, 2.4, r'$\overline{{Q}}={:1.2f}$'.format(means[3]), color='white', fontsize=22,
            horizontalalignment="right", verticalalignment="center")
    ax.set_yticks([0.6, 1.2, 1.8, 2.4])
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', which='both', length=0)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(26)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(26)
    # ax.set_ylabel("Sample Type", fontsize=26)
    ax.set_xlabel("Q", fontsize=26)
    legend_patches = [mpatches.Patch(color='#d6604d'), mpatches.Patch(color='#4393c3')]
    legend_labels = ["Non-Agile", "Agile"]
    ax.legend(legend_patches, legend_labels, fontsize=24, loc="center right")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')

    # plt.show()
    fig.savefig(
            "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/violins/100000p_%0.2ff_violin_%s%s.png" % (
            f, avg_func.__name__, surfstring))