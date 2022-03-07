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
plot_3depthsx2boringVagileSims = False
plot_violins = True

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

timestamps = list(range(61))

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
# Plot violins

for t_end in [21, 31, 41, 51]: # make one violin plot for each of the timesteps calculated above
    nonAgileSims = boring_motile_concentrations
    agileSims = agile_motile_concentrations
    nonAgileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])][0:t_end] for sim in nonAgileSims]), np.concatenate([sim['Q'][str(depth_slices[1])][0:t_end] for sim in nonAgileSims])]),
                          np.concatenate([sim['Q'][str(depth_slices[0])][0:t_end] for sim in nonAgileSims])] # deep nonagile
    agileViolinData = [np.concatenate([np.concatenate([sim['Q'][str(depth_slices[2])][0:t_end] for sim in agileSims]), np.concatenate([sim['Q'][str(depth_slices[1])][0:t_end] for sim in agileSims])]),
                          np.concatenate([sim['Q'][str(depth_slices[0])][0:t_end] for sim in agileSims])] # deep agile
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
    ax.set_xlim([-0.8, 1.7])
    ax.set_xticks([-0.5, 0., 0.5, 1., 1.5])
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
            "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/violins/100000p_%0.2ff_violin_%s%s_0-%ds.png" % (
            f, avg_func.__name__, surfstring, t_end))