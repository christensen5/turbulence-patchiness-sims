import os
import numpy as np
from tqdm import tqdm
from simulations.analysis.analysis_tools import plot_densities
import matplotlib.pyplot as plt

# specify paths to simulation output files
filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead"
# filepath_mot = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot"
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
voronoi_dead = np.load(os.path.join(filepath_dead, 'vols.npy'))
voronoi_v10_B1 = np.load(os.path.join(filepath_v10_B1, 'vols.npy'))
voronoi_v10_B3 = np.load(os.path.join(filepath_v10_B3, 'vols.npy'))
voronoi_v10_B5 = np.load(os.path.join(filepath_v10_B5, 'vols.npy'))
voronoi_v100_B1 = np.load(os.path.join(filepath_v100_B1, 'vols.npy'))
voronoi_v100_B3 = np.load(os.path.join(filepath_v100_B3, 'vols.npy'))
voronoi_v100_B5 = np.load(os.path.join(filepath_v100_B5, 'vols.npy'))
voronoi_v500_B1 = np.load(os.path.join(filepath_v500_B1, 'vols.npy'))
voronoi_v500_B3 = np.load(os.path.join(filepath_v500_B3, 'vols.npy'))
voronoi_v500_B5 = np.load(os.path.join(filepath_v500_B5, 'vols.npy'))

concentrations_dead = np.reciprocal(voronoi_dead)
concentrations_v10_B1 = np.reciprocal(voronoi_v10_B1)
concentrations_v10_B3 = np.reciprocal(voronoi_v10_B3)
concentrations_v10_B5 = np.reciprocal(voronoi_v10_B5)
concentrations_v100_B1 = np.reciprocal(voronoi_v100_B1)
concentrations_v100_B3 = np.reciprocal(voronoi_v100_B3)
concentrations_v100_B5 = np.reciprocal(voronoi_v100_B5)
concentrations_v500_B1 = np.reciprocal(voronoi_v500_B1)
concentrations_v500_B3 = np.reciprocal(voronoi_v500_B3)
concentrations_v500_B5 = np.reciprocal(voronoi_v500_B5)

f = 1.
timestamps_for_dists = np.arange(0, 61, 12)
fig_dists = plt.figure(figsize=(15, 9))
plotindex = 231
xlims = (np.log10(concentrations_v500_B1).min(), np.log10(concentrations_v500_B1).max())
ylims = (1e0, 1e5)
for t in timestamps_for_dists:
    conc_dead = concentrations_dead[:, t].flatten()
    conc_mot = np.log10(concentrations_v10_B5[:, t].flatten())
    Cm_t = np.mean(np.log10(conc_dead))
    ax = plt.subplot(plotindex)
    hgram = ax.hist(conc_mot, 100, range=xlims, log=True)
    plt.text(0.8, 0.9, str(t)+"s", fontsize=16, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_ylim(ylims)
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], 'k')
    ax.vlines(Cm_t, ylims[0], ylims[1], 'r')
    if int(str(plotindex)[-1]) > 3:
        ax.set_xlabel(r'log(Concentration)', fontsize=15)
    if int(str(plotindex)[-1]) % 3 == 1:
        ax.set_ylabel("Count", fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    plt.box(False)
    plotindex += 1
fig_dists.suptitle(r'Distribution of logged patch voronoi concentrations over time $(f=%0.2f, B=5.0s^{-1}, v=10ums^{-1})$' % f, fontsize=18)
fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/C/5.0B_10v_Cdist_loglog_timesnaps.png")