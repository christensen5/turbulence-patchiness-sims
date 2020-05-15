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
densities_dead = {'data': np.load(os.path.join(filepath_dead, 'density_pwise.npy')), 'V': 'dead', 'B': 'dead'}
densities_v10_B1 = {'data': np.load(os.path.join(filepath_v10_B1, 'density_pwise.npy')), 'V': 10, 'B': 1.0}
densities_v10_B3 = {'data': np.load(os.path.join(filepath_v10_B3, 'density_pwise.npy')), 'V': 10, 'B': 3.0}
densities_v10_B5 = {'data': np.load(os.path.join(filepath_v10_B5, 'density_pwise.npy')), 'V': 10, 'B': 5.0}
densities_v100_B1 = {'data': np.load(os.path.join(filepath_v100_B1, 'density_pwise.npy')), 'V': 100, 'B': 1.0}
densities_v100_B3 = {'data': np.load(os.path.join(filepath_v100_B3, 'density_pwise.npy')), 'V': 100, 'B': 3.0}
densities_v100_B5 = {'data': np.load(os.path.join(filepath_v100_B5, 'density_pwise.npy')), 'V': 100, 'B': 5.0}
densities_v500_B1 = {'data': np.load(os.path.join(filepath_v500_B1, 'density_pwise.npy')), 'V': 500, 'B': 1.0}
densities_v500_B3 = {'data': np.load(os.path.join(filepath_v500_B3, 'density_pwise.npy')), 'V': 500, 'B': 3.0}
densities_v500_B5 = {'data': np.load(os.path.join(filepath_v500_B5, 'density_pwise.npy')), 'V': 500, 'B': 5.0}

all_motile_densities = [densities_v10_B1, densities_v10_B3, densities_v10_B5,
                        densities_v100_B1, densities_v100_B3, densities_v100_B5,
                        densities_v500_B1, densities_v500_B3, densities_v500_B5]

# ======================================================================================================================
# PLOT Q-DISTRIBUTION AT 6 TIMESNAPS.

avg_func = np.mean
if not (avg_func.__name__ == 'mean' or avg_func.__name__ == 'median'):
    raise NotImplementedError("Q-analysis must use either mean or median, not %s." % avg_func.__name__)

# find max and min densities (for plot limits)
xmin = densities_dead["data"].min()
xmax = densities_dead["data"].max()
for simdata_dict in all_motile_densities:
    mindensity = np.amin(simdata_dict["data"])
    maxdensity = np.amax(simdata_dict["data"])
    if mindensity < xmin:
        xmin = mindensity
    if maxdensity > xmax:
        xmax = maxdensity

f = 0.5
timestamps_for_dists = np.arange(0, 6, 1)  # density_high.npy was only computed for 6 timestamps (0-60s in 12s steps)
xlims = [-.5, 16]  #[0, 1.2]
ylims = (1e0, 1e4)

# Q distribution plots
for simdata_dict in tqdm(all_motile_densities):
    fig_dists = plt.figure(figsize=(15, 9))
    plotindex = 231
    for t in timestamps_for_dists:
        conc_dead = densities_dead["data"][:, t]
        conc_mot = simdata_dict["data"][:, t]
        # keep only particles contained in patches, defined by f.
        C_dead = conc_dead[conc_dead.argsort()[-int(f * conc_dead.size):]]
        C_mot = conc_mot[conc_mot.argsort()[-int(f * conc_mot.size):]]
        Cm = avg_func(conc_dead)
        # compute Q distribution & Q
        Qdist = (C_mot - C_dead) / Cm
        Q = (avg_func(C_mot) - avg_func(C_dead)) / Cm
        # # for logging
        # Qdist = np.log10(np.clip(Qdist, 1, None))
        # Q = np.log10(np.clip(Q, 1, None))
        # subplot the Q-distribution at this timestep
        ax = plt.subplot(plotindex)
        hgram = ax.hist(Qdist, 100, range=xlims, log=True)
        plt.text(0.8, 0.9, str(t*12)+"s", fontsize=16, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        ax.set_ylim(ylims)
        ax.hlines(1, ax.get_xlim()[0], ax.get_xlim()[1], 'k')
        ax.vlines(0, ylims[0], ylims[1], 'k')
        ax.vlines(Q, ylims[0], ylims[1], 'r')
        if int(str(plotindex)[-1]) > 3:
            ax.set_xlabel(r'Q', fontsize=15)
        if int(str(plotindex)[-1]) % 3 == 1:
            ax.set_ylabel("Count", fontsize=15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        plt.box(False)
        plotindex += 1
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='Q')]
    fig_dists.legend(handles=legend_elements, loc='center right')
    fig_dists.suptitle(r'Distribution of cell-wise patch concentration enhancement factor (Q) over time $(f=%3.2f, B=%3.1fs^{-1}, v=%dums^{-1})$' % (f, simdata_dict["B"], simdata_dict["V"]), fontsize=18)
    fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/density_pwise/Q/Qdist_pwise_f=%3.2f_%3.1fB_%dv_semilog_timesnaps_%s.png" % (f, simdata_dict["B"], simdata_dict["V"], avg_func.__name__))



