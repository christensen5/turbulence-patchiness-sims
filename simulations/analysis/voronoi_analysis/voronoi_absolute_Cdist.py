"""
This script plots histograms of the Q-statistic of all our simulations at different timesteps. The Q-statistic here
is computed using the volumes of the Voronoi tessellation of the particle positions at each timestep.
"""

import numpy as np
import scipy.stats
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

# 10s window of peak Q for each sim, converted to "0.1s-timesteps after 20s"
t_offset = -200
# peaks
data_v10_B1['peak_Q_timesteps'] = (np.arange(41, 51, 0.1) * 10 + t_offset).astype("int32")
data_v10_B3['peak_Q_timesteps'] = (np.arange(42, 52, 0.1) * 10 + t_offset).astype("int32")
data_v10_B5['peak_Q_timesteps'] = (np.arange(36, 46, 0.1) * 10 + t_offset).astype("int32")
data_v100_B1['peak_Q_timesteps'] = (np.arange(40, 50, 0.1) * 10 + t_offset).astype("int32")
data_v100_B3['peak_Q_timesteps'] = (np.arange(42, 52, 0.1) * 10 + t_offset).astype("int32")
data_v100_B5['peak_Q_timesteps'] = (np.arange(42, 52, 0.1) * 10 + t_offset).astype("int32")
data_v500_B1['peak_Q_timesteps'] = (np.arange(27, 37, 0.1) * 10 + t_offset).astype("int32")
data_v500_B3['peak_Q_timesteps'] = (np.arange(33, 43, 0.1) * 10 + t_offset).astype("int32")
data_v500_B5['peak_Q_timesteps'] = (np.arange(40, 50, 0.1) * 10 + t_offset).astype("int32")
# nadirs
data_v10_B1['nadir_Q_timesteps'] = (np.arange(26, 36, 0.1) * 10 + t_offset).astype("int32")
data_v10_B3['nadir_Q_timesteps'] = (np.arange(20, 30, 0.1) * 10 + t_offset).astype("int32")
data_v10_B5['nadir_Q_timesteps'] = (np.arange(24, 34, 0.1) * 10 + t_offset).astype("int32")
data_v100_B1['nadir_Q_timesteps'] = (np.arange(50, 60, 0.1) * 10 + t_offset).astype("int32")
data_v100_B3['nadir_Q_timesteps'] = (np.arange(30, 40, 0.1) * 10 + t_offset).astype("int32")
data_v100_B5['nadir_Q_timesteps'] = (np.arange(29, 39, 0.1) * 10 + t_offset).astype("int32")
data_v500_B1['nadir_Q_timesteps'] = (np.arange(50, 60, 0.1) * 10 + t_offset).astype("int32")
data_v500_B3['nadir_Q_timesteps'] = (np.arange(50, 60, 0.1) * 10 + t_offset).astype("int32")
data_v500_B5['nadir_Q_timesteps'] = (np.arange(29, 39, 0.1) * 10 + t_offset).astype("int32")

all_motile_concentrations = [data_v10_B1, data_v10_B3, data_v10_B5,
                             data_v100_B1, data_v100_B3, data_v100_B5,
                             data_v500_B1, data_v500_B3, data_v500_B5]

# ======================================================================================================================
# PLOT absolute C-DISTRIBUTION AT 401 TIMESTEPS BETWEEN 20s-60s.

avg_func = np.median
if not (avg_func.__name__ == 'mean' or avg_func.__name__ == 'median'):
    raise NotImplementedError("Q-analysis must use either mean or median, not %s." % avg_func.__name__)

f = 0.01

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

timesteps = np.arange(50, 401, 1)  # timestamps over which to count concentrations. vols_v.npy was computed for 401 timestamps (20-60s in 0.1s steps)
xlims = [-.5, 1500]  #[0, 1.2]
ylims = (1e0, 1e5)

# # Non-motile first.
# C_dead_all_times = []
# fig_dists = plt.figure(figsize=(15, 9))
# for t in tqdm(timesteps):
#     conc_dead = data_dead["concs"][:, t].flatten()
#     if nosurf:
#         depths_dead = data_dead["depth"][:, t].flatten()
#         conc_dead = conc_dead[depths_dead < 300]  # keep concs only of nosurf particles
#         depths_dead = depths_dead[depths_dead < 300]  # keep depths only of nosurf particles
#     # keep only particles contained in patches, defined by f.
#     nparticles = conc_dead.size  # voro++ may have truncated pset by a few particles so choose smallest array
#     patch_inds = conc_dead.argsort()[-int(f * nparticles):]
#     C_dead = conc_dead[patch_inds]  # keep conc only of particles in patches
#     depths_dead_patches = depths_dead[patch_inds]  # # keep depths only of particles in patches
#     # keep only particles in patches within the Deep region.
#     C_dead_all_times.append(C_dead[depths_dead_patches < 170])
# C_dead_all_times = np.concatenate(C_dead_all_times, axis=0)  # unpack list into np array
#
# # absolute concentration C
# absC_dead = C_dead_all_times * 5717.87 # convert to microbes per mililitre
# # plot the absC-distribution for this simulation
# ax = plt.subplot(111)
# hgram = ax.hist(absC_dead, 100, range=xlims, log=True, color="green")
# ax.set_xlim([0, 2500])
# ax.vlines(avg_func(absC_dead), ylims[0], ylims[1], 'green', linestyles="--", lw=2)
# ax.annotate(" %3.2f" % avg_func(absC_dead), xy=[avg_func(absC_dead), 1e5], xycoords="data", ha="left", va="top", fontsize=15, color="green")
# # ax.set_xticks([0, 5, 10, 15])
# ax.set_xlabel('Absolute concentration'+r'[$mL^{{-1}}$]', fontsize=18)
# ax.set_ylabel("Count", fontsize=18)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(16)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(16)
# legend_elements = [Line2D([0], [0], color='green', lw=2, ls="--", label='median')]
# fig_dists.legend(handles=legend_elements, loc=(0.75, 0.5), fontsize=18)
# if nosurf:
#     fig_dists.suptitle(
#         r'Distribution of Voronoi-based absolute microbe concentration within patches (excl. surface particles) ($f=%3.2f$, non-motile microbes)' % (
#         f), fontsize=18,
#     wrap=True)
# else:
#     fig_dists.suptitle(
#         r'Distribution of Voronoi-based absolute microbe concentration within patches (incl. surface particles) $(f=%3.2f$, non-motile microbes)' % (
#             f), fontsize=18,
#     wrap=True)
# fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/absolute_concentration/absCdist_vor_%3.2ff_dead_semilog_timesnaps_%s%s.png" % (f, avg_func.__name__, surfstring))

# Now all motile sims (Q peak plots)
for simdata_dict in tqdm(all_motile_concentrations):
    fig_dists = plt.figure(figsize=(15, 9))
    C_dead_all_times = []
    C_mot_all_times = []
    for t in simdata_dict["peak_Q_timesteps"]:
        conc_dead = data_dead["concs"][:, t].flatten()
        conc_mot = simdata_dict["concs"][:, t].flatten()
        if nosurf:
            depths_dead = data_dead["depth"][:, t].flatten()
            conc_dead = conc_dead[depths_dead < 300]  # keep concs only of nosurf particles
            depths_dead = depths_dead[depths_dead < 300]  # keep depths only of nosurf particles
            depths_mot = simdata_dict["depth"][:, t].flatten()
            conc_mot = conc_mot[depths_mot < 300]  # keep concs only of nosurf particles
            depths_mot = depths_mot[depths_mot < 300]  # keep depths only of nosurf particles
        # keep only particles contained in patches, defined by f.
        nparticles = min(conc_dead.size, conc_mot.size)  # voro++ may have truncated pset by a few particles so choose smallest array
        patch_inds_dead = conc_dead.argsort()[-int(f * nparticles):]
        patch_inds_mot = conc_mot.argsort()[-int(f * nparticles):]
        C_dead = conc_dead[patch_inds_dead]  # keep conc only of particles in patches
        depths_dead_patches = depths_dead[patch_inds_dead]  # # keep depths only of particles in patches
        C_mot = conc_mot[patch_inds_mot]  # keep conc only of particles in patches
        depths_mot_patches = depths_mot[patch_inds_mot]  # keep depths only of particles in patches
        # keep only particles in patches within the Deep region.
        C_dead_all_times.append(C_dead[depths_dead_patches < 170])
        C_mot_all_times.append(C_mot[depths_mot_patches < 170])
    C_dead_all_times = np.concatenate(C_dead_all_times, axis=0)  # unpack list into np array
    C_mot_all_times = np.concatenate(C_mot_all_times, axis=0)  # unpack list into np array

    # absolute concentration C
    absC_dead = C_dead_all_times * 5717.87  # convert to microbes per mililitre
    absC = C_mot_all_times * 5717.87  # convert to microbes per mililitre
    # plot the absC-distribution for this simulation
    ax = plt.subplot(111)
    hgram = ax.hist(absC, 100, range=xlims, log=True)
    hgram_dead = ax.hist(absC_dead, 100, range=xlims, log=True, alpha=0.5, color="green")
    ax.set_xlim([0, 1000])
    ax.set_xlabel('Absolute concentration'+r'[$mL^{{-1}}$]', fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.annotate("a", xy=(0, 1), color='k', xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='top', fontsize=20, fontweight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    if nosurf:
        ax.set_title(
            'Distribution of Voronoi-based absolute microbe concentration within patches when Q is high \n (excl. surface particles) (f=%3.2f, B=%3.1fs, v=%d' % (
            f, simdata_dict["B"], simdata_dict["V"]) + r'$\mu m s^{{-1}})$', fontsize=20,
        wrap=True)
    else:
        fig_dists.title(
            'Distribution of Voronoi-based absolute microbe concentration within patches when Q is high \n (incl. surface particles)(f=%3.2f, B=%3.1fs, v=%d' % (
            f, simdata_dict["B"], simdata_dict["V"]) + r'$\mu m s^{{-1}})$', fontsize=20,
        wrap=True)
    fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/absolute_concentration/Qpeak/absCdist_vor_%3.2ff_%3.1fB_%dv_semilog_%s%s.png" % (f, simdata_dict["B"], simdata_dict["V"], avg_func.__name__, surfstring))
    
    
# Now all motile sims (Q nadir plots)
for simdata_dict in tqdm(all_motile_concentrations):
    fig_dists = plt.figure(figsize=(15, 9))
    C_dead_all_times = []
    C_mot_all_times = []
    for t in simdata_dict["nadir_Q_timesteps"]:
        conc_dead = data_dead["concs"][:, t].flatten()
        conc_mot = simdata_dict["concs"][:, t].flatten()
        if nosurf:
            depths_dead = data_dead["depth"][:, t].flatten()
            conc_dead = conc_dead[depths_dead < 300]  # keep concs only of nosurf particles
            depths_dead = depths_dead[depths_dead < 300]  # keep depths only of nosurf particles
            depths_mot = simdata_dict["depth"][:, t].flatten()
            conc_mot = conc_mot[depths_mot < 300]  # keep concs only of nosurf particles
            depths_mot = depths_mot[depths_mot < 300]  # keep depths only of nosurf particles
        # keep only particles contained in patches, defined by f.
        nparticles = min(conc_dead.size, conc_mot.size)  # voro++ may have truncated pset by a few particles so choose smallest array
        patch_inds_dead = conc_dead.argsort()[-int(f * nparticles):]
        patch_inds_mot = conc_mot.argsort()[-int(f * nparticles):]
        C_dead = conc_dead[patch_inds_dead]  # keep conc only of particles in patches
        depths_dead_patches = depths_dead[patch_inds_dead]  # # keep depths only of particles in patches
        C_mot = conc_mot[patch_inds_mot]  # keep conc only of particles in patches
        depths_mot_patches = depths_mot[patch_inds_mot]  # keep depths only of particles in patches
        # keep only particles in patches within the Deep region.
        C_dead_all_times.append(C_dead[depths_dead_patches < 170])
        C_mot_all_times.append(C_mot[depths_mot_patches < 170])
    C_dead_all_times = np.concatenate(C_dead_all_times, axis=0)  # unpack list into np array
    C_mot_all_times = np.concatenate(C_mot_all_times, axis=0)  # unpack list into np array

    # absolute concentration C
    absC_dead = C_dead_all_times * 5717.87  # convert to microbes per mililitre
    absC = C_mot_all_times * 5717.87 # convert to microbes per mililitre
    # plot the absC-distribution for this simulation
    ax = plt.subplot(111)
    hgram = ax.hist(absC, 100, range=xlims, log=True)
    hgram_dead = ax.hist(absC_dead, 100, range=xlims, log=True, alpha=0.5, color="green")
    ax.set_xlim([0, 1000])
    ax.set_xlabel('Absolute concentration'+r'[$mL^{{-1}}$]', fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.annotate("b", xy=(0, 1), color='k', xycoords='figure fraction',
                        horizontalalignment='left', verticalalignment='top', fontsize=20, fontweight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    if nosurf:
        ax.set_title(
            'Distribution of Voronoi-based absolute microbe concentration within patches when Q is low \n (excl. surface particles) (f=%3.2f, B=%3.1fs, v=%d' % (
            f, simdata_dict["B"], simdata_dict["V"]) + r'$\mu m s^{{-1}})$', fontsize=20,
        wrap=True)
    else:
        fig_dists.title(
            'Distribution of Voronoi-based absolute microbe concentration within patches when Q is low \n (incl. surface particles)(f=%3.2f, B=%3.1fs, v=%d' % (
            f, simdata_dict["B"], simdata_dict["V"]) + r'$\mu m s^{{-1}})$', fontsize=20,
        wrap=True)
    fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/Q/absolute_concentration/Qnadir/absCdist_vor_%3.2ff_%3.1fB_%dv_semilog_%s%s.png" % (f, simdata_dict["B"], simdata_dict["V"], avg_func.__name__, surfstring))



