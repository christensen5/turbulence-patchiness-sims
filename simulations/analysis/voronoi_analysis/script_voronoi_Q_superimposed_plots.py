"""density analysis

This script takes two numpy files containing particle density data from a motile and a non-motile simulation.
The script will compute the distribution of the Q-statistic for density at each timestep, and plot the corresponding
patchiness plot."""
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
fig_dists.suptitle(r'Distribution of logged patch concentrations over time $(f=%0.2f, B=5.0s^{-1}, v=10ums^{-1})$' % f, fontsize=18)
fig_dists.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/C/5.0B_10v_Cdist_loglog_timesnaps.png")




timestamps = np.arange(0, 61, 1)#, 30]

Q = []
Cm = []
C_dead = []
C_v10_B1 = []
C_v10_B3 = []
C_v10_B5 = []
C_v100_B1 = []
C_v100_B3 = []
C_v100_B5 = []
C_v500_B1 = []
C_v500_B3 = []
C_v500_B5 = []

for t in tqdm(timestamps):
    conc_dead = concentrations_dead[:, t].flatten()
    conc_v10_B1 = concentrations_v10_B1[:, t].flatten()
    conc_v10_B3 = concentrations_v10_B3[:, t].flatten()
    conc_v10_B5 = concentrations_v10_B5[:, t].flatten()
    conc_v100_B1 = concentrations_v100_B1[:, t].flatten()
    conc_v100_B3 = concentrations_v100_B3[:, t].flatten()
    conc_v100_B5 = concentrations_v100_B5[:, t].flatten()
    conc_v500_B1 = concentrations_v500_B1[:, t].flatten()
    conc_v500_B3 = concentrations_v500_B3[:, t].flatten()
    conc_v500_B5 = concentrations_v500_B5[:, t].flatten()

    Cm_t = np.mean(conc_dead) #100000/(600*600*150)

    for f in [0.1]: #[0.01, 0.1, 0.5]:
        Cdead_t = conc_dead[conc_dead.argsort()[-int(f * conc_dead.size):]]
        # assert(np.allclose(C_v10_B1, C_v10_B1_0))
        # assert(np.allclose(Cdead, Cdead0))
        # print("f = %f" %f)
        # print("mean(C) = %f, mean(Cp) = %f" %(np.mean(C), np.mean(Cp)))
        # print("C/Cm = %f" %(np.mean(C)/np.mean(Cm)))
        # print("Cp/Cm = %f" % (np.mean(Cp) / np.mean(Cm)))
        # print("Q_f = %f \n" % Q_f)
        C_v10_B1_t = conc_v10_B1[conc_v10_B1.argsort()[-int(f * conc_v10_B1.size):]]
        C_v10_B3_t = conc_v10_B3[conc_v10_B3.argsort()[-int(f * conc_v10_B3.size):]]
        C_v10_B5_t = conc_v10_B5[conc_v10_B5.argsort()[-int(f * conc_v10_B5.size):]]
        C_v100_B1_t = conc_v100_B1[conc_v100_B1.argsort()[-int(f * conc_v100_B1.size):]]
        C_v100_B3_t = conc_v100_B3[conc_v100_B3.argsort()[-int(f * conc_v100_B3.size):]]
        C_v100_B5_t = conc_v100_B5[conc_v100_B5.argsort()[-int(f * conc_v100_B5.size):]]
        C_v500_B1_t = conc_v500_B1[conc_v500_B1.argsort()[-int(f * conc_v500_B1.size):]]
        C_v500_B3_t = conc_v500_B3[conc_v500_B3.argsort()[-int(f * conc_v500_B3.size):]]
        C_v500_B5_t = conc_v500_B5[conc_v500_B5.argsort()[-int(f * conc_v500_B5.size):]]

        Q_v10_B1 = (np.mean(C_v10_B1_t) - np.mean(Cdead_t)) / Cm_t
        Q_v10_B3 = (np.mean(C_v10_B3_t) - np.mean(Cdead_t)) / Cm_t
        Q_v10_B5 = (np.mean(C_v10_B5_t) - np.mean(Cdead_t)) / Cm_t
        Q_v100_B1 = (np.mean(C_v100_B1_t) - np.mean(Cdead_t)) / Cm_t
        Q_v100_B3 = (np.mean(C_v100_B3_t) - np.mean(Cdead_t)) / Cm_t
        Q_v100_B5 = (np.mean(C_v100_B5_t) - np.mean(Cdead_t)) / Cm_t
        Q_v500_B1 = (np.mean(C_v500_B1_t) - np.mean(Cdead_t)) / Cm_t
        Q_v500_B3 = (np.mean(C_v500_B3_t) - np.mean(Cdead_t)) / Cm_t
        Q_v500_B5 = (np.mean(C_v500_B5_t) - np.mean(Cdead_t)) / Cm_t


        Q_t = [Q_v10_B1, Q_v10_B3, Q_v10_B5, Q_v100_B1, Q_v100_B3, Q_v100_B5, Q_v500_B1, Q_v500_B3, Q_v500_B5]
    Q.append(Q_t)
    Cm.append(Cm_t)
    C_dead.append(np.mean(Cdead_t))
    C_v10_B1.append(np.mean(C_v10_B1_t))
    C_v10_B3.append(np.mean(C_v10_B3_t))
    C_v10_B5.append(np.mean(C_v10_B5_t))
    C_v100_B1.append(np.mean(C_v100_B1_t))
    C_v100_B3.append(np.mean(C_v100_B3_t))
    C_v100_B5.append(np.mean(C_v100_B5_t))
    C_v500_B1.append(np.mean(C_v500_B1_t))
    C_v500_B3.append(np.mean(C_v500_B3_t))
    C_v500_B5.append(np.mean(C_v500_B5_t))

Q = np.array(Q)
for list in [Cm, C_dead, C_v10_B1, C_v10_B3, C_v10_B5, C_v100_B1, C_v100_B3, C_v100_B5, C_v500_B1, C_v500_B3, C_v500_B5]:
    list = np.array(list)
# Q = np.log10(np.clip(Q, 1, None))

# C plotting
fig = plt.figure(figsize=(15, 9))
st = fig.suptitle("Patch concentration over time (f=%0.2f)" % f, fontsize=25)

colours = np.zeros((3, 3))
colours[1, :] = np.linspace(0, 1, 3)

plt.box(False)
ax0 = plt.subplot(111)
l0 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_dead, Cm)]), '-o', color='red', linewidth=2, markersize=3, label='non-motile')
l1 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B1, Cm)]), '-o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=10um')
l2 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B3, Cm)]), '-o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=10um')
l3 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B5, Cm)]), '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=10um')
l4 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B1, Cm)]), '--o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=100um')
l5 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B3, Cm)]), '--o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=100um')
l6 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B5, Cm)]), '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=100um')
l7 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B1, Cm)]), ':o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=500um')
l8 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B3, Cm)]), ':o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=500um')
l9 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B5, Cm)]), ':o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=500um')
plt.hlines(0, ax0.get_xlim()[0], ax0.get_xlim()[1], 'k')
ax0.set_xlabel("Time", fontsize=25)
ax0.set_ylabel("Concentration (C/Cm)", fontsize=25)
for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax0.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax0.legend(fontsize=15)
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/100000p_%0.2ff_concs_over_time_mean.png" %f)
fig.close()

# Q plotting
fig = plt.figure(figsize=(15, 9))
st = fig.suptitle("Q statistic over time (f=%0.2f)" % f, fontsize=25)

colours = np.zeros((3, 3))
colours[1, :] = np.linspace(0, 1, 3)

plt.box(False)
ax1 = plt.subplot(311)
l1 = ax1.plot(timestamps, Q[:, 0], '-o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=10um')
l2 = ax1.plot(timestamps, Q[:, 1], '-o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=10um')
l3 = ax1.plot(timestamps, Q[:, 2], '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=10um')
plt.hlines(0, ax1.get_xlim()[0], ax1.get_xlim()[1], 'k')
ax1.set_xlabel("Time", fontsize=25)
ax1.set_ylabel("Q", fontsize=25)
for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax1.legend(fontsize=15)
ax2 = plt.subplot(312)
l1 = ax2.plot(timestamps, Q[:, 3], '--o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=100um')
l2 = ax2.plot(timestamps, Q[:, 4], '--o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=100um')
l3 = ax2.plot(timestamps, Q[:, 5], '--o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=100um')
plt.hlines(0, ax2.get_xlim()[0], ax2.get_xlim()[1], 'k')
ax2.set_xlabel("Time", fontsize=25)
ax2.set_ylabel("Q", fontsize=25)
for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax2.legend(fontsize=15)
ax3 = plt.subplot(313)
l1 = ax3.plot(timestamps, Q[:, 6], ':o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=500um')
l2 = ax3.plot(timestamps, Q[:, 7], ':o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=500um')
l3 = ax3.plot(timestamps, Q[:, 8], ':o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=500um')
plt.hlines(0, ax3.get_xlim()[0], ax3.get_xlim()[1], 'k')
ax3.set_xlabel("Time", fontsize=25)
ax3.set_ylabel("Q", fontsize=25)
for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax3.legend(fontsize=15)

# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/100000p_%0.2ff_Q_over_time_mean.png" %f)
