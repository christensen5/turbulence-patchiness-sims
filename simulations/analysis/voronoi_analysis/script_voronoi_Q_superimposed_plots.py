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
voronoi_dead = np.load(os.path.join(filepath_dead, 'vols_v.npy'))
voronoi_v10_B1 = np.load(os.path.join(filepath_v10_B1, 'vols_v.npy'))
voronoi_v10_B3 = np.load(os.path.join(filepath_v10_B3, 'vols_v.npy'))
voronoi_v10_B5 = np.load(os.path.join(filepath_v10_B5, 'vols_v.npy'))
voronoi_v100_B1 = np.load(os.path.join(filepath_v100_B1, 'vols_v.npy'))
voronoi_v100_B3 = np.load(os.path.join(filepath_v100_B3, 'vols_v.npy'))
voronoi_v100_B5 = np.load(os.path.join(filepath_v100_B5, 'vols_v.npy'))
voronoi_v500_B1 = np.load(os.path.join(filepath_v500_B1, 'vols_v.npy'))
voronoi_v500_B3 = np.load(os.path.join(filepath_v500_B3, 'vols_v.npy'))
voronoi_v500_B5 = np.load(os.path.join(filepath_v500_B5, 'vols_v.npy'))

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


ones = np.ones_like(Cm)
plt.box(False)
ax0 = plt.subplot(111)
l0 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_dead, ones)]), '-o', color='red', linewidth=2, markersize=3, label='non-motile')
l1 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B1, ones)]), '-o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=10um')
l2 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B3, ones)]), '-o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=10um')
l3 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v10_B5, ones)]), '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=10um')
l4 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B1, ones)]), '--o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=100um')
l5 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B3, ones)]), '--o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=100um')
l6 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v100_B5, ones)]), '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=100um')
l7 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B1, ones)]), ':o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0 v=500um')
l8 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B3, ones)]), ':o', color=colours[:, 1], linewidth=2, markersize=3, label='B=3.0 v=500um')
l9 = ax0.plot(timestamps, np.log10([i / j for i, j in zip(C_v500_B5, ones)]), ':o', color=colours[:, 2], linewidth=2, markersize=3, label='B=5.0 v=500um')
# plt.hlines(0, ax0.get_xlim()[0], ax0.get_xlim()[1], 'k')
ax0.set_xlabel("Time", fontsize=25)
ax0.set_ylabel("log(Concentration)", fontsize=25)
for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax0.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax0.legend(fontsize=15)
# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/C/100000p_%0.2ff_concs_over_time_mean.png" %f)
fig.close()
