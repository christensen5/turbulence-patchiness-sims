"""density analysis

This script takes two numpy files containing particle density data from a motile and a non-motile simulation.
The script will compute the distribution of the Q-statistic for density at each timestep, and plot the corresponding
patchiness plot."""
import os
import numpy as np
from tqdm import tqdm
from simulations.analysis.analysis_tools import plot_densities
import matplotlib.pyplot as plt


filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/100000p_30s_0.01dt_0.05sdt_initunif_dead"
filepath_mot = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.1vswim/sim2"

deps_dead = np.load(os.path.join(filepath_dead, 'deps.npy'))
deps_mot = np.load(os.path.join(filepath_mot, 'deps.npy'))
voronoi_volumes_dead = np.load(os.path.join(filepath_dead, 'vols.npy'))
voronoi_volumes_mot = np.load(os.path.join(filepath_mot, 'vols.npy'))

concentrations_dead = np.reciprocal(voronoi_volumes_dead)
concentrations_mot = np.reciprocal(voronoi_volumes_mot)

timestamps = np.arange(0, 31, 1)#, 30]

Q = []

for t in tqdm(timestamps):
    conc_dead = concentrations_dead[deps_dead[t, :] > 1, t].flatten()
    conc_mot = concentrations_mot[deps_mot[t, :] > 1, t].flatten()
    Cm = np.sum(conc_dead)/conc_dead.size #100000/(600*600*150)
    Q_t = []
    for f in [0.01, 0.1, 0.5]:
        C = conc_mot[conc_mot.argsort()[-int(f * conc_mot.size):]]
        C0 = np.sort(conc_mot)[-int(f * conc_mot.size):]
        Cp = conc_dead[conc_dead.argsort()[-int(f * conc_dead.size):]]
        Cp0 = np.sort(conc_dead)[-int(f * conc_dead.size):]
        assert(np.allclose(C, C0))
        assert(np.allclose(Cp, Cp0))
        # print("f = %f" %f)
        # print("mean(C) = %f, mean(Cp) = %f" %(np.mean(C), np.mean(Cp)))
        # print("C/Cm = %f" %(np.mean(C)/np.mean(Cm)))
        # print("Cp/Cm = %f" % (np.mean(Cp) / np.mean(Cm)))
        Q_f = (np.mean(C) - np.mean(Cp)) / Cm
        # print("Q_f = %f \n" % Q_f)
        Q_t.append(Q_f)
    Q.append(Q_t)

Q = np.array(Q)
Q = np.log10(np.clip(Q, 1, None))

fig = plt.figure(figsize=(15, 9))
st = fig.suptitle("Q statistic over time for differing f-values", fontsize=25)

plt.box(False)
ax1 = plt.subplot(131)
l1 = ax1.plot(timestamps, Q[:, 0], '-*', color='orange', linewidth=2, markersize=3, label='f=0.01')
plt.hlines(0, ax1.get_xlim()[0], ax1.get_xlim()[1], 'k')
ax2 = plt.subplot(132)
l2 = ax2.plot(timestamps, Q[:, 1], '-bo', linewidth=2, markersize=3, label='f=0.1')
plt.hlines(0, ax2.get_xlim()[0], ax2.get_xlim()[1], 'k')
ax3 = plt.subplot(133)
l3 = ax3.plot(timestamps, Q[:, 2], '-cv', linewidth=2, markersize=3, label='f=0.5')
plt.hlines(0, ax3.get_xlim()[0], ax3.get_xlim()[1], 'k')
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("Q", fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    ax.legend(fontsize=15)

plt.show()
# fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/comparison/vor/10000p_Q_over_time")