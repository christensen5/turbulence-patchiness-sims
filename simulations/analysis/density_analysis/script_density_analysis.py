"""density analysis

This script takes two numpy files containing particle density data from a motile and a non-motile simulation.
The script will compute the distribution of the Q-statistic for density at each timestep, and plot the corresponding
patchiness plot."""
import numpy as np
from tqdm import tqdm
from simulations.analysis.analysis_tools import plot_densities
import matplotlib.pyplot as plt

filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/trajectories_10000p_30s_0.01dt_0.1sdt_initunif_dead_density.npy"
filepath_mot = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/trajectories_10000p_30s_0.01dt_0.05sdt_initunif_mot_density.npy"

densities_dead = np.load(filepath_dead)
densities_mot = np.load(filepath_mot)

timestamps = np.arange(0, 31, 1)#, 30]

f = 0.1
Q = []
for t in tqdm(timestamps):
    density_dead = densities_dead[:, :, :, t].flatten()
    density_mot = densities_mot[:, :, :, t].flatten()
    Cm = np.sum(density_dead)/density_dead.size
    Q_t = []
    fig = plt.figure(figsize=(12, 9))
    i = -1
    for f in [0.01, 0.1, 0.5, 0.001]:
        i += 1
        C = density_mot[density_mot.argsort()[-int(f * density_mot.size):]]
        Cp = density_dead[density_dead.argsort()[-int(f * density_dead.size):]]

        Q_f = (np.mean(C) - np.mean(Cp)) / Cm
        Q_t.append(Q_f)
        if i < 2:
            xlims = [0.3, 0.5]
            ylims = [400, 1000]
            text_x = [0.1, 0.2]
            text_y = [200, 500]
            Q_hist, bin_edges = np.histogram((C-Cp)/Cm, 100)
            plt_hist = fig.add_subplot(1, 2, i+1)
            width = (bin_edges[1] - bin_edges[0])
            plt_hist.bar(bin_edges[1:], Q_hist, width=width)
            plt_hist.set_title("f=%0.2f" %f, fontsize=20)
            plt_hist.set_xlim(-xlims[i], xlims[i])
            plt_hist.set_ylim(0., ylims[i])
            plt_hist.set_xlabel("Q Statistic", fontsize=18)
            plt_hist.set_ylabel("Count", fontsize=18)
            plt_hist.axvline(0, ymin=0., ymax=plt_hist.get_ylim()[1], color='red')
            plt_hist.axvline(Q_f, ymin=0., ymax=plt_hist.get_ylim()[1], color='limegreen')
            plt_hist.text(text_x[i], text_y[i], "Q=%0.3f" % Q_f, fontsize=18, color='limegreen')

    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/10000p_Qhist_" + str(t) + "s")
    plt.close()
    Q.append(Q_t)

Q = np.array(Q)

fig = plt.figure(figsize=(12, 9))
plt.box(False)
ax = plt.subplot(111)
l4 = ax.plot(timestamps, Q[:, 3], '-*', color='orange', linewidth=2, markersize=3, label='f=0.001')
l1 = ax.plot(timestamps, Q[:, 0], '-bo', linewidth=2, markersize=3, label='f=0.01')
l2 = ax.plot(timestamps, Q[:, 1], '-gs', linewidth=2, markersize=3, label='f=0.1')
l3 = ax.plot(timestamps, Q[:, 2], '-cv', linewidth=2, markersize=3, label='f=0.5')
plt.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], 'k')
ax.set_title("Q statistic over time for differing f-values", fontsize=25)
ax.set_xlabel("Time", fontsize=25)
ax.set_ylabel("Q", fontsize=25)
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax.legend(fontsize=25)
# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/10000p_Q_over_time")
print("Done")

