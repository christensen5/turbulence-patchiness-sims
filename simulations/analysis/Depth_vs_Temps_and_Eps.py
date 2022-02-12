import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

# LOAD DATA
epsilon_csv_file = "/media/alexander/AKC Passport 2TB/epsilon.csv"

epsilon_array = np.genfromtxt(epsilon_csv_file, delimiter=",")
time_offset = 30.0  # epsilon csv file includes DNS spin-up timesteps, which the Parcels simulation does not.
print("Time offset set to %.1f seconds. Ensure this is correct." % time_offset)
time_offset_index = np.searchsorted(epsilon_array[:, 0], time_offset)
epsilon_array = epsilon_array[time_offset_index:, :]  # remove rows corresponding to spin-up timesteps
epsilon_array[:, 0] -= time_offset  # align timestamps with the Parcels simulation.
epsilon_array[:, 3] = -epsilon_array[:, 3]  # make epsilon column positive
epsilon_timestamps = np.unique(epsilon_array[:, 0])  # extract timestamps at which we have epsilon data

tempsWithDeps = np.load("/media/alexander/AKC Passport 2TB/0-30/twd.npy")
temps = tempsWithDeps[:, 0]
deps = tempsWithDeps[:, 1] * (0.3/tempsWithDeps[:, 1].max())  #offset depth by 6-cell buffer from DNS.
heatmap, xedges, yedges = np.histogram2d(temps, deps, bins=[np.linspace(-0.2, 0.8, 301), np.linspace(0, 0.31, 311)])

plt.clf()
fig = plt.figure(figsize=(18, 12))
# plot temps vs depths
xmin, xmax = [-0.3, 0.9]
ax_temp = plt.subplot(121)
xidx, yidx = np.where(heatmap>0)
ax_plot = ax_temp.scatter(xedges[xidx], yedges[yidx], s=8, edgecolors="none", alpha=0.7)
ax_temp.annotate("a", xy=(0, 1.04), color='k', xycoords='axes fraction', ha='right', va='top', fontweight='bold')
ax_temp.hlines(y=[0.1, 0.17, 0.24, 0.3], xmin=xmin, xmax=1.5, colors="r", linestyles=":", lw=2)
# axis parameters and labels etc
ax_temp.set_xlim(xmin, xmax)
ax_temp.set_ylim(0, 0.31)
ax_temp.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
ax_temp.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax_temp.set_xlabel(r"Temperature (relative to reference temp $\theta_{z_0}$) [$^{\circ}C$]")
ax_temp.set_ylabel(r"z [m]")
ax_temp.spines["top"].set_visible(False)
ax_temp.spines["right"].set_visible(False)

# plot eps vs depths
xmin, xmax = [-3e-5, 3.0e-4]
ax_eps = plt.subplot(122)
eps_plot = ax_eps.scatter(epsilon_array[:, 3], epsilon_array[:, 1], s=8, edgecolors="none", alpha=0.7)
ax_eps.annotate("b", xy=(0, 1.04), color='k', xycoords='axes fraction', ha='left', va='top', fontweight='bold')
#annotate depth regions
# ax_eps.annotate('Shallow', (0.000365, 0.27), xycoords='data', ha='right', va='center', color='red')
# ax_eps.annotate('Mid', (0.000365, 0.205), xycoords='data', ha='right', va='center', color='red')
# ax_eps.annotate('Deep', (0.000365, 0.135), xycoords='data', ha='right', va='center', color='red')
ax_eps.hlines(y=[0.1, 0.17, 0.24, 0.3], xmin=xmin, xmax=xmax, colors="r", linestyles=":", lw=2)
# axis parameters and labels etc
ax_eps.set_xlim(xmin, xmax)
ax_eps.set_ylim(0, 0.31)
ax_eps.set_xticks([0e-4, 1e-4, 2e-4, 3e-4])
ax_eps.set_xticklabels([r"0", r"$1\times 10^{-4}$", r"$2\times 10^{-4}$", r"$3\times 10^{-4}$"])
ax_eps.set_yticks([])#([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax_eps.set_xlabel(r"$\epsilon$ [$m^2 s^{-3}$]")
ax_eps.set_ylabel("")#(r"z [m]")
ax_eps.spines["top"].set_visible(False)
ax_eps.spines["left"].set_visible(False)
# label depth regions with z-values at boundaries
ax_right = ax_eps.twinx()
ax_right.tick_params(axis='y', colors="red")
ax_right.set(ylim=ax_eps.get_ylim())
ax_right.set_yticks([0.10, 0.17, 0.24, 0.30])
ax_right.spines["top"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.spines["bottom"].set_visible(False)

# annotate depth regions
plt.annotate('Shallow', (0.51, 0.83), xycoords='figure fraction', ha='center', va='center', color='red')
plt.annotate('Mid', (0.51, 0.65), xycoords='figure fraction', ha='center', va='center', color='red')
plt.annotate('Deep', (0.51, 0.46), xycoords='figure fraction', ha='center', va='center', color='red')

fig.tight_layout()
plt.subplots_adjust(wspace=0.1)
# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/DepthsvsTempsandEps/Depth_vs_Temps_and_Eps.png")