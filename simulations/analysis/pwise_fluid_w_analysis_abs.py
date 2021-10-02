import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4
from tqdm import tqdm

cells_to_m = 1./1200  # conversion factor from cells/s to m/s
timestamps = np.arange(0, 601, 10)  # every second to align with extracted pwise velocities in npy files

nosurf = True

# LOAD DATA
filepath_dead = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-60s_0.01dt_0.1sdt_initunif_dead.nc"
filepath_v10_B1 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot.nc"
filepath_v10_B3 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot.nc"
filepath_v10_B5 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot.nc"
filepath_v100_B1 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot.nc"
filepath_v100_B3 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot.nc"
filepath_v100_B5 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot.nc"
filepath_v500_B1 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot.nc"
filepath_v500_B3 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot.nc"
filepath_v500_B5 = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot.nc"

data_dead = {'nc': filepath_dead, 'V': 0, 'B': np.inf}
data_v10_B1 = {'nc': filepath_v10_B1, 'V': 10, 'B': 1.0}
data_v10_B3 = {'nc': filepath_v10_B3, 'V': 10, 'B': 3.0}
data_v10_B5 = {'nc': filepath_v10_B5, 'V': 10, 'B': 5.0}
data_v100_B1 = {'nc': filepath_v100_B1, 'V': 100, 'B': 1.0}
data_v100_B3 = {'nc': filepath_v100_B3, 'V': 100, 'B': 3.0}
data_v100_B5 = {'nc': filepath_v100_B5, 'V': 100, 'B': 5.0}
data_v500_B1 = {'nc': filepath_v500_B1, 'V': 500, 'B': 1.0}
data_v500_B3 = {'nc': filepath_v500_B3, 'V': 500, 'B': 3.0}
data_v500_B5 = {'nc': filepath_v500_B5, 'V': 500, 'B': 5.0}

# all_sims = [data_dead,
#             data_v10_B1, data_v100_B1, data_v500_B1,
#             data_v10_B3, data_v100_B3, data_v500_B3,
#             data_v10_B5, data_v100_B5, data_v500_B5]

all_sims = [data_v500_B5, data_v10_B3, data_dead, data_v10_B5, data_v10_B1, data_v100_B5,
            data_v100_B3, data_v500_B3, data_v100_B1, data_v500_B1]

fig = plt.figure(figsize=(18, 12))
# fig, (ax_means, ax_cdf) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5]}, figsize=(18, 12))
ax_cdf = fig.add_subplot(1, 1, 1)
left, bottom, width, height = [0.5, 0.3, 0.4, 0.1]
ax_means = fig.add_axes([left, bottom, width, height])
# discrete spectral colourmap
# cmap = plt.cm.jet  # define the colormap
# cmaplist = [cmap(i) for i in range(cmap.N)]  # extract desired colours (0-21 total)
# cmaplist[0] = (.5, .5, .5, 1.0)  # force the first color entry (for non-motile microbes) to be grey
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Discrete spectral cmap (with grey 0)', cmaplist, cmap.N)
colormap_discrete=plt.cm.RdBu(np.linspace(0, 1, 21))
colours = colormap_discrete[[0, 1, 10, 2, 3, 4, 16, 17, 18, 19, 20], :]
colours[2] = [1., 0., 1., 1.]  # non-motile microbes in pink
# define the bins and normalize
# bounds = np.linspace(0, 1, 10)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

print("Conversion factor set to %f - ensure this is correct." % cells_to_m)
sim_id = 0
i = 0
for sim in tqdm(all_sims):
    sim_id += 1
    nc = netCDF4.Dataset(sim["nc"])
    deps = nc.variables["z"][:][:, timestamps]
    # load fluid velocities and convert to mm/s
    fluid_u = (1000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "u_pwise.npy"))
    fluid_v = (1000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "v_pwise.npy"))
    sim["fluid_w"] = (1000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "w_pwise.npy"))
    sim["fluid_mag"] = np.power(np.power(fluid_u, 2) + np.power(fluid_v, 2) + np.power(sim["fluid_w"], 2), 0.5)
    assert deps.shape == fluid_u.shape   # check arrays are right (i.e. correct timesteps and all particles chosen)
    nc.close()

    if nosurf:
        # remove surface particles
        inds = np.logical_and(deps > 100, deps < 170)
        sim["fluid_w"] = sim["fluid_w"][inds]
        sim["fluid_mag"] = sim["fluid_mag"][inds]
        deps = deps[inds]


    sim["hist_fluid_w"], sim["bin_edges"] = np.histogram(np.abs(sim["fluid_w"]), bins=100, density=True)

    if i == 2:
        ax_cdf.plot((sim["bin_edges"][1:] + sim["bin_edges"][:-1]) / 2, np.cumsum(sim["hist_fluid_w"])/sim["hist_fluid_w"].sum(), c=colours[i, :],
                label="non-motile")
    else:
        ax_cdf.plot((sim["bin_edges"][1:] + sim["bin_edges"][:-1]) / 2, np.cumsum(sim["hist_fluid_w"])/sim["hist_fluid_w"].sum(), c=colours[i, :], label="B=%d, V=%d" % (sim["B"], sim["V"]))
    ax_means.scatter(np.abs(sim["fluid_w"]).mean(), 0, color=colours[i, :])

    i += 1

leg = ax_cdf.legend(loc="upper right", bbox_to_anchor=[1, 0.9], fontsize=22)
for l in leg.get_lines():
    l.set_linewidth(3)
ax_cdf.set_xlabel(r"$|u_z|$ at microbe locations ($mm/s$)", fontsize=22)
ax_cdf.set_ylabel("CDF", fontsize=22)
ax_cdf.tick_params(axis='x', which='major', labelsize=22)
ax_cdf.tick_params(axis='y', which='major', labelsize=22)
ax_cdf.spines['right'].set_visible(False)
ax_cdf.spines['top'].set_visible(False)

ax_means.set_xlabel(r"$mean(|u_z|)$ at microbe locations ($mm/s$)", fontsize=22)
# ax_means.xaxis.set_label_position('top')
# ax_means.xaxis.tick_top()
ax_means.tick_params(axis='x', which='major', labelsize=22)
ax_means.set_yticks([])
# ax_means.spines['left'].set_visible(False)
# ax_means.spines['right'].set_visible(False)
# ax_means.spines['bottom'].set_visible(False)

# fig.tight_layout()
plt.savefig("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/pwise_fluid_velocity_deep_absvert.png")