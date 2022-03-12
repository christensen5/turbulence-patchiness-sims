import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import os
import netCDF4
from tqdm import tqdm

cells_to_m = 1./1200  # conversion factor from cells/s to m/s
timestamps = np.arange(0, 601, 10)  # every second to align with extracted pwise velocities in npy files

nosurf = True

## LOAD DATA
mount_point = "/media/alexander/" # laptop
# mount_point = "/home/alexander/Documents/" # desktop

filepath_dead = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-60s_0.01dt_0.1sdt_initunif_dead.nc")
filepath_v10_B1 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot.nc")
filepath_v10_B3 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot.nc")
filepath_v10_B5 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot.nc")
filepath_v100_B1 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot.nc")
filepath_v100_B3 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot.nc")
filepath_v100_B5 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot.nc")
filepath_v500_B1 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot.nc")
filepath_v500_B3 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot.nc")
filepath_v500_B5 = os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot.nc")

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

all_sims = [data_v10_B1, data_v10_B3, data_v10_B5,
            data_v100_B1, data_v100_B3, data_v100_B5,
            data_v500_B1, data_v500_B3, data_v500_B5]

# all_sims = [data_v500_B5, data_v10_B3, data_dead, data_v10_B5, data_v10_B1, data_v100_B5,
#             data_v100_B3, data_v500_B3, data_v100_B1, data_v500_B1]

fig = plt.figure(figsize=(18, 12))
ax_cdf = fig.add_subplot(1, 1, 1)
left, bottom, width, height = [0.5, 0.3, 0.4, 0.1]
# ax_means = fig.add_axes([left, bottom, width, height])
colormap_discrete=plt.cm.RdBu(np.linspace(0, 1, 21))
colours = colormap_discrete[[20, 19, 18, 17, 16, 4, 3, 2, 1, 0], :]#[[0, 1, 10, 2, 3, 4, 16, 17, 18, 19, 20], :]
# colours[10] = [1., 0., 1., 1.] #[2] = [1., 0., 1., 1.]  # non-motile microbes in pink

print("Conversion factor set to %f - ensure this is correct." % cells_to_m)
sim_id = 0
i = 0
violin_labels = []
boxplot_data = []
for sim in tqdm(all_sims):
    sim_id += 1
    nc = netCDF4.Dataset(sim["nc"])
    deps = nc.variables["z"][:][:, timestamps]
    # load fluid velocities and convert to um/s
    fluid_u = (1000000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "u_pwise.npy"))
    fluid_v = (1000000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "v_pwise.npy"))
    fluid_w = (1000000 * cells_to_m) * np.load(os.path.join(os.path.dirname(sim['nc']), "w_pwise.npy"))
    sim["fluid_uv_mag"] = np.power(np.power(fluid_u, 2) + np.power(fluid_v, 2), 0.5)
    # sim["fluid_mag"] = np.power(np.power(fluid_u, 2) + np.power(fluid_v, 2) + np.power(fluid_w, 2), 0.5)
    assert deps.shape == fluid_u.shape   # check arrays are right (i.e. correct timesteps and all particles chosen)
    nc.close()

    if nosurf:
        # remove surface particles
        inds = np.logical_and(deps > 100, deps < 170)
        sim["fluid_uv_mag"] = sim["fluid_uv_mag"][inds]
        # sim["fluid_mag"] = sim["fluid_mag"][inds]
        deps = deps[inds]

    sim["ratio_vswim_uvmag"] = sim["V"] / sim["fluid_uv_mag"]

    sim["hist"], sim["bin_edges"] = np.histogram(sim["ratio_vswim_uvmag"], bins=100, density=True)

    boxplot_data.append(sim["ratio_vswim_uvmag"])
    # violins = ax_cdf.boxplot(sim["ratio_vswim_uvmag"], positions=[sim_id], vert=False, showmeans=True)
    # violin_labels.append((matplotlib.patches.Patch(color=colours[i, 0:3]), "B=%d, V=%d" % (sim["B"], sim["V"])))
    # for pc in violins['bodies']:
    #     pc.set_facecolor(colours[i, 0:3])
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(1)
    # ax_cdf.plot((sim["bin_edges"][1:] + sim["bin_edges"][:-1]) / 2, np.cumsum(sim["hist"])/sim["hist"].sum(), c=colours[i, :], label="B=%d, V=%d" % (sim["B"], sim["V"]))
    # ax_means.scatter(np.abs(sim["fluid_w"]).mean(), 0, color=colours[i, :])

    i += 1

box = ax_cdf.boxplot(boxplot_data, vert=False)
ax_cdf.axvline(x=1, ls="--", lw=2, c="k")

ax_cdf.set_xlabel(r"$\frac{v_{\mathrm{swim}}}{|(v_{\mathrm{fluid},x}, v_{\mathrm{fluid},y})|}$ at microbe locations", fontsize=22)
ax_cdf.tick_params(axis='x', which='major', labelsize=22)
ax_cdf.tick_params(axis='y', which='major', labelsize=22)
ax_cdf.set_xlim([0, 3])
ax_cdf.set_yticklabels([r"B=%d$s$, V=%d$\mu m/s$" % (sim["B"], sim["V"]) for sim in all_sims])
ax_cdf.spines['right'].set_visible(False)
ax_cdf.spines['top'].set_visible(False)

plt.subplots_adjust(left=0.2)

# ax_means.set_xlabel(r"$mean(|(u_x, u_y)|)$ at microbe locations (" + r"$\mu m/s$)", fontsize=22)
# ax_means.tick_params(axis='x', which='major', labelsize=22)
# ax_means.set_yticks([])

# fig.tight_layout()
plt.savefig(os.path.join(mount_point, "DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/pwise_horizontal_fluid_velocity_deep.png"))