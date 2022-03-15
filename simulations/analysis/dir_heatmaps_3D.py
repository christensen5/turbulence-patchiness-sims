from physt import spherical, spherical_surface
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from tqdm import tqdm
import os

sims = ["/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot.nc",
        "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot.nc",
        "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot.nc"]

vmax = 372395.0677895795 #969968.901287883  # pre-computed from sims

for sim in tqdm(sims):
    # savepath = os.path.join(os.path.dirname(sim), "dir_heatmap_3D_" + os.path.basename(sim).split("_")[5] + "_" + os.path.basename(sim).split("_")[6] + ".png")
    savepath = os.path.join("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/dir",
                            "dir_heatmap_3D_" + os.path.basename(sim).split("_")[5] + "_" + os.path.basename(sim).split("_")[6] + ".png")
    nc = netCDF4.Dataset(sim)
    dir_x = nc.variables["dir_x"][:]
    dir_y = nc.variables["dir_y"][:]
    dir_z = nc.variables["dir_z"][:]
    nc.close()

    data = np.empty((100000*41, 3))
    data[:, 0] = dir_x[:, np.arange(200, 601, 10)].flatten()
    data[:, 1] = dir_y[:, np.arange(200, 601, 10)].flatten()
    data[:, 2] = dir_z[:, np.arange(200, 601, 10)].flatten()

    # h = spherical(data, radial_bins=1)
    h = spherical_surface(data)

    globe=h.projection("theta", "phi")
    ax = globe.plot.globe_map(density=True, figsize=(7,7), cmap="Reds", cmap_max=vmax)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([-1., 0., 1.])
    ax.set_yticks([-1., 0., 1.])
    ax.set_zticks([-1., 0., 1.])
    plt.savefig(savepath)
    plt.clf()

POL = h.get_bin_edges()[0]
AZI = h.get_bin_edges()[1]
fig = plt.figure(figsize=(7, 7))
ax_tmp = fig.add_subplot(1, 1, 1)
cbar_tmp = ax_tmp.pcolormesh(POL, AZI, h.densities, cmap='Reds', vmax=vmax)
fig.subplots_adjust(top=0.9, left=0.3)
cbar_ax = fig.add_axes([0.05, 0.1, 0.1, 0.8])
fig.colorbar(cbar_tmp, cax=cbar_ax)
cbar_ax.yaxis.set_label_position('left')
cbar_ax.yaxis.set_ticks_position('left')
cbar_ax.tick_params(axis='y', which='major', labelsize=18)
fig.show()