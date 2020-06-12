import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from simulations.util.util import cart2spher
from tqdm import tqdm
import os, sys

# ======================================================================================================================
# LOAD DATA

filepath_v10_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot.nc"
filepath_v10_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot.nc"
filepath_v10_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot.nc"
filepath_v100_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot.nc"
filepath_v100_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot.nc"
filepath_v100_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot.nc"
filepath_v500_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot.nc"
filepath_v500_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot.nc"
filepath_v500_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot.nc"

data_v10_B1 = {'nc': filepath_v10_B1,
               'V': 10, 'B': 1.0}
data_v10_B3 = {'nc': filepath_v10_B3,
               'V': 10, 'B': 3.0}
data_v10_B5 = {'nc': filepath_v10_B5,
               'V': 10, 'B': 5.0}
data_v100_B1 = {'nc': filepath_v100_B1,
                'V': 100, 'B': 1.0}
data_v100_B3 = {'nc': filepath_v100_B3,
                'V': 100, 'B': 3.0}
data_v100_B5 = {'nc': filepath_v100_B5,
                'V': 100, 'B': 5.0}
data_v500_B1 = {'nc': filepath_v500_B1,
                'V': 500, 'B': 1.0}
data_v500_B3 = {'nc': filepath_v500_B3,
                'V': 500, 'B': 3.0}
data_v500_B5 = {'nc': filepath_v500_B5,
                'V': 500, 'B': 5.0}

all_sims = [data_v10_B1, data_v100_B1, data_v500_B1,
            data_v10_B3, data_v100_B3, data_v500_B3,
            data_v10_B5, data_v100_B5, data_v500_B5]

# ======================================================================================================================
# Extract dir for each motile simulation and plot heatmap/histogram.
heatmap = True
heatstring = "hmap" if heatmap else "hist"
timestamps = np.arange(200, 600)#np.linspace(0, 600, 6, dtype="int")
fig = plt.figure(figsize=(15, 9))

sim_id = 0
for sim in tqdm(all_sims):
    sim_id += 1
    nc = netCDF4.Dataset(sim["nc"])
    deps = nc.variables["z"][:][:, timestamps]
    dir_x = nc.variables["dir_x"][:][:, timestamps]
    dir_y = nc.variables["dir_y"][:][:, timestamps]
    dir_z = nc.variables["dir_z"][:][:, timestamps]
    nc.close()

    r, pol, azi = cart2spher(dir_x, dir_y, dir_z)
    assert(np.allclose(r, np.ones_like(r)))  # ensure all particle directions were indeed normalised in the kernel.  # mean particle polar angle at each timestamp.
    # remove surface particles
    nosurf_indicies = deps < 360
    azi = azi[nosurf_indicies]
    pol = pol[nosurf_indicies]
    pol_avg = np.median(pol)
    pol = np.multiply(pol, np.sign(azi))  # polar angles with negative azimuthal angle are now negative, otherwise positive.

    n_bins = 73
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    if heatmap:
        h, pol_bins = np.histogram(pol, bins, normed=True)
        h = h.reshape((1, h.size))
        pol_bins = pol_bins.reshape((1, pol_bins.size)).repeat(2, axis=0)
        rad_bins = np.vstack((np.zeros(n_bins + 1), np.ones(n_bins + 1)))
        ax = plt.subplot(3, 3, sim_id, projection='polar')
        c = ax.pcolormesh(pol_bins, rad_bins, h, vmax=0.3, cmap='coolwarm')
        ax.axvline(pol_avg, color='k', lw=1.)
        ax.axvline(-pol_avg, color='k', lw=1.)
        ax.annotate("{:.1f}".format(np.rad2deg(pol_avg)), xy=(pol_avg, ax.get_ylim()[1]), color='k')
        ax.set_xticklabels(['0', '45', '90', '135', '180', '-135', '', '-45'])
        ax.set_yticks([])
        if sim_id < 4:
            ax.set_title("v = {:n}um".format(sim["V"]), pad=20,  fontsize=15)
        if sim_id % 3 == 1:
            ax.set_ylabel("B = {:1.1f}".format(sim["B"]), labelpad=25, fontsize=15)
        # if sim_id == 3:
        fig.colorbar(c, ax=ax)
        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(180)

        if sim_id == 9:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(c, cax=cbar_ax)

    else:
        h, pol_bins = np.histogram(pol, bins, normed=True)
        width = 2 * np.pi / n_bins
        ax = plt.subplot(3, 3, sim_id, projection='polar')
        bar = ax.bar(bins[:n_bins], h, width=width, bottom=0.0)
        ax.axvline(pol_avg, color='r')
        ax.axvline(-pol_avg, color='r')
        ax.annotate("{:.1f}".format(np.rad2deg(pol_avg)), xy=(pol_avg, ax.get_ylim()[1] - .05), color='red')
        ax.set_xticklabels(['0', '45', '90', '135', '180', '-135', '-90', '-45'])
        ax.set_yticks([0., 0.1, 0.2, 0.3])
        ax.set_ylim(0, .3)
        if sim_id < 4:
            ax.set_title("v = {:n}um".format(sim["V"]), pad=20,  fontsize=15)
        if sim_id % 3 == 1:
            ax.set_ylabel("B = {:1.1f}".format(sim["B"]), labelpad=25, fontsize=15)
        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(180)

# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/dir/100000p_dir_%s_20-60s_nosurf.png" % (heatstring))
            # bbox_inches='tight')


