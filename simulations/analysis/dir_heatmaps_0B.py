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

filepath_v10_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_0B_10um_initunif_mot.nc"
filepath_v100_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_0B_100um_initunif_mot.nc"
filepath_v500_B0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_0B_500um_initunif_mot.nc"

data_v10_B0 = {'nc': filepath_v10_B0,
               'V': 10, 'B': 0}
data_v100_B0 = {'nc': filepath_v100_B0,
                'V': 100, 'B': 0}
data_v500_B0 = {'nc': filepath_v500_B0,
                'V': 500, 'B': 0}

all_sims = [data_v10_B0, data_v100_B0, data_v500_B0]

# choose plots
plot_1x3_Vvar = True

# ======================================================================================================================

if plot_1x3_Vvar:
    # Extract dir for each motile simulation with v=100um and plot heatmap/histogram. We ignore the other two swim
    # speeds and focus on varying B-value since vswim had no discernible effect on swim direction.
    heatmap = True
    heatstring = "hmap" if heatmap else "hist"
    timestamps = np.arange(200, 600)  # np.linspace(0, 600, 6, dtype="int")
    fig = plt.figure(figsize=(11, 5))

    sim_id = 0
    for sim in tqdm([data_v10_B0, data_v100_B0, data_v500_B0]):
        sim_id += 1
        nc = netCDF4.Dataset(sim["nc"])
        deps = nc.variables["z"][:][:, timestamps]
        dir_x = nc.variables["dir_x"][:][:, timestamps]
        dir_y = nc.variables["dir_y"][:][:, timestamps]
        dir_z = nc.variables["dir_z"][:][:, timestamps]
        nc.close()

        r, pol, azi = cart2spher(dir_x, dir_y, dir_z)
        assert (np.allclose(r, np.ones_like(
            r)))  # ensure all particle directions were indeed normalised in the kernel.  # mean particle polar angle at each timestamp.
        # remove surface particles
        nosurf_indicies = deps < 360
        azi = azi[nosurf_indicies]
        pol = pol[nosurf_indicies]
        pol_avg = np.median(pol)
        pol = np.multiply(pol, np.sign(
            azi))  # polar angles with negative azimuthal angle are now negative, otherwise positive.

        n_bins = 73
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        if heatmap:
            h, pol_bins = np.histogram(pol, bins, normed=True)
            h = h.reshape((1, h.size))
            pol_bins = pol_bins.reshape((1, pol_bins.size)).repeat(2, axis=0)
            rad_bins = np.vstack((np.zeros(n_bins + 1), np.ones(n_bins + 1)))
            ax = plt.subplot(1, 3, sim_id, projection='polar')
            c = ax.pcolormesh(pol_bins, rad_bins, h, vmax=0.3, cmap='Reds')
            ax.axvline(pol_avg, color='k', lw=1.5)
            ax.axvline(-pol_avg, color='k', lw=1.5)
            ax.annotate("{:.1f}".format(np.rad2deg(pol_avg)), xy=(pol_avg-np.deg2rad(1), 0.95*ax.get_ylim()[1]), color='k', fontsize=15)
            ax.set_xticklabels(['0', '45', '90', '135', '180', '-135', '', '-45'])
            ax.set_yticks([])
            ax.set_title("No reorientation\n"+"$v = {:n}\mu m \ s^{{-1}}$".format(sim["V"]), pad=20, fontsize=15)
            ax.set_theta_zero_location("N")
            ax.set_rlabel_position(180)

            if sim_id == 3:
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.20, 0.02, 0.60])
                fig.colorbar(c, cax=cbar_ax)

        else:
            NotImplementedError("Only heatmap plot is implemented for the 1x3 B-varying plot.")

    # plt.show()
    fig.savefig(
        "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/dir/100000p_dir_%s_20-60s_nosurf_0B_Vvar.png" % (
            heatstring))
    # bbox_inches='tight')



