"""
This script analyses the relationship between particle depth and effective velocity (swim velocity + fluid velocity)
in each of our different microbe simulations.
"""
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from simulations.util.util import cart2spher
from tqdm import tqdm
import os, sys
import math

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

data_v10_B1 = {'nc': filepath_v10_B1, 'V': 10, 'B': 1.0}
data_v10_B3 = {'nc': filepath_v10_B3, 'V': 10, 'B': 3.0}
data_v10_B5 = {'nc': filepath_v10_B5, 'V': 10, 'B': 5.0}
data_v100_B1 = {'nc': filepath_v100_B1, 'V': 100, 'B': 1.0}
data_v100_B3 = {'nc': filepath_v100_B3, 'V': 100, 'B': 3.0}
data_v100_B5 = {'nc': filepath_v100_B5, 'V': 100, 'B': 5.0}
data_v500_B1 = {'nc': filepath_v500_B1, 'V': 500, 'B': 1.0}
data_v500_B3 = {'nc': filepath_v500_B3, 'V': 500, 'B': 3.0}
data_v500_B5 = {'nc': filepath_v500_B5, 'V': 500, 'B': 5.0}

all_sims = [data_v10_B1, data_v100_B1, data_v500_B1,
            data_v10_B3, data_v100_B3, data_v500_B3,
            data_v10_B5, data_v100_B5, data_v500_B5]

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

# define depth region (in cells)
depth_slice = False #[120, 200]#[[100, 170], [170, 240], [240, 305]]

fig = plt.figure(figsize=(15, 9))

# Extract particle dir and particlewise fluid velocity for each motile simulation and compute Veff.
timestamps = np.arange(0, 601, 10)
timestamps_to_plot = np.arange(20, 61)
cells_to_m = 1./1200  # conversion factor from cells/s to m/s
print("Conversion factor set to %f - ensure this is correct." % cells_to_m)
sim_id = 0
for sim in tqdm(all_sims):
    sim_id += 1
    nc = netCDF4.Dataset(sim["nc"])
    deps = nc.variables["z"][:][:, timestamps]
    dir_x = nc.variables["dir_x"][:][:, timestamps]
    dir_y = nc.variables["dir_y"][:][:, timestamps]
    dir_z = nc.variables["dir_z"][:][:, timestamps]
    vswim = nc.variables["v_swim"][:][:, timestamps]  # in cells/s
    nc.close()
    u = np.load(os.path.join(os.path.dirname(sim["nc"]), "u_pwise.npy"))  # in cells/s
    v = np.load(os.path.join(os.path.dirname(sim["nc"]), "v_pwise.npy"))
    w = np.load(os.path.join(os.path.dirname(sim["nc"]), "w_pwise.npy"))
    Veff_u = np.multiply(dir_x, vswim) + u  # in cells/s
    Veff_v = np.multiply(dir_y, vswim) + v
    Veff_w = np.multiply(dir_z, vswim) + w

    # convert to m/s
    Veff_u *= cells_to_m
    Veff_v *= cells_to_m
    Veff_w *= cells_to_m

    # plot V_eff distribution over 20-60s of simulation time.
    Veff_u_t = Veff_u[:, timestamps_to_plot].flatten()
    Veff_v_t = Veff_v[:, timestamps_to_plot].flatten()
    Veff_w_t = Veff_w[:, timestamps_to_plot].flatten()
    deps_t = deps[:, timestamps_to_plot].flatten()

    if nosurf:
        # remove surface particles
        Veff_u_t = Veff_u_t[deps_t < 360]
        Veff_v_t = Veff_v_t[deps_t < 360]
        Veff_w_t = Veff_w_t[deps_t < 360]
        deps_t = deps_t[deps_t < 360]

    if depth_slice:
        # keep only particles in current region
        Veff_u_t = Veff_u_t[np.logical_and(depth_slice[0] < deps_t, deps_t < depth_slice[1])]
        Veff_v_t = Veff_v_t[np.logical_and(depth_slice[0] < deps_t, deps_t < depth_slice[1])]
        Veff_w_t = Veff_w_t[np.logical_and(depth_slice[0] < deps_t, deps_t < depth_slice[1])]


    r, pol, azi = cart2spher(Veff_u_t, Veff_v_t, Veff_w_t)

    r_avg = np.mean(r)
    pol_avg = np.median(pol)
    pol = np.multiply(pol, np.sign(azi))  # polar angles with negative azimuthal angle are now negative, otherwise positive.

    n_rbins = 10
    n_polbins = 33#73
    rmax = r.max()
    rbin_upper = (rmax // 1) + math.ceil(100 * (rmax % 1))/100  # next hundredth after (rmax rounded to 2 decimal places).
    rbins = np.arange(0, rmax, 0.01)#np.linspace(0, rbin_upper, n_rbins + 1)
    polbins = np.linspace(-np.pi, np.pi, n_polbins + 1)

    h, _, _ = np.histogram2d(pol, r, [polbins, rbins], normed=False)

    R, POL = np.meshgrid(rbins, polbins)

    ax = plt.subplot(3, 3, sim_id, projection='polar')
    c = ax.pcolormesh(POL, R, h, cmap='Reds', vmax=60000)
    ax.axvline(pol_avg, color='k', lw=1.)
    ax.axvline(-pol_avg, color='k', lw=1.)
    ax.annotate(r'{:.1f}$\degree$'.format(np.rad2deg(pol_avg)), xy=(0.45, 0.55), color='k', xycoords='axes fraction')
    ax.annotate(r'{:.1f} mm/s'.format(1000 * r_avg), xy=(0.37, 0.45), color='k', xycoords='axes fraction')
    if sim_id < 4:
        ax.set_title("v = {:n}um".format(sim["V"]), pad=20, fontsize=15)
    if sim_id % 3 == 1:
        ax.set_ylabel("B = {:1.1f}".format(sim["B"]), labelpad=25, fontsize=15)
    ax.set_xticklabels(['',  r'$45\degree$',  r'$90\degree$',  r'$135\degree$',  r'$180\degree$',  r'$-135\degree$', '', r'$-45\degree$'])
    ax.set_ylim([0, 0.06])
    ax.set_yticks([0, 0.06])
    ax.set_yticklabels(['0', '60'])
    ax.set_rorigin(-0.06)
    plt.colorbar(c, ax=ax)
    ax.set_theta_zero_location("N")
    ax.set_rlabel_position(0)
# plt.show()
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/Veff/100000p_Veff_20-60s%s.png" % (surfstring))