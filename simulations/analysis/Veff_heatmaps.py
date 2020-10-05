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

# full- or half-polar plots
half_polar = True
halfstring = "_half" if half_polar is True else ""

# define depth regions (in cells)
depth_slices = [[100, 170], [170, 240], [240, 305], [170, 305]]  # deep, mid, shallow, shallow-mid

fig = plt.figure(figsize=(12, 16))
# Extract particle dir and particlewise fluid velocity for each motile simulation and compute Veff.
timestamps = np.arange(0, 601, 10)
timestamps_to_plot = np.arange(20, 61)
cells_to_m = 1. / 1200  # conversion factor from cells/s to m/s
print("Conversion factor set to %f - ensure this is correct." % cells_to_m)
sim_id = 0
i = 0
subfiglabels = "adbecf"
for sim in tqdm([data_v10_B5, data_v500_B1]):
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

    # extract V_eff distribution over 20-60s of simulation time.
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

    Veff_u_t_shallowmid = Veff_u_t[np.logical_and(depth_slices[3][0] < deps_t, deps_t < depth_slices[3][1])]
    Veff_v_t_shallowmid = Veff_v_t[np.logical_and(depth_slices[3][0] < deps_t, deps_t < depth_slices[3][1])]
    Veff_w_t_shallowmid = Veff_w_t[np.logical_and(depth_slices[3][0] < deps_t, deps_t < depth_slices[3][1])]

    Veff_u_t_shallow = Veff_u_t[np.logical_and(depth_slices[2][0] < deps_t, deps_t < depth_slices[2][1])]
    Veff_v_t_shallow = Veff_v_t[np.logical_and(depth_slices[2][0] < deps_t, deps_t < depth_slices[2][1])]
    Veff_w_t_shallow = Veff_w_t[np.logical_and(depth_slices[2][0] < deps_t, deps_t < depth_slices[2][1])]

    Veff_u_t_mid = Veff_u_t[np.logical_and(depth_slices[1][0] < deps_t, deps_t < depth_slices[1][1])]
    Veff_v_t_mid = Veff_v_t[np.logical_and(depth_slices[1][0] < deps_t, deps_t < depth_slices[1][1])]
    Veff_w_t_mid = Veff_w_t[np.logical_and(depth_slices[1][0] < deps_t, deps_t < depth_slices[1][1])]
    
    Veff_u_t_deep = Veff_u_t[np.logical_and(depth_slices[0][0] < deps_t, deps_t < depth_slices[0][1])]
    Veff_v_t_deep = Veff_v_t[np.logical_and(depth_slices[0][0] < deps_t, deps_t < depth_slices[0][1])]
    Veff_w_t_deep = Veff_w_t[np.logical_and(depth_slices[0][0] < deps_t, deps_t < depth_slices[0][1])]

    # convert to spherical coords
    r_shallowmid, pol_shallowmid, azi_shallowmid = cart2spher(Veff_u_t_shallowmid, Veff_v_t_shallowmid, Veff_w_t_shallowmid)
    r_shallow, pol_shallow, azi_shallow = cart2spher(Veff_u_t_shallow, Veff_v_t_shallow, Veff_w_t_shallow)
    r_mid, pol_mid, azi_mid = cart2spher(Veff_u_t_mid, Veff_v_t_mid, Veff_w_t_mid)
    r_deep, pol_deep, azi_deep = cart2spher(Veff_u_t_deep, Veff_v_t_deep, Veff_w_t_deep)

    # compute means
    r_avg_shallowmid = np.mean(r_shallowmid)
    r_avg_shallow = np.mean(r_shallow)
    r_avg_mid = np.mean(r_mid)
    r_avg_deep = np.mean(r_deep)
    pol_avg_shallowmid = np.median(pol_shallowmid)
    pol_avg_shallow = np.median(pol_shallow)
    pol_avg_mid = np.median(pol_mid)
    pol_avg_deep = np.median(pol_deep)
    if not half_polar:
        pol_shallowmid = np.multiply(pol_shallowmid, np.sign(azi_shallowmid))  # polar angles with negative azimuthal angle are now negative, otherwise positive.
        pol_shallow = np.multiply(pol_shallow, np.sign(azi_shallow))  # polar angles with negative azimuthal angle are now negative, otherwise positive.
        pol_mid = np.multiply(pol_mid, np.sign(azi_mid))  # polar angles with negative azimuthal angle are now negative, otherwise positive.
        pol_deep = np.multiply(pol_deep, np.sign(azi_deep))  # polar angles with negative azimuthal angle are now negative, otherwise positive.
    
    # set up histogram parameters
    n_rbins = 10
    n_polbins = 33#73
    rmax = max(r_shallowmid.max(), r_shallow.max(), r_mid.max(), r_deep.max())
    # rbin_upper = (rmax // 1) + math.ceil(100 * (rmax % 1))/100  # next hundredth after (rmax rounded to 2 decimal places).
    rbins = np.arange(0, 0.06, 0.01)#np.arange(0, rmax, 0.01)#np.linspace(0, rbin_upper, n_rbins + 1)
    polbins = np.linspace(-np.pi, np.pi, n_polbins + 1)
    # generate histograms
    h_shallowmid, _, _ = np.histogram2d(pol_shallowmid, r_shallowmid, [polbins, rbins], normed=True)
    h_shallow, _, _ = np.histogram2d(pol_shallow, r_shallow, [polbins, rbins], normed=True)
    h_mid, _, _ = np.histogram2d(pol_mid, r_mid, [polbins, rbins], normed=True)
    h_deep, _, _ = np.histogram2d(pol_deep, r_deep, [polbins, rbins], normed=True)
    R, POL = np.meshgrid(rbins, polbins)

    # vmax = 9 if not half_polar else 16
    # ax_shallowmid = fig_shallowmid.add_subplot(1, 2, sim_id, projection='polar')
    # cbar_shallowmid = ax_shallowmid.pcolormesh(POL, R, h_shallowmid, cmap='Reds', vmax=vmax)
    # ax_shallowmid.axvline(pol_avg_shallowmid, color='k', lw=1.)
    # ax_shallowmid.axvline(-pol_avg_shallowmid, color='k', lw=1.)
    # if half_polar:
    #     ax_shallowmid.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_shallowmid)), xy=(0.61, 0.55), color='k', xycoords='axes fraction')
    #     ax_shallowmid.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000 * r_avg_shallowmid), xy=(0.53, 0.45), color='k', xycoords='axes fraction')
    # else:
    #     ax_shallowmid.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_shallowmid)),
    #                            xy=(0.41, 0.55), color='k', xycoords='axes fraction')
    #     ax_shallowmid.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000 * r_avg_shallowmid),
    #                            xy=(0.33, 0.45), color='k', xycoords='axes fraction')
    # ax_shallowmid.set_title(r"$B = {:1.1f}s^{{-1}}$" "\n" "$v = {:n}\mu m \ s^{{-1}}$".format(sim["B"], sim["V"]), pad=20, fontsize=15)
    # ax_shallowmid.set_xticklabels(['', r'$45\degree$', r'$90\degree$', r'$135\degree$', r'$180\degree$', r'$-135\degree$', '', r'$-45\degree$'])
    # ax_shallowmid.set_ylim([0, 60])
    # ax_shallowmid.set_yticks([0, 60])
    # ax_shallowmid.set_yticklabels([r'$0mm \ s^{-1}$', r'$60mm \ s^{-1}$'])
    # ax_shallowmid.set_rorigin(-60)
    # ax_shallowmid.set_theta_zero_location("N")
    # ax_shallowmid.set_rlabel_position(0)
    # if half_polar:
    #     ax_shallowmid.set_thetamin(0)
    #     ax_shallowmid.set_thetamax(180)

    vmax = 9 if not half_polar else 30
    ax_shallow = fig.add_subplot(3, 2, sim_id, projection='polar')
    cbar_shallow = ax_shallow.pcolormesh(POL, R, h_shallow, cmap='Reds', vmax=vmax)
    ax_shallow.axvline(pol_avg_shallow, color='k', lw=1.)
    ax_shallow.axvline(-pol_avg_shallow, color='k', lw=1.)
    # label subplot with boldface lowercase letter for NatComms
    ax_shallow.annotate("ab"[sim_id - 1], xy=(0, 1), color='k', xycoords='axes fraction',
                                horizontalalignment='left', verticalalignment='top', fontsize=20, fontweight='bold')
    # label depth region
    if sim_id == 2:
            ax_shallow.annotate('Shallow', xy=(1.3, 0.5), color='k', xycoords='axes fraction',
                                horizontalalignment='center', verticalalignment='center', rotation=270, fontsize=24)
    if half_polar:
        ax_shallow.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_shallow)),
                               xy=(0.61, 0.55), color='k', xycoords='axes fraction', fontsize=20)
        ax_shallow.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_shallow),
                               xy=(0.53, 0.4), color='k', xycoords='axes fraction', fontsize=20)
    else:
        ax_shallow.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_shallow)),
                               xy=(0.41, 0.55), color='k', xycoords='axes fraction')
        ax_shallow.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_shallow),
                               xy=(0.33, 0.45), color='k', xycoords='axes fraction')
    # ax_shallow.set_title(r"$B = {:1.1f}s^{{-1}}$" "\n" "$v = {:n}\mu m \ s^{{-1}}$".format(sim["B"], sim["V"]),
    #                         pad=20, fontsize=25)
    ax_shallow.set_title(["Non-Agile", "Agile"][sim_id - 1], pad=20, fontsize=28)
    ax_shallow.set_rorigin(-0.06)
    ax_shallow.set_theta_zero_location("N")
    ax_shallow.set_rlabel_position(0)
    if half_polar:
        ax_shallow.set_thetamin(0)
        ax_shallow.set_thetamax(180)
    ax_shallow.tick_params(axis='x', which='major', labelsize=18, pad=15)
    ax_shallow.tick_params(axis='y', which='major', labelsize=18)
    ax_shallow.set_xticks([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi])
    ax_shallow.set_xticklabels(['', r'$45\degree$', r'$90\degree$', r'$135\degree$', ''])
    ax_shallow.set_ylim([0, 0.06])
    ax_shallow.set_yticks([0, 0.06])
    ax_shallow.set_yticklabels([r'$0mm \ s^{-1}$', r'$60mm \ s^{-1}$'])

    ax_mid = fig.add_subplot(3, 2, 2 + sim_id, projection='polar')
    cbar_mid = ax_mid.pcolormesh(POL, R, h_mid, cmap='Reds', vmax=vmax)
    ax_mid.axvline(pol_avg_mid, color='k', lw=1.)
    ax_mid.axvline(-pol_avg_mid, color='k', lw=1.)
    # label subplot with boldface lowercase letter for NatComms
    ax_mid.annotate("cd"[sim_id - 1], xy=(0, 1), color='k', xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top', fontsize=20, fontweight='bold')
    # label depth region
    if sim_id == 2:
            ax_mid.annotate('Mid', xy=(1.3, 0.5), color='k', xycoords='axes fraction',
                                horizontalalignment='center', verticalalignment='center', rotation=270, fontsize=24)
    if half_polar:
        ax_mid.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_mid)),
                               xy=(0.61, 0.55), color='k', xycoords='axes fraction', fontsize=20)
        ax_mid.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_mid),
                               xy=(0.53, 0.4), color='k', xycoords='axes fraction', fontsize=20)
    else:
        ax_mid.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_mid)),
                               xy=(0.41, 0.55), color='k', xycoords='axes fraction')
        ax_mid.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_mid),
                               xy=(0.33, 0.45), color='k', xycoords='axes fraction')
    # ax_mid.set_title(r"$B = {:1.1f}s^{{-1}}$" "\n" "$v = {:n}\mu m \ s^{{-1}}$".format(sim["B"], sim["V"]),
    #                         pad=20, fontsize=25)
    ax_mid.set_rorigin(-0.06)
    ax_mid.set_theta_zero_location("N")
    ax_mid.set_rlabel_position(0)
    if half_polar:
        ax_mid.set_thetamin(0)
        ax_mid.set_thetamax(180)
    ax_mid.tick_params(axis='x', which='major', labelsize=18, pad=15)
    ax_mid.tick_params(axis='y', which='major', labelsize=18)
    ax_mid.set_xticks([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi])
    ax_mid.set_xticklabels(['', r'$45\degree$', r'$90\degree$', r'$135\degree$', ''])
    ax_mid.set_ylim([0, 0.06])
    ax_mid.set_yticks([0, 0.06])
    ax_mid.set_yticklabels([r'$0mm \ s^{-1}$', r'$60mm \ s^{-1}$'])

    ax_deep = fig.add_subplot(3, 2, 4 + sim_id, projection='polar')
    cbar_deep = ax_deep.pcolormesh(POL, R, h_deep, cmap='Reds', vmax=vmax)
    ax_deep.axvline(pol_avg_deep, color='k', lw=1.)
    ax_deep.axvline(-pol_avg_deep, color='k', lw=1.)
    # label subplot with boldface lowercase letter for NatComms
    ax_deep.annotate("ef"[sim_id - 1], xy=(0, 1), color='k', xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top', fontsize=20, fontweight='bold')
    # label depth region
    if sim_id == 2:
            ax_deep.annotate('Deep', xy=(1.3, 0.5), color='k', xycoords='axes fraction',
                                horizontalalignment='center', verticalalignment='center', rotation=270, fontsize=24)
    if half_polar:
        ax_deep.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_deep)),
                               xy=(0.61, 0.55), color='k', xycoords='axes fraction', fontsize=20)
        ax_deep.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_deep),
                               xy=(0.53, 0.4), color='k', xycoords='axes fraction', fontsize=20)
    else:
        ax_deep.annotate(r'$\overline{{\theta}}={:.1f}\degree$'.format(np.rad2deg(pol_avg_deep)),
                               xy=(0.41, 0.55), color='k', xycoords='axes fraction')
        ax_deep.annotate(r'$\overline{{V_{{eff}}}}={:.1f} mm \ s^{{-1}}$'.format(1000*r_avg_deep),
                               xy=(0.33, 0.45), color='k', xycoords='axes fraction')
    # ax_deep.set_title(r"$B = {:1.1f}s^{{-1}}$" "\n" "$v = {:n}\mu m \ s^{{-1}}$".format(sim["B"], sim["V"]),
    #                         pad=20, fontsize=25)
    ax_deep.set_rorigin(-0.06)
    ax_deep.set_theta_zero_location("N")
    ax_deep.set_rlabel_position(0)
    if half_polar:
        ax_deep.set_thetamin(0)
        ax_deep.set_thetamax(180)
    ax_deep.tick_params(axis='x', which='major', labelsize=18, pad=15)
    ax_deep.tick_params(axis='y', which='major', labelsize=18)
    ax_deep.set_xticks([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi])
    ax_deep.set_xticklabels(['', r'$45\degree$', r'$90\degree$', r'$135\degree$', ''])
    ax_deep.set_ylim([0, 0.06])
    ax_deep.set_yticks([0, 0.06])
    ax_deep.set_yticklabels([r'$0mm \ s^{-1}$', r'$60mm \ s^{-1}$'])

    i += 1

# fig_shallowmid.suptitle(r"$V_{eff}$ direction and magnitude in Shallow-Mid regions.", fontsize=20)
# fig_shallowmid.subplots_adjust(top=0.8, right=0.8)
# cbar_ax_shallowmid = fig_shallowmid.add_axes([0.85, 0.20, 0.02, 0.60])
# fig_shallowmid.colorbar(cbar_shallowmid, cax=cbar_ax_shallowmid)
#
fig.subplots_adjust(top=0.9, left=0.1)
cbar_ax = fig.add_axes([0.05, 0.1, 0.03, 0.8])
fig.colorbar(cbar_shallow, cax=cbar_ax)
cbar_ax.yaxis.set_label_position('left')
cbar_ax.yaxis.set_ticks_position('left')
cbar_ax.tick_params(axis='y', which='major', labelsize=18)


# plt.show()
fig.savefig('/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/Veff/3x2/100000p_Veff_slowVsAgile_20-60s%s%s.png' % (surfstring, halfstring))