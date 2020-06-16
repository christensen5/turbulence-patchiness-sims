import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns; sns.set(font_scale=.8)
import os, sys
import matplotlib.gridspec as gridspec
import netCDF4

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

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

# ======================================================================================================================
# SCATTER PLOT Veff vs Depth AT 6 TIMESNAPS.

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

# Extract particle dir and particlewise fluid velocity for each motile simulation and compute Veff.
timestamps = np.arange(0, 601, 10)
timesteps_for_jointplots = np.arange(20, 60, 10)
cells_to_m = 1./1200  # conversion factor from cells/s to m/s
print("Conversion factor set to %f - ensure this is correct." % cells_to_m)
splots_mag = []
splots_w = []
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

    # convert to mm/s
    Veff_u *= (1000 * cells_to_m)
    Veff_v *= (1000 * cells_to_m)
    Veff_w *= (1000 * cells_to_m)
    Veff_mag = np.power(np.power(Veff_u, 2) + np.power(Veff_v, 2) + np.power(Veff_w, 2), 0.5)

    Veff_w_t = Veff_w[:, timesteps_for_jointplots].flatten()
    Veff_mag_t = Veff_mag[:, timesteps_for_jointplots].flatten()
    deps_t = deps[:, timesteps_for_jointplots].flatten()

    plot_mag = sns.jointplot(Veff_mag_t, deps_t, xlim=[0, 175], ylim=[180, 360],
                         marginal_kws=dict(bins=100, norm_hist=True),
                         s=0.5, alpha=0.05)
    plot_w = sns.jointplot(Veff_w_t, deps_t, xlim=[-150, 100], ylim=[180, 360],
                         marginal_kws=dict(bins=100, norm_hist=True),
                         s=0.5, alpha=0.05)
    # plt.text(0.9*xlims[0], 1.1*ylims[1], str(t) + "s", fontsize=16, horizontalalignment='left',
        #          verticalalignment='top')  # , transform=ax.transAxes)
    splots_mag.append(plot_mag)
    splots_w.append(plot_w)
    if sim_id % 3 == 1:
        splots_mag[-1].set_axis_labels("", "B = %d" % sim["B"])
        splots_w[-1].set_axis_labels("", "B = %d" % sim["B"])
    if sim_id < 4:
        splots_mag[-1].fig.suptitle("V = %d" % sim["V"])
        splots_w[-1].fig.suptitle("V = %d" % sim["V"])
    if sim_id == 7:
        splots_mag[-1].set_axis_labels("|Veff| (mm)", "B = %d" % sim["B"])
        splots_w[-1].set_axis_labels("|Veff| (mm)", "B = %d" % sim["B"])
    if sim_id > 7:
        splots_mag[-1].set_axis_labels("|Veff| (mm)", "")
        splots_w[-1].set_axis_labels("|Veff| (mm)", "")

# Compose and save magnitude plots
fig = plt.figure(figsize=(15.2, 9))
gs = gridspec.GridSpec(3, 3)
for i in range(len(splots_mag)):
    SeabornFig2Grid(splots_mag[i], fig, gs[i])
gs.update(top=0.95)
fig.suptitle(r'Effective speed vs depth $(B=%3.1fs^{-1}, v=%dums^{-1})$' % (sim["B"], sim["V"]), fontsize=18)
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/Veff/VeffMagvsDepth/VeffMagvsDepth_%3.1fB_%dv_semilog_timesnaps.png" % (sim["B"], sim["V"]))
# plt.show()
plt.clf()

# Compose and save W plots
fig = plt.figure(figsize=(15.2, 9))
gs = gridspec.GridSpec(3, 3)
for i in range(len(splots_w)):
    SeabornFig2Grid(splots_w[i], fig, gs[i])
gs.update(top=0.95)
fig.suptitle(r'Effective upwards velocity vs depth $(B=%3.1fs^{-1}, v=%dums^{-1})$' % (sim["B"], sim["V"]), fontsize=18)
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/Veff/VeffWvsDepth/VeffWvsDepth_%3.1fB_%dv_semilog_timesnaps.png" % (sim["B"], sim["V"]))
# plt.sho