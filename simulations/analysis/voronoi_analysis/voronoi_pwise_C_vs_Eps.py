import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns; sns.set(font_scale=.8)
import os, sys
import matplotlib.gridspec as gridspec

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

# specify paths to simulation output files
filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead"
filepath_v10_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot"
filepath_v10_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot"
filepath_v10_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot"
filepath_v100_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot"
filepath_v100_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot"
filepath_v100_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot"
filepath_v500_B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
filepath_v500_B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot"
filepath_v500_B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot"

# load relevant files
data_dead = {'conc': np.reciprocal(np.load(os.path.join(filepath_dead, 'vols_v.npy'))),
             'eps': np.load(os.path.join(filepath_dead, 'eps_vor.npy')),
             'V': 'dead', 'B': 'dead'}
data_v10_B1 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v10_B1, 'vols_v.npy'))),
               'eps': np.load(os.path.join(filepath_v10_B1, 'eps_vor.npy')),
               'V': 10, 'B': 1.0}
data_v10_B3 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v10_B3, 'vols_v.npy'))),
               'eps': np.load(os.path.join(filepath_v10_B3, 'eps_vor.npy')),
               'V': 10, 'B': 3.0}
data_v10_B5 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v10_B5, 'vols_v.npy'))),
               'eps': np.load(os.path.join(filepath_v10_B5, 'eps_vor.npy')),
               'V': 10, 'B': 5.0}
data_v100_B1 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v100_B1, 'vols_v.npy'))),
                'eps': np.load(os.path.join(filepath_v100_B1, 'eps_vor.npy')),
                'V': 100, 'B': 1.0}
data_v100_B3 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v100_B3, 'vols_v.npy'))),
                'eps': np.load(os.path.join(filepath_v100_B3, 'eps_vor.npy')),
                'V': 100, 'B': 3.0}
data_v100_B5 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v100_B5, 'vols_v.npy'))),
                'eps': np.load(os.path.join(filepath_v100_B5, 'eps_vor.npy')),
                'V': 100, 'B': 5.0}
data_v500_B1 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v500_B1, 'vols_v.npy'))),
                'deps': np.reciprocal(np.load(os.path.join(filepath_v500_B1, 'vols_d.npy'))),
                'eps': np.load(os.path.join(filepath_v500_B1, 'eps_vor.npy')),
                'V': 500, 'B': 1.0}
data_v500_B3 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v500_B3, 'vols_v.npy'))),
                'eps': np.load(os.path.join(filepath_v500_B3, 'eps_vor.npy')),
                'V': 500, 'B': 3.0}
data_v500_B5 = {'conc': np.reciprocal(np.load(os.path.join(filepath_v500_B5, 'vols_v.npy'))),
                'eps': np.load(os.path.join(filepath_v500_B5, 'eps_vor.npy')),
                'V': 500, 'B': 5.0}

all_motile_data = [data_v10_B1, data_v10_B3, data_v10_B5,
                   data_v100_B1, data_v100_B3, data_v100_B5,
                   data_v500_B1, data_v500_B3, data_v500_B5]


# ======================================================================================================================
# SCATTER PLOT C vs EPS AT 6 TIMESNAPS.

# keep or remove surface particles
nosurf = True
surfstring = "_nosurf" if nosurf is True else ""

# find max and min eps (for plot limits)
ymin = data_dead["eps"].min()
ymax = data_dead["eps"].max()
for simdata_dict in all_motile_data:
    mineps = np.amin(simdata_dict["eps"])
    maxeps = np.amax(simdata_dict["eps"])
    if mineps < ymin:
        ymin = mineps
    if maxeps > ymax:
        ymax = maxeps
xlims = [-6, -1] if nosurf else [-6, 12]
ylims = [ymin, ymax]

f = 0.1
timesteps_for_jointplots = [0, 12, 24, 36, 48, 60]
# motile sim plots
for simdata_dict in tqdm(all_motile_data):
    splots = []
    for t in timesteps_for_jointplots:
        conc = simdata_dict["conc"][:, t]
        eps = simdata_dict["eps"][:, t]
        # remove surface particles
        if nosurf:
            conc = conc[eps != 0]
            eps = eps[eps != 0]
        plot = sns.jointplot(np.log10(conc), eps, xlim=xlims, ylim=ylims,
                                    marginal_kws=dict(bins=100, norm_hist=True),
                                    s=0.5, alpha=0.05)
        plot.ax_marg_x.axvline(x=np.median(np.log10(conc)))  # median conc of all particles
        patchmin = conc[conc.argsort()[-int(f * conc.size)]]  # index of lowest conc value in patches
        patchall = conc[conc.argsort()[-int(f * conc.size):]]  # indicies of all conc values in patches
        plot.ax_marg_x.fill_between(np.linspace(np.log10(patchmin), np.log10(conc.max()), 100),
                                    plot.ax_marg_x.get_ylim()[0], plot.ax_marg_x.get_ylim()[1],
                                     color="red", alpha=0.2)  # shade patches in red
        plot.ax_marg_x.axvline(x=np.log10(np.median(patchall)), color="red",
                               linestyle="--")  # median conc of particles in patches
        plt.text(0.9*xlims[0], 1.1*ylims[1], str(t)+"s", fontsize=16, horizontalalignment='left', verticalalignment='top')
        splots.append(plot)
        if t == 0:
            splots[-1].set_axis_labels("", "epsilon")
        elif t == 36:
            splots[-1].set_axis_labels("log(C)", "epsilon")
        elif t > 36:
            splots[-1].set_axis_labels("log(C)", "")


    fig = plt.figure(figsize=(15.2, 9))
    gs = gridspec.GridSpec(2, 3)

    for i in range(len(splots)):
        SeabornFig2Grid(splots[i], fig, gs[i])


    gs.update(top=0.95)
    if nosurf:
        fig.suptitle(r'Semi-log Voronoi microbe concentration vs turbulent dissipation rate (excl. surface microbes)$(B=%3.1fs^{-1}, v=%dums^{-1})$' % (simdata_dict["B"], simdata_dict["V"]), fontsize=18)
    else:
        fig.suptitle(r'Semi-log Voronoi microbe concentration vs turbulent dissipation rate (incl. surface microbes)$(B=%3.1fs^{-1}, v=%dums^{-1})$' % (simdata_dict["B"], simdata_dict["V"]), fontsize=18)
    fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/C/CvsEps/CvsEps_vor_%3.1fB_%dv%s_semilog_timesnaps.png" % (simdata_dict["B"], simdata_dict["V"], surfstring))
    # plt.show()
    plt.clf()

#non-motile sim plot
splots = []
for t in timesteps_for_jointplots:
    conc = data_dead["conc"][:, t]
    eps = data_dead["eps"][:, t]
    # remove surface particles
    if nosurf:
        conc = conc[eps != 0]
        eps = eps[eps != 0]
    plot = sns.jointplot(np.log10(conc), eps, xlim=xlims, ylim=ylims,
                         marginal_kws=dict(bins=100, norm_hist=True),
                         s=0.5, alpha=0.05)
    plot.ax_marg_x.axvline(x=np.median(np.log10(conc)))  # median conc of all particles
    patchmin = conc[conc.argsort()[-int(f * conc.size)]]  # index of lowest conc value in patches
    patchall = conc[conc.argsort()[-int(f * conc.size):]]  # indicies of all conc values in patches
    plot.ax_marg_x.fill_between(np.linspace(np.log10(patchmin), np.log10(conc.max()), 100),
                                plot.ax_marg_x.get_ylim()[0], plot.ax_marg_x.get_ylim()[1],
                                 color="red", alpha=0.2)  # shade patches in red
    plot.ax_marg_x.axvline(x=np.log10(np.median(patchall)), color="red",
                           linestyle="--")  # median conc of particles in patches
    plt.text(0.9*xlims[0], 1.1*ylims[1], str(t)+"s", fontsize=16, horizontalalignment='left', verticalalignment='top')
    splots.append(plot)
    if t == 0:
        splots[-1].set_axis_labels("", "epsilon")
    elif t == 36:
        splots[-1].set_axis_labels("log(C)", "epsilon")
    elif t > 36:
        splots[-1].set_axis_labels("log(C)", "")


fig = plt.figure(figsize=(15.2, 9))
gs = gridspec.GridSpec(2, 3)

for i in range(len(splots)):
    SeabornFig2Grid(splots[i], fig, gs[i])


gs.update(top=0.95)

if nosurf:
    fig.suptitle(r'Semi-log Voronoi microbe concentration vs turbulent dissipation rate (excl. surface microbes)(non-motile)', fontsize=18)
else:
    fig.suptitle(r'Semi-log Voronoi microbe concentration vs turbulent dissipation rate (incl. surface microbes)(non-motile)', fontsize=18)
fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/vor/C/CvsEps/CvsEps_vor_dead%s_semilog_timesnaps.png" % surfstring)