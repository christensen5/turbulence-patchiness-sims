import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
# import seaborn as sns
import os, sys
import matplotlib.gridspec as gridspec

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade
hv.extension('matplotlib')

# class SeabornFig2Grid():
#
#     def __init__(self, seaborngrid, fig,  subplot_spec):
#         self.fig = fig
#         self.sg = seaborngrid
#         self.subplot = subplot_spec
#         if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
#             isinstance(self.sg, sns.axisgrid.PairGrid):
#             self._movegrid()
#         elif isinstance(self.sg, sns.axisgrid.JointGrid):
#             self._movejointgrid()
#         self._finalize()
#
#     def _movegrid(self):
#         """ Move PairGrid or Facetgrid """
#         self._resize()
#         n = self.sg.axes.shape[0]
#         m = self.sg.axes.shape[1]
#         self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
#         for i in range(n):
#             for j in range(m):
#                 self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])
#
#     def _movejointgrid(self):
#         """ Move Jointgrid """
#         h= self.sg.ax_joint.get_position().height
#         h2= self.sg.ax_marg_x.get_position().height
#         r = int(np.round(h/h2))
#         self._resize()
#         self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)
#
#         self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
#         self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
#         self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])
#
#     def _moveaxes(self, ax, gs):
#         #https://stackoverflow.com/a/46906599/4124317
#         ax.remove()
#         ax.figure=self.fig
#         self.fig.axes.append(ax)
#         self.fig.add_axes(ax)
#         ax._subplotspec = gs
#         ax.set_position(gs.get_position(self.fig))
#         ax.set_subplotspec(gs)
#
#     def _finalize(self):
#         plt.close(self.sg.fig)
#         self.fig.canvas.mpl_connect("resize_event", self._resize)
#         self.fig.canvas.draw()
#
#     def _resize(self, evt=None):
#         self.sg.fig.set_size_inches(self.fig.get_size_inches())

# ======================================================================================================================
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

# # Joint plot with distributions on axes
# sns.set(font_scale=.8, rc={'figure.figsize':(15,15)})
# plot = sns.jointplot(epsilon_array[:, 3], epsilon_array[:, 1],
#                          #marginal_kws=dict(bins=100, norm_hist=True),
#                          s=0.5, alpha=0.1)
# # Draw boundaries between depth ranges (see voronoi_Qovertime_vs_Depth.py) and shade them.
# plot.ax_joint.axhline(y=0.1, color="k", ls=":", lw=1)
# plot.ax_joint.axhline(y=0.17, color="k", ls=":", lw=1)
# plot.ax_joint.axhline(y=0.24, color="k", ls=":", lw=1)
# plot.ax_joint.axhline(y=0.3, color="k", ls=":", lw=1)
# plot.ax_joint.fill_between(np.linspace(plot.ax_joint.get_xlim()[0], plot.ax_joint.get_xlim()[1], 100),
#                            0.1, 0.17,#plot.ax_joint.get_ylim()[1],
#                            color="k", alpha=0.2)
# plot.set_axis_labels("epsilon", "depth (m)")
# plt.subplots_adjust(top=0.9)
# plot.fig.suptitle(r'Fluid simulation depth vs turbulent dissipation rate.', fontsize=16)
# # plt.show()
# plot.savefig(
#     "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/DepthvsEps/DepthvsEps_withDist.png")
# # plt.show()
# plt.clf()

#
# # Paper plot# eps_plot = plt.scatter(x[xidx],y[yidx], s=1, edgecolors="none", alpha=0.7)

# # sns.set_style("white")
# # sns.despine()
# plot = sns.scatterplot(x=epsilon_array[:, 3], y=epsilon_array[:, 1], linewidth=0, alpha=0.2, s=10)
# # Draw boundaries between depth ranges (see voronoi_Qovertime_vs_Depth.py) and shade them.
# plot.axhline(y=0.1, color="k", ls=":", lw=1)
# plot.axhline(y=0.17, color="k", ls=":", lw=1)
# plot.axhline(y=0.24, color="k", ls=":", lw=1)
# plot.axhline(y=0.3, color="k", ls=":", lw=1)
# plot.fill_between(np.linspace(plot.get_xlim()[0], plot.get_xlim()[1], 100),
#                            0.1, 0.3,
#                            color="k", alpha=0.15)
# plot.set(xlabel=r"$\epsilon$ [$m^2 s^{-3}$]", ylabel="depth [m]",
#          xlim=[-0.2e-4, 3.4e-4])
# # show depth region boundaries on right-hand y-axis
# ax_right = plot.twinx()
# ax_right.set(ylim=[0, 0.3])
# ax_right.set_ylim(plot.get_ylim())
# ax_right.set_yticks([0.10, 0.17, 0.24, 0.30])
# # label depth regions

# plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
# plt.subplots_adjust(top=0.9)
# plot.set_title(r'Fluid DNS turbulent dissipation rate ($\epsilon$) vs depth', fontsize=16)
# plt.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/DepthvsEps/DepthvsEps.png")
# # plt.show()
# # plt.clf()

# generate scatter plot with holoviews
from matplotlib.ticker import ScalarFormatter
xfmt = ScalarFormatter()
xfmt.set_powerlimits((0, 0))
xfmt.set_scientific(True)
points_eps = hv.Points(np.array([epsilon_array[:, 3], epsilon_array[:, 1]]).transpose())
<<<<<<< HEAD
# tempsWithDeps = tempsWithDeps[np.random.choice(tempsWithDeps.shape[0], 10000), :]
# generate scatter plots with holoviews
points_temps = hv.Points(tempsWithDeps)
=======
tempsWithDeps = tempsWithDeps[np.random.choice(tempsWithDeps.shape[0], 100000000), :]
twd_plot = hv.Bivariate(tempsWithDeps)
hv.save(twd_plot, "/home/alexander/Desktop/temp/twd.png")

# generate scatter plots with holoviews
points_temps = hv.Points(tempsWithDeps)
print("points_temps generated.")
>>>>>>> cfb092486ebc5b7de6bdaf3f7b1c76474437dc97
text_eps = hv.Text(0.000365, 0.27, text='Shallow', halign='right', valign='center') * \
        hv.Text(0.000365, 0.205, text='Mid', halign='right', valign='center') * \
        hv.Text(0.000365, 0.135, text='Deep', halign='right', valign='center')
text_temps = hv.Text(-0.2, 0.27, text='Shallow', halign='left', valign='center') * \
        hv.Text(-0.2, 0.205, text='Mid', halign='left', valign='center') * \
        hv.Text(-0.2, 0.135, text='Deep', halign='left', valign='center')
hlines = hv.HLine(0.1) * hv.HLine(0.17) * hv.HLine(0.24) * hv.HLine(0.3)



# points.opts(alpha=0.1, s=50,
#             xlabel=r"$\epsilon$ [$m^2 s^{-3}$]",
#             ylabel=r"Depth [m]",
#             xformatter=xfmt,
#             xlim=(0, 0.00035),
#             # fontscale=2.5,
#             aspect=1,
#             fig_inches=15,
#             edgecolors='none')
# hlines.opts(color=['k']*4,
#             linewidth=1.,
#             linestyle=':')
(points_eps * text_eps * hlines).opts(
    opts.Points(alpha=0.1, s=50,
            xlabel=r"$\epsilon$ [$m^2 s^{-3}$]",
            ylabel=r"Depth [m]",
            xformatter=xfmt,
            xlim=(-0.00002, 0.00037),
            # fontscale=2.5,
            aspect=1,
            fig_inches=11,
            edgecolors='none'),
    opts.HLine(color='k',
            linewidth=1.,
            linestyle=':')
)

print("points_eps options set.")


(points_temps * text_temps * hlines).opts(
        opts.Points(alpha=0.1, s=2,
            xlabel=r"Temperature (relative to reference temp $\theta_{z_0}$) [$^{\circ}C$]",
            ylabel=r"Depth [m]",
            # fontscale=2.5,
            aspect=1,
            fig_inches=11,
            edgecolors='none'),
        opts.HLine(color='k',
                   linewidth=1.,
                   linestyle=':')
)

print("points_temps options set.")
hv.save((points_eps * text_eps * hlines) + (points_temps * text_temps * hlines), "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/comparison/DepthvsEpsandTemps.png", fmt='svg')
print("hv plot saved.")
