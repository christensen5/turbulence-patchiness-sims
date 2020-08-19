import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from DBSCAN_reach_and_scatter3d import optics_dbscan_wrapper

# # PLOT REACHABILITY FOR EACH MINSAMPLE VALUE (rows) AND TSPAN (cols)
reachabilities_4x4 = plt.figure(figsize=(15, 15))
G = reachabilities_4x4.add_gridspec(4, 4, top=0.9)

col = 0
for tspan in [1, 5, 10, 20]:
    row = 0
    filepath = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/"
    min10 = optics_dbscan_wrapper(os.path.join(filepath, "10minSamp"), tspan)
    min50 = optics_dbscan_wrapper(os.path.join(filepath, "50minSamp"), tspan)
    min100 = optics_dbscan_wrapper(os.path.join(filepath, "100minSamp"), tspan)
    min500 = optics_dbscan_wrapper(os.path.join(filepath, "500minSamp"), tspan)
    for r in [min10["reach"], min50["reach"], min100["reach"], min500["reach"]]:
        ax = reachabilities_4x4.add_subplot(G[row, col])
        ax.plot(r, color='royalblue', marker='.', markersize=0.5, alpha=0.3)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
        if col == 0:
            ax.annotate(r'$s_{min} = %d$' % [10, 50, 100, 500][row], xy=(-0.35, 0.5), color='k', fontsize=20,
                              xycoords="axes fraction", ha="center", va="center", rotation=90)
            ax.set_ylabel("Reachability (epsilon distance)")
        if row == 0:
           ax.annotate(r'tspan $= %ds$' % tspan, xy=(0.5, 1.12), color='k', fontsize=20,
                       xycoords="axes fraction", ha="center", va="center")
        row += 1
    col += 1

# plt.show()
plt.savefig("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/figs/4x4_all_reachabilities_grid")
plt.clf()

# # PLOT REACHABILITY FOR EACH MINSAMPLE VALUE (rows) with 10TSPAN vs 10TSPAN_smalldt vs 20TSPAN (cols)
reachabilities_4x3_smalldt = plt.figure(figsize=(12, 15))
G = reachabilities_4x3_smalldt.add_gridspec(4, 3, top=0.9)

col = 0
for tspan, xarg in zip([1, 1, 10], ["", "_smalldt", ""]):
    row = 0
    filepath = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/"
    min10 = optics_dbscan_wrapper(os.path.join(filepath, "10minSamp"), tspan, None, xarg)
    min50 = optics_dbscan_wrapper(os.path.join(filepath, "50minSamp"), tspan, None, xarg)
    min100 = optics_dbscan_wrapper(os.path.join(filepath, "100minSamp"), tspan, None, xarg)
    min500 = optics_dbscan_wrapper(os.path.join(filepath, "500minSamp"), tspan, None, xarg)
    for r in [min10["reach"], min50["reach"], min100["reach"], min500["reach"]]:
        ax = reachabilities_4x3_smalldt.add_subplot(G[row, col])
        ax.plot(r, color='royalblue', marker='.', markersize=0.5, alpha=0.3)
        ax.set_ylim([0, 1750])
        plt.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
        if col == 0:
            ax.annotate(r'$s_{min} = %d$' % [10, 50, 100, 500][row], xy=(-0.35, 0.5), color='k', fontsize=20,
                              xycoords="axes fraction", ha="center", va="center", rotation=90)
            ax.set_ylabel("Reachability (epsilon distance)")
        if row == 0:
           ax.annotate(r'tspan $= %ds$%s' % (tspan, xarg), xy=(0.5, 1.12), color='k', fontsize=20,
                       xycoords="axes fraction", ha="center", va="center")
        row += 1
    col += 1

# plt.show()
plt.savefig("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/figs/4x3_reachabilities_grid_compare_1s_smalldt")

