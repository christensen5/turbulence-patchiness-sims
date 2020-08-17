import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import cluster_optics_xi, cluster_optics_dbscan
import numpy as np
import os

def all_the_optics(filepath, tspan, eps_dbscan=200):
    reachability = np.load(os.path.join(filepath, "reachability_%stspan.npy" % str(tspan)))
    core_distances = np.load(os.path.join(filepath, "core_distances_%stspan.npy" % str(tspan)))
    ordering = np.load(os.path.join(filepath, "ordering_%stspan.npy" % str(tspan)))
    predecessor = np.load(os.path.join(filepath, "predecessor_%stspan.npy" % str(tspan)))

    # dbscan_labels_inf = cluster_optics_dbscan(reachability=reachability, core_distances=core_distances, ordering=ordering,
    #                                    eps=eps)

    labels_dbscan = cluster_optics_dbscan(reachability=reachability,
                                              core_distances=core_distances,
                                              ordering=ordering,
                                              eps=eps_dbscan)
    reachability = reachability[ordering]
    labels_dbscan = labels_dbscan[ordering]
    return {"reach": reachability, "db": labels_dbscan, "eps_db": eps_dbscan, "ordering": ordering}

## MAIN
tspan = 20
path_to_trajectories = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
timesteps = list(np.arange(200, 401, 10))  # full is 20-40s
particles = list(np.arange(0, 100000))  # list(np.random.randint(0, 100000, 10000))
# load simulation trajectories
lons = np.load(os.path.join(path_to_trajectories, "lons.npy"))
lats = np.load(os.path.join(path_to_trajectories, "lats.npy"))
deps = np.load(os.path.join(path_to_trajectories, "deps.npy"))
# clip surface particles
deps = np.clip(deps, 0., 360.)
# transpose and concatenate trajectories to meet the (numparticles, 3 * numtimesteps) shape required by OPTICS.
stackedTrajectories = np.hstack((lons.transpose(), lats.transpose(), deps.transpose()))
space = np.arange(len(stackedTrajectories))

min10 = all_the_optics("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/10minSamp",
                       tspan, eps_dbscan=200)
min50 = all_the_optics("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/50minSamp",
                       tspan, eps_dbscan=200)
min100 = all_the_optics("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/100minSamp",
                        tspan, eps_dbscan=200)
min500 = all_the_optics("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/500minSamp",
                        tspan, eps_dbscan=200)

# # ====================================================================================================================
# # PLOT BASIC REACHABILITY FOR EACH MINSAMPLE VALUE
# reachabilities_2x2 = plt.figure(figsize=(15, 13))
# G = reachabilities_2x2.add_gridspec(2, 2, top=0.9)
# ax1 = reachabilities_2x2.add_subplot(G[0, 0])
# ax2 = reachabilities_2x2.add_subplot(G[0, 1])
# ax3 = reachabilities_2x2.add_subplot(G[1, 0])
# ax4 = reachabilities_2x2.add_subplot(G[1, 1])
#
# i = 0
# for ax, r in zip([ax1, ax2, ax3, ax4], [min10["reach"], min50["reach"], min100["reach"], min500["reach"]]):
#     ax.plot(r, lw=.5)
#     ax.set_title(['min10', 'min50', 'min100', 'min500'][i])
#     i += 1
# reachabilities_2x2.suptitle("Reachability plots for tspan=%ds" % tspan, fontsize=20)
# plt.show()

# # ====================================================================================================================
# # PLOT REACHABILITY (with dbscan label colours) AND SCATTER3D PLOTS AT EACH MINSAMPLE.
fig = plt.figure(figsize=(15, 12))
gs_4x2 = fig.add_gridspec(4, 3, top=0.9)

i = 0
for plotrow in [min10, min50, min100, min500]:
    reachability = plotrow["reach"]
    labels_db = plotrow["db"]
    eps_db = plotrow["eps_db"]
    ax_reach = fig.add_subplot(gs_4x2[i, 0])
    ax_patch3D_t0 = fig.add_subplot(gs_4x2[i, 1], projection='3d')
    ax_patch3D_t1 = fig.add_subplot(gs_4x2[i, 2], projection='3d')
    # Reachability plot
    num_patches = plotrow["db"].max()
    colors = ['g', 'r', 'b', 'y', 'c']*int(1 + num_patches/5)  # repeat 5 colors enough times to cover all patches
    for patch_id, color in zip(range(num_patches), colors[:num_patches]):
        Xk = space[labels_db == patch_id]
        Rk = reachability[labels_db == patch_id]
        ax_reach.plot(Xk, Rk, color, alpha=0.3, markersize=0.5)
    ax_reach.plot(space[labels_db == -1], reachability[labels_db == -1], 'k.', markersize=0.5, alpha=0.3)
    ax_reach.plot(space, np.full_like(space, eps_db, dtype=float), 'k:', alpha=0.5)
    ax_reach.set_ylabel('Reachability (epsilon distance)')
    if i==0:
        ax_reach.set_title('Reachability Plot with DBScan colouring')
    ax_reach.annotate([r'$s_{min}=10 \ \Rightarrow \ %d$ patches' %num_patches,
                       r'$s_{min}=50 \ \Rightarrow \ %d$ patches' %num_patches,
                       r'$s_{min}=100 \ \Rightarrow \ %d$ patches' %num_patches,
                       r'$s_{min}=500 \ \Rightarrow \ %d$ patches' %num_patches][i],
                      xy=(0.05, 0.9), color='k', xycoords="axes fraction")
    # 3D scatter plot at 20s
    ordering = plotrow["ordering"]
    X = lons[200, :][ordering]
    Y = lats[200, :][ordering]
    Z = deps[200, :][ordering]
    for patch_id, color in zip(range(num_patches), colors[:num_patches]):
        ax_patch3D_t0.scatter(X[labels_db==patch_id], Y[labels_db==patch_id], Z[labels_db==patch_id],
                              c=color, s=1.)
        ax_patch3D_t0.elev = 10
        ax_patch3D_t0.set_xticks([0, 180, 360, 540, 720])
        ax_patch3D_t0.set_yticks([0, 180, 360, 540, 720])
        ax_patch3D_t0.set_yticklabels([0, 180, 360, 540, 720], rotation=-45)
        ax_patch3D_t0.set_zticks([0, 180, 360])
        ax_patch3D_t0.set_zlabel("Z")
    if i==0:
        ax_patch3D_t0.set_title(r'Particle Positions at $t=20s$ with DBScan colouring')
    # 3D scatter plot at 40s
    ordering = plotrow["ordering"]
    X = lons[400, :][ordering]
    Y = lats[400, :][ordering]
    Z = deps[400, :][ordering]
    for patch_id, color in zip(range(num_patches), colors[:num_patches]):
        ax_patch3D_t1.scatter(X[labels_db == patch_id], Y[labels_db == patch_id], Z[labels_db == patch_id],
                              c=color, s=1.)
        ax_patch3D_t1.elev = 10
        ax_patch3D_t1.set_xticks([0, 180, 360, 540, 720])
        ax_patch3D_t1.set_yticks([0, 180, 360, 540, 720])
        ax_patch3D_t1.set_yticklabels([0, 180, 360, 540, 720], rotation=-45)
        ax_patch3D_t1.set_zticks([0, 180, 360])
        ax_patch3D_t1.set_zlabel("Z")
    if i == 0:
        ax_patch3D_t1.set_title(r'Particles at $t=40s$ with DBScan colouring')
    i += 1
plt.show()