import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import cluster_optics_dbscan
import numpy as np
import os, sys

def optics_dbscan_wrapper(filepath, tspan, eps_dbscan=None, xargs=""):
    reachability = np.load(os.path.join(filepath, "reachability_%stspan%s.npy" % (str(tspan), str(xargs))))
    core_distances = np.load(os.path.join(filepath, "core_distances_%stspan%s.npy" % (str(tspan), str(xargs))))
    ordering = np.load(os.path.join(filepath, "ordering_%stspan%s.npy" % (str(tspan), str(xargs))))
    predecessor = np.load(os.path.join(filepath, "predecessor_%stspan%s.npy" % (str(tspan), str(xargs))))

    if eps_dbscan:
        labels_dbscan = cluster_optics_dbscan(reachability=reachability,
                                              core_distances=core_distances,
                                              ordering=ordering,
                                              eps=eps_dbscan)
    reachability = reachability[ordering]
    labels_dbscan = labels_dbscan[ordering] if eps_dbscan else None
    return {"reach": reachability, "labels_db": labels_dbscan, "eps_db": eps_dbscan, "ordering": ordering}


if __name__ == "__main__":
    eps_db = int(sys.argv[1])
    tspan = 1  # in seconds
    path_to_trajectories = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
    timesteps = list(np.arange(200, 200 + tspan*10 + 1, 1))  # full is 20-40s
    particles = list(np.arange(0, 100000))  # list(np.random.randint(0, 100000, 10000))
    # load simulation trajectories
    lons = np.load(os.path.join(path_to_trajectories, "lons.npy"))[timesteps, :]
    lats = np.load(os.path.join(path_to_trajectories, "lats.npy"))[timesteps, :]
    deps = np.load(os.path.join(path_to_trajectories, "deps.npy"))[timesteps, :]

    # transpose and concatenate trajectories to meet the (numparticles, 3 * numtimesteps) shape required by OPTICS.
    stackedTrajectories = np.hstack((lons.transpose(), lats.transpose(), deps.transpose()))
    space = np.arange(len(stackedTrajectories))

    min10 = optics_dbscan_wrapper("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/10minSamp",
                                  tspan, eps_dbscan=eps_db)
    min50 = optics_dbscan_wrapper("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/50minSamp",
                                  tspan, eps_dbscan=eps_db)
    min100 = optics_dbscan_wrapper("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/100minSamp",
                                   tspan, eps_dbscan=eps_db)
    min500 = optics_dbscan_wrapper("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/500minSamp",
                                   tspan, eps_dbscan=eps_db)

    # # ====================================================================================================================
    # # PLOT REACHABILITY (with dbscan label colours) AND SCATTER3D PLOTS AT EACH MINSAMPLE.
    fig = plt.figure(figsize=(15, 12))
    gs_4x2 = fig.add_gridspec(4, 3, top=0.9)

    i = 0
    for plotrow in [min10, min50, min100, min500]:
        reachability = plotrow["reach"]
        labels_db = plotrow["labels_db"]
        eps_db = plotrow["eps_db"]
        ax_reach = fig.add_subplot(gs_4x2[i, 0])
        ax_patch3D_t0 = fig.add_subplot(gs_4x2[i, 1], projection='3d')
        ax_patch3D_t1 = fig.add_subplot(gs_4x2[i, 2], projection='3d')
        # Reachability plot
        num_patches = plotrow["labels_db"].max()
        colors = ['g', 'r', 'b', 'y', 'c']*int(1 + num_patches/5)  # repeat 5 colors enough times to cover all patches
        for patch_id, color in zip(range(0, num_patches), colors[0:num_patches]):
            Xk = space[labels_db == patch_id]
            Rk = reachability[labels_db == patch_id]
            ax_reach.plot(Xk, Rk, color, alpha=0.3, markersize=0.5)
        ax_reach.plot(space[labels_db == -1], reachability[labels_db == -1], 'k.', markersize=0.5, alpha=0.3)
        ax_reach.plot(space, np.full_like(space, eps_db, dtype=float), 'k--', lw=0.5)
        ax_reach.set_xlim([0, 100000])
        ax_reach.set_xlabel(r'$i$')
        ax_reach.set_ylabel(r'Reachability (epsilon distance) $r(p_i)$')
        if i==0:
            ax_reach.set_title('Reachability Plot with DBScan colouring')
        ax_reach.annotate([r'$s_{min}=10 \ \Rightarrow \ %d$ patches' %num_patches,
                           r'$s_{min}=50 \ \Rightarrow \ %d$ patches' %num_patches,
                           r'$s_{min}=100 \ \Rightarrow \ %d$ patches' %num_patches,
                           r'$s_{min}=500 \ \Rightarrow \ %d$ patches' %num_patches][i],
                          xy=(0.05, 0.9), color='k', xycoords="axes fraction")
        ax_reach.annotate(r'-$\epsilon = %d$' % eps_db, xy=(ax_reach.get_xlim()[1]-1000, eps_db), color='k', ha="left", va="center")
        # 3D scatter plot at 20s
        ordering = plotrow["ordering"]
        X = lons[0, :][ordering]
        Y = lats[0, :][ordering]
        Z = deps[0, :][ordering]
        for patch_id, color in zip(range(0, num_patches), colors[:num_patches-0]):
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
        X = lons[-1, :][ordering]
        Y = lats[-1, :][ordering]
        Z = deps[-1, :][ordering]
        for patch_id, color in zip(range(0, num_patches), colors[:num_patches-0]):
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
    # plt.show()
    plt.savefig("/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/figs/4x3_dbscan_%deps_%dtspan_reach_and_scatter_all_minsamps" % (eps_db, tspan))



    ## =====================================================================================================================
    ## SNIPPET FOR PLOTTING ALL PATCHES AT A GIVEN TIMESTEP (from state as in line 121)
    # for patch_id, color in zip(range(1, num_patches), colors[:num_patches - 1]):
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(X[labels_db == patch_id], Y[labels_db == patch_id], Z[labels_db == patch_id],
    #                c=color, s=1.)
    #     ax.elev = 10
    #     ax.set_xlim([0, 720])
    #     ax.set_ylim([0, 720])
    #     ax.set_zlim([0, 360])
    #     ax.set_xticks([0, 180, 360, 540, 720])
    #     ax.set_yticks([0, 180, 360, 540, 720])
    #     ax.set_yticklabels([0, 180, 360, 540, 720], rotation=-45)
    #     ax.set_zticks([0, 180, 360])
    #     ax.set_zlabel("Z")
    #     fig.savefig("/home/alexander/Desktop/temp/%03d.png" % patch_id)
    #     plt.close()