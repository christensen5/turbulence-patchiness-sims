import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.cluster import cluster_optics_xi, cluster_optics_dbscan
import numpy as np
import time
import os


path_to_trajectories = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot"
timesteps = list(np.arange(200, 401, 10))  # full is 20-40s
timesteps = list(np.linspace(200, 400, 11, dtype=int))  # always 10 timesteps
particles = list(np.arange(0, 100000))  #list(np.random.randint(0, 100000, 10000))

save_clustering = True

# load simulation trajectories
lons = np.load(os.path.join(path_to_trajectories, "lons.npy"))[timesteps, :][:, particles]
lats = np.load(os.path.join(path_to_trajectories, "lats.npy"))[timesteps, :][:, particles]
deps = np.load(os.path.join(path_to_trajectories, "deps.npy"))[timesteps, :][:, particles]

# # remove particles that ever breached surface
# breached_particles = np.unique(np.argwhere(deps > 360)[:, 1])
# keep_indicies = [i for i in np.arange(0, len(particles)) if i not in breached_particles]
# lons = lons[:, keep_indicies]
# lats = lats[:, keep_indicies]
# deps = deps[:, keep_indicies]

# clip surface particles
# deps = np.clip(deps, 0., 360.)

# transpose and concatenate trajectories to meet the (numparticles, 3 * numtimesteps) shape required by OPTICS.
stackedTrajectories = np.hstack((lons.transpose(), lats.transpose(), deps.transpose()))

# define OPTICS parameters
min_samples = 50

# define clustering method parameters
# xi =
eps = np.inf

# Assemble the Padberg-Gehle-Schneide network, compute reachability & core-distances for each trajectory and obtain the ordering.
print("Beginning OPTICS clustering.")
t0 = time.time()
optics_clustering = OPTICS(min_samples=min_samples, metric="euclidean").fit(stackedTrajectories)
t1 = time.time()
print("OPTICS clustering took {:.2f} seconds.".format(t1-t0))
reachability = optics_clustering.reachability_
core_distances = optics_clustering.core_distances_
predecessor = optics_clustering.predecessor_
ordering = optics_clustering.ordering_

if save_clustering:
    savedir = "/home/alexander/Documents/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/optics/%dminSamp" % min_samples
    tspan = len(timesteps) - 1  # if np.arange method
    tspan = (timesteps[-1] - timesteps[0]) / 10
    np.save(os.path.join(savedir, "reachability_%dtspan_smalldt" % tspan), reachability)
    np.save(os.path.join(savedir, "core_distances_%dtspan_smalldt" % tspan), core_distances)
    np.save(os.path.join(savedir, "predecessor_%dtspan_smalldt" % tspan), predecessor)
    np.save(os.path.join(savedir, "ordering_%dtspan_smalldt" % tspan), ordering)

# eps_dbscan = 500
# dbscan_labels = cluster_optics_dbscan(reachability=reachability, core_distances=core_distances, ordering=ordering,
#                                    eps=eps_dbscan)
#
# space = np.arange(len(stackedTrajectories))
# reachability = optics_clustering.reachability_[ordering]
# labels = optics_clustering.labels_[ordering]

# plt.figure(figsize=(10, 7))
# G = gridspec.GridSpec(2, 2)
# ax1 = plt.subplot(G[0, :])
# ax2 = plt.subplot(G[1, 0])
# ax3 = plt.subplot(G[1, 1])
# # ax4 = plt.subplot(G[1, 2])
#
# # Reachability plot
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = space[labels == klass]
#     Rk = reachability[labels == klass]
#     ax1.plot(Xk, Rk, color, alpha=0.3)
# ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
# ax1.plot(space, np.full_like(space, 500., dtype=float), 'k-', alpha=0.5)
# ax1.set_ylabel('Reachability (epsilon distance)')
# ax1.set_title('Reachability Plot')
#
# # OPTICS
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = stackedTrajectories[optics_clustering.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(stackedTrajectories[optics_clustering.labels_ == -1, 0], stackedTrajectories[optics_clustering.labels_ == -1, 1], 'k+', alpha=0.1)
# ax2.set_title('Automatic Clustering\nOPTICS')
#
# # DBSCAN at eps
# colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
# for klass, color in zip(range(0, 6), colors):
#     Xk = stackedTrajectories[dbscan_labels == klass]
#     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
# ax3.plot(stackedTrajectories[dbscan_labels == -1, 0], stackedTrajectories[dbscan_labels == -1, 1], 'k+', alpha=0.1)
# ax3.set_title('Clustering at {:.1f} epsilon cut\nDBSCAN'.format(eps_dbscan))
# plt.show()

print("Yay")



