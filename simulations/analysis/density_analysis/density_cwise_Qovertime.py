#=======================================================================================================================
#cwise density superimposed Q plots
#=======================================================================================================================
# Q_B = np.zeros((5, len(timestamps)))
# for t in tqdm(timestamps):
#     density_dead = densities_dead[:, :, :, t].flatten()
#     density_B1 = densities_B1[:, :, :, t].flatten()
#     density_B2 = densities_B2[:, :, :, t].flatten()
#     density_B3 = densities_B3[:, :, :, t].flatten()
#     density_B5 = densities_B5[:, :, :, t].flatten()
#     density_B7 = densities_B7[:, :, :, t].flatten()
#     density_Bs = np.vstack((density_B1, density_B2, density_B3, density_B5, density_B7))
#
#     Cm = np.sum(density_dead)/density_dead.size
#
#     fig = plt.figure(figsize=(12, 9))
#
#     C = np.zeros((5, int(f * density_Bs.shape[1])))
#     for row in range(density_Bs.shape[0]):
#         cols = density_Bs[row, :].argsort()[-int(f * density_Bs.shape[1]):]
#         C[row, :] = density_Bs[row, cols]
#     Cp = density_dead[density_dead.argsort()[-int(f * density_dead.size):]]
#
#     Q_B[:, t] = (np.mean(C, axis=1) - np.mean(Cp)) / Cm
#
# fig = plt.figure(figsize=(12, 9))
# plt.box(False)
# ax = plt.subplot(111)
# colours = np.zeros((3, 5))
# colours[0, :] = np.linspace(0, 1, 5)
# ax.plot(times, Q_B[0, :], '-o', color=colours[:, 0], linewidth=2, markersize=3, label='B=1.0')
# ax.plot(times, Q_B[1, :], '-o', color=colours[:, 1], linewidth=2, markersize=3, label='B=2.0')
# ax.plot(times, Q_B[2, :], '-o', color=colours[:, 2], linewidth=2, markersize=3, label='B=3.0')
# ax.plot(times, Q_B[3, :], '-o', color=colours[:, 3], linewidth=2, markersize=3, label='B=5.0')
# ax.plot(times, Q_B[4, :], '-o', color=colours[:, 4], linewidth=2, markersize=3, label='B=7.0')
# plt.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], 'k')
# ax.set_title("Q statistic over time for differing B-values, f=%0.3f" %f, fontsize=25)
# ax.set_xlabel("Time", fontsize=25)
# ax.set_ylabel("Q", fontsize=25)
# ax.set_ylim(-1.5, 1.5)
# for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
# for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
# ax.legend(fontsize=25)
#
# # plt.show()
# fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/comparison/high/Q_B_entropy_0.001f.png")
#
# #=========================================================================================================
# #=========================================================================================================
# #=========================================================================================================
# #=========================================================================================================
#
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/dead/100000p_30s_0.01dt_0.05sdt_initunif_dead/density_high.npy"
# filepath_v0pt1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.1vswim/sim2/density_high.npy"
# filepath_v0pt5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.5vswim/sim2/density_high.npy"
# # filepath_v1pt0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim/density_high.npy"
# filepath_v1pt5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.5vswim/sim2/density_high.npy"
# filepath_v2pt0 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_2.0vswim/sim2/density_high.npy"
#
# densities_dead = np.load(filepath_dead)
# densities_v1 = np.load(filepath_v0pt1)
# densities_v2 = np.load(filepath_v0pt5)
# # densities_v3 = np.load(filepath_v1pt0)
# densities_v5 = np.load(filepath_v1pt5)
# densities_v7 = np.load(filepath_v2pt0)
#
# times = np.arange(0, 31, 1)
# timestamps = np.arange(0, 31, 1)
# # f = 0.1
#
# Q_v = np.zeros((5, len(timestamps)))
# for t in tqdm(timestamps):
#     density_dead = densities_dead[:, :, :, t].flatten()
#     density_v1 = densities_v1[:, :, :, t].flatten()
#     density_v2 = densities_v2[:, :, :, t].flatten()
#     density_v3 = np.zeros((density_v2.shape)) #densities_v3[:, :, :, t].flatten()
#     density_v5 = densities_v5[:, :, :, t].flatten()
#     density_v7 = densities_v7[:, :, :, t].flatten()
#     density_vs = np.vstack((density_v1, density_v2, density_v3, density_v5, density_v7))
#
#     Cm = np.sum(density_dead)/density_dead.size
#
#     fig = plt.figure(figsize=(12, 9))
#
#     C = np.zeros((5, int(f * density_vs.shape[1])))
#     for row in range(density_vs.shape[0]):
#         cols = density_vs[row, :].argsort()[-int(f * density_vs.shape[1]):]
#         C[row, :] = density_vs[row, cols]
#     Cp = density_dead[density_dead.argsort()[-int(f * density_dead.size):]]
#
#     Q_v[:, t] = (np.mean(C, axis=1) - np.mean(Cp)) / Cm
#
# fig = plt.figure(figsize=(12, 9))
# plt.box(False)
# ax = plt.subplot(111)
# colours = np.zeros((3, 5))
# colours[1, :] = np.linspace(0, 1, 5)
# ax.plot(times, Q_v[0, :], '-o', color=colours[:, 0], linewidth=2, markersize=3, label='v=0.1')
# ax.plot(times, Q_v[1, :], '-o', color=colours[:, 1], linewidth=2, markersize=3, label='v=0.5')
# ax.plot(times, Q_v[2, :], '-o', color=colours[:, 2], linewidth=2, markersize=3, label='v=DNF')
# ax.plot(times, Q_v[3, :], '-o', color=colours[:, 3], linewidth=2, markersize=3, label='v=1.5')
# ax.plot(times, Q_v[4, :], '-o', color=colours[:, 4], linewidth=2, markersize=3, label='v=2.0')
# plt.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], 'k')
# ax.set_title("Q statistic over time for differing v-values, f=%0.3f" %f, fontsize=25)
# ax.set_xlabel("Time", fontsize=25)
# ax.set_ylabel("Q", fontsize=25)
# ax.set_ylim(-1.5, 1.5)
# for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
# for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
# ax.legend(fontsize=25)
#
# # plt.show()
# fig.savefig("/media/alexander/DATA/Ubuntu/Maarten/outputs/results022/initunif/comparison/high/Q_v_density_sim2_%0.3f.png" %f)