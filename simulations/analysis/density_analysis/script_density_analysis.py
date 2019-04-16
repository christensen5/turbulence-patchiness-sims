"""density analysis

This script takes two numpy files containing particle density data from a motile and a non-motile simulation.
The script will compute the distribution of the Q-statistic for density at each timestep, and plot the corresponding
patchiness plot."""
import numpy as np
from tqdm import tqdm
from simulations.analysis.analysis_tools import plot_densities
import matplotlib.pyplot as plt

filepath_dead = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/trajectories_10000p_30s_0.01dt_0.1sdt_initunif_dead_density.npy"
filepath_mot = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/trajectories_10000p_30s_0.01dt_0.05sdt_initunif_mot_density.npy"

densities_dead = np.load(filepath_dead)
densities_mot = np.load(filepath_mot)
densities_mot = densities_mot[:, :, :, ::2]

timestamps = 12 #density_dead.shape[3]

f = 0.5
for t in tqdm(range(timestamps)):
    density_dead = densities_dead[:, :, :, t]
    density_mot = densities_mot[:, :, :, t]
    C = density_mot[np.where(density_mot >= (1 - f) * np.max(density_mot))]
    Cp = density_dead[np.where(density_dead >= (1 - f) * np.max(density_dead))]
    Cm = np.sum(density_dead)/density_dead.size
    Q = (C - Cp) / Cm
    H, bin_edges = np.histogram(Q, 100)

    width = (bin_edges[1] - bin_edges[0])

    fig = plt.figure()

    # plot pdf
    plt_pdf = fig.add_subplot(1, 2, 1)
    plt_pdf.bar(bin_edges[1:], H, width=width)
    x = bin_edges
    xlim = plt_pdf.get_xlim()
    # # compute cutoffs and plot as vertical red lines on both plots
    # cutoff_50 = np.argmax(y >= 0.5)
    # cutoff_95 = np.argmax(y >= 0.95)
    # cutoff_99 = np.argmax(y >= 0.99)
    # plt_pdf.axvline(x[cutoff_50], ymin=0., ymax=plt_pdf.get_ylim()[1], color='limegreen')
    # plt_pdf.axvline(x[cutoff_95], ymin=0., ymax=plt_pdf.get_ylim()[1], color='r')
    # plt_pdf.axvline(x[cutoff_99], ymin=0., ymax=plt_pdf.get_ylim()[1], color='r')
    # pdf_text_ypos = 0.9 * plt_pdf.get_ylim()[1]
    # plt_pdf.text(x[cutoff_50], pdf_text_ypos, "x=%0.2f" % x[cutoff_50], fontsize=18, color='limegreen')
    # plt_pdf.text(x[cutoff_95], pdf_text_ypos, "x=%0.2f" % x[cutoff_95], fontsize=18, color='r')
    # plt_pdf.text(x[cutoff_99], pdf_text_ypos, "x=%0.2f" % x[cutoff_99], fontsize=18, color='r')
    # set labels, titles, etc...
    plt_pdf.set_title("Histogram", fontsize=20)
    plt_pdf.set_xlabel("Q Statistic", fontsize=18)
    plt_pdf.set_ylabel("Count", fontsize=18)

