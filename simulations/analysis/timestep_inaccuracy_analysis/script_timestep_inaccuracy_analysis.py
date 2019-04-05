"""timestep inaccuracy analysis

This script takes two netCDF4 files containing particle position data from two simulations. These simulations should be
identical except for the timestep, and in particular the start and endtimes must be the same. The script will compare
the end points of each particle and generate histograms to display the distribution of errors between the shorter and
longer timestep."""

import sys
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

def compute_trajectory_errors(sim_t, sim_T, savepath=None):
    """ This method will compare the endpoints of two simulations and generate a histogram of the "error size" for
    each particle, measured as the distance between the endpoints of the particle trajectories in the two sims.
    :param sim_t: A netCDF4 Dataset object containing particle trajectories for the simulation with the smaller timestep.
    :param sim_T: A netCDF4 Dataset object containing particle trajectories for the simulation with the larger timestep.
    :param savepath: path to save the histogram. Default value is None.
    :return:
    """

    # Ensure start times and end times match.
    assert np.allclose(sim_t.variables["time"][0][0], sim_T.variables["time"][0][0]), "Datasets have different start times."
    assert np.allclose(sim_t.variables["time"][0][-1], sim_T.variables["time"][0][-1]), "Datasets have different end times."
    # endtime_ind_dt = np.argwhere(np.in1d(sim_t.variables["time"][0][1:], sim_T.variables["time"][0][1:]))[-1]
    # endtime_ind_dT = np.argwhere(np.in1d(sim_T.variables["time"][0][1:], sim_t.variables["time"][0][1:]))[-1]

    # Ensure particle starting positions match.
    init_dt = np.array((sim_t.variables["lon"][:][:, 0], sim_t.variables["lat"][:][:, 0], sim_t.variables["z"][:][:, 0]))
    init_dT = np.array((sim_T.variables["lon"][:][:, 0], sim_T.variables["lat"][:][:, 0], sim_T.variables["z"][:][:, 0]))
    assert np.allclose(init_dt, init_dT), "Datasets have different initial particle distributions."

    # Determine particle-wise errors (accounting for periodic boundary).
    errors_lon = sim_t.variables["lon"][:][:, -1] - sim_T.variables["lon"][:][:, -1]
    errors_lon = np.where(errors_lon > 0.5 * 722, errors_lon - 722, errors_lon)
    errors_lat = sim_t.variables["lat"][:][:, -1] - sim_T.variables["lat"][:][:, -1]
    errors_lat = np.where(errors_lat > 0.5 * 722, errors_lat - 722, errors_lon)
    errors_dep = sim_t.variables["z"][:][:, -1] - sim_T.variables["z"][:][:, -1]
    errors_dep = np.where(errors_dep > 0.5 * 362, errors_dep - 362, errors_dep)
    errors = np.power(np.power(errors_lon, 2) + np.power(errors_lat, 2) + np.power(errors_dep, 2), 0.5)

    fig = plt.figure()

    # plot pdf
    plt_pdf = fig.add_subplot(1, 2, 1)
    plt_pdf.hist(errors, bins=100)
    xlim = plt_pdf.get_xlim()

    # plot cdf
    plt_cdf = fig.add_subplot(1, 2, 2)
    plt_cdf.set_xlim(xlim)
    plt_cdf.set_ylim(0., 1.)
    n = sim_T.variables['lon'].shape[0]
    x = np.sort(errors)
    y = np.array(range(n))/float(n)
    plt_cdf.plot(x, y)

    # compute cutoffs and plot as vertical red lines on both plots
    cutoff_50 = np.argmax(y >= .5)
    cutoff_95 = np.argmax(y >= 0.95)
    cutoff_99 = np.argmax(y >= 0.99)
    plt_pdf.axvline(x[cutoff_50], ymin=0., ymax=plt_pdf.get_ylim()[1], color='limegreen')
    plt_pdf.axvline(x[cutoff_95], ymin=0., ymax=plt_pdf.get_ylim()[1], color='r')
    plt_pdf.axvline(x[cutoff_99], ymin=0., ymax=plt_pdf.get_ylim()[1], color='r')
    pdf_text_ypos = 0.9 * plt_pdf.get_ylim()[1]
    plt_pdf.text(x[cutoff_50], pdf_text_ypos, "x=%0.2f" % x[cutoff_50], fontsize=18, color='limegreen')
    plt_pdf.text(x[cutoff_95], pdf_text_ypos, "x=%0.2f" % x[cutoff_95], fontsize=18, color='r')
    plt_pdf.text(x[cutoff_99], pdf_text_ypos, "x=%0.2f" % x[cutoff_99], fontsize=18, color='r')
    plt_cdf.axvline(x[cutoff_50], ymin=0., ymax=plt_cdf.get_ylim()[1], color='limegreen')
    plt_cdf.axvline(x[cutoff_95], ymin=0., ymax=1., color='r')
    plt_cdf.axvline(x[cutoff_99], ymin=0., ymax=1., color='r')
    cdf_text_ypos = 0.9 * plt_cdf.get_ylim()[1]
    plt_cdf.text(x[cutoff_50], cdf_text_ypos, "x=%0.2f" % x[cutoff_50], fontsize=18, color='limegreen')
    plt_cdf.text(x[cutoff_95], cdf_text_ypos, "x=%0.2f" % x[cutoff_95], fontsize=18, color='r')
    plt_cdf.text(x[cutoff_99], cdf_text_ypos, "x=%0.2f" % x[cutoff_99], fontsize=18, color='r')

    # set labels, titles, etc...
    plt_pdf.set_title("Histogram", fontsize=20)
    plt_pdf.set_xlabel("Distance (# cells) between endpoints", fontsize=18)
    plt_pdf.set_ylabel("Count", fontsize=18)

    plt_cdf.set_title("CDF", fontsize=20)
    plt_cdf.set_xlabel("Distance (# cells) between endpoints", fontsize=18)
    plt_cdf.set_ylabel("Fraction of Data", fontsize=18)

    plt.show()

    sim_t.close()
    sim_T.close()


if __name__=="__main__":
    # Load data from netCDF files.
    sim_dt = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/dt_expts/0.1s_total/100x0.001s.nc")
    sim_dT = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/dt_expts/0.1s_total/10x0.01s.nc")
    compute_trajectory_errors(sim_dt, sim_dT)