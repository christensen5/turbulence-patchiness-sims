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

    # Ensure start times match, and find matching endtime.
    assert np.allclose(sim_dt.variables["time"][0][0], sim_dT.variables["time"][0][0]), "Datasets have different start times."
    endtime_ind_dt = np.argwhere(np.in1d(sim_dt.variables["time"][0][:], sim_dT.variables["time"][0][:]))[-1]
    endtime_ind_dT = np.argwhere(np.in1d(sim_dT.variables["time"][0][:], sim_dt.variables["time"][0][:]))[-1]

    # Ensure particle starting positions match.
    init_dt = np.array((sim_dt.variables["lon"][:][:, 0], sim_dt.variables["lat"][:][:, 0], sim_dt.variables["z"][:][:, 0]))
    init_dT = np.array((sim_dT.variables["lon"][:][:, 0], sim_dT.variables["lat"][:][:, 0], sim_dT.variables["z"][:][:, 0]))
    assert np.allclose(init_dt, init_dT), "Datasets have different initial particle distributions."

    # Determine particle-wise errors (accounting for periodic boundary).
    errors_lon = sim_dt.variables["lon"][:][:, endtime_ind_dt] - sim_dT.variables["lon"][:][:, endtime_ind_dT]
    errors_lon = np.where(errors_lon > 0.5 * 722, errors_lon - 722, errors_lon)
    errors_lat = sim_dt.variables["lat"][:][:, endtime_ind_dt] - sim_dT.variables["lat"][:][:, endtime_ind_dT]
    errors_lat = np.where(errors_lat > 0.5 * 722, errors_lat - 722, errors_lon)
    errors_dep = sim_dt.variables["z"][:][:, endtime_ind_dt] - sim_dT.variables["z"][:][:, endtime_ind_dT]
    errors_dep = np.where(errors_dep > 0.5 * 362, errors_dep - 362, errors_dep)
    errors = np.power(np.power(errors_lon, 2) + np.power(errors_lat, 2) + np.power(errors_dep, 2), 0.5)

    print("ERROR UNITS ARE CELLS, NOT millimetres. Still too large? Also PER 0.1 SECONDS.")

    plt.hist(errors, bins=100)
    plt.title("Histogram -- Trajectory Endpoint Difference after %1.3f s." % sim_dT.variables["time"][0][endtime_ind_dT])
    plt.xlabel("Distance (# cells) between endpoints")
    plt.ylabel("Count")
    plt.show()

    sim_dt.close()
    sim_dT.close()


if __name__=="__main__":
    # Load data from netCDF files.
    sim_dt = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/100x0.001s.nc")
    sim_dT = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/20x0.005s.nc")
    compute_trajectory_errors(sim_dt, sim_dT)