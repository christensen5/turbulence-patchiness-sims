"""timestep inaccuracy analysis

This script takes two netCDF4 files containing particle position data from two simulations. These simulations should be
identical except for the timestep, and in particular the start and endtimes must be the same. The script will compare
the end points of each particle and generate histograms to display the distribution of errors between the shorter and
longer timestep."""

import sys
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

def main():
    # Load data from netCDF files.
    nc_dt = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/100x0.001s.nc")
    nc_dT = netCDF4.Dataset("/home/alexander/Desktop/temp_maarten/1x0.1s.nc")

    # Ensure start times match, and find matching endtime.
    assert np.allclose(nc_dt.variables["time"][0][0], nc_dT.variables["time"][0][0]), "Datasets have different start times."
    endtime_ind_dt = np.argwhere(np.in1d(nc_dt.variables["time"][0][:], nc_dT.variables["time"][0][:]))[-1]
    endtime_ind_dT = np.argwhere(np.in1d(nc_dT.variables["time"][0][:], nc_dt.variables["time"][0][:]))[-1]

    # Ensure particle starting positions match.
    init_dt = np.array((nc_dt.variables["lon"][:][:, 0], nc_dt.variables["lat"][:][:, 0], nc_dt.variables["z"][:][:, 0]))
    init_dT = np.array((nc_dT.variables["lon"][:][:, 0], nc_dT.variables["lat"][:][:, 0], nc_dT.variables["z"][:][:, 0]))
    assert np.allclose(init_dt, init_dT), "Datasets have different initial particle distributions."

    # Determine particle-wise errors.
    errors_lon = nc_dt.variables["lon"][:][:, endtime_ind_dt] - nc_dT.variables["lon"][:][:, endtime_ind_dT]
    errors_lat = nc_dt.variables["lat"][:][:, endtime_ind_dt] - nc_dT.variables["lat"][:][:, endtime_ind_dT]
    errors_dep = nc_dt.variables["z"][:][:, endtime_ind_dt] - nc_dT.variables["z"][:][:, endtime_ind_dT]
    errors = np.power(np.power(errors_lon, 2) + np.power(errors_lat, 2) + np.power(errors_dep, 2), 0.5)

    print("ERROR UNITS ARE CELLS, NOT millimetres. Still too large?")

    nc_dt.close()
    nc_dT.close()

    plt.hist(errors[errors<20], bins=100)
    plt.show()


if __name__=="__main__":
    main()