import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
from scipy import stats
from mayavi import mlab

all = ['reformat_for_animate', 'histogram_cell_velocities', 'plot_densities']

# Load matplotlib style file
# plt.style.use('/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/plotstyle.mplstyle')

def reformat_for_animate(filepath):
    """
    A method to take the native parcels netcdf4 output file and reformat the particle position data to a matplotlib
    3D scatterplot-friendly format.

    :param filepath: path to the netCDF file containing particle trajectories WITHOUT THE .NC
    """
    particlefile = netCDF4.Dataset(str(filepath + ".nc"))

    num_frames = particlefile.variables["lon"].shape[1]

    masked = np.ma.is_masked(particlefile.variables["lon"][:])

    if masked:
        lons = particlefile.variables["lon"][:].data[:, 0]
        lats = particlefile.variables["lat"][:].data[:, 0]
        deps = particlefile.variables["z"][:].data[:, 0]
        for f in tqdm(range(num_frames-1)):
            lons = np.vstack((lons, particlefile.variables["lon"][:].data[:, f + 1]))
            lats = np.vstack((lats, particlefile.variables["lat"][:].data[:, f + 1]))
            deps = np.vstack((deps, particlefile.variables["z"][:].data[:, f + 1]))
    else:
        lons = particlefile.variables["lon"][:][:, 0]
        lats = particlefile.variables["lat"][:][:, 0]
        deps = particlefile.variables["z"][:][:, 0]
        for f in tqdm(range(num_frames-1)):
            lons = np.vstack((lons, particlefile.variables["lon"][:][:, f + 1]))
            lats = np.vstack((lats, particlefile.variables["lat"][:][:, f + 1]))
            deps = np.vstack((deps, particlefile.variables["z"][:][:, f + 1]))

    np.save(filepath + '_lons.npy', lons)
    np.save(filepath + '_lats.npy', lats)
    np.save(filepath + '_deps.npy', deps)

    particlefile.close()

def histogram_cell_velocities(filepaths, n_bins, saveplot=None):
    """
    A method to check the velocities in each cell of the fluid simulation and return a histogram of them, to help
    determining the Courant number.

    :param filepaths: String or list of strings representing the path(s) to the file(s).

    :param n_bins: Int representing number of bins to pass to np.histogram()

    :param plot: Boolean indicating whether to produce the histogram plot or just compute the H-array and bin edges.

    :return: H : array containing the values of the histogram. See density and weights for a description of the
    possible semantics.

    :return: bin_edges : array of dtype float containing the bin edges.
    """

    count = 0

    paths = sorted(glob(str(filepaths))) if not isinstance(filepaths, list) else filepaths
    if len(paths) == 0:
        notfound_paths = filepaths
        raise IOError("FieldSet files not found: %s" % str(notfound_paths))
    with tqdm(total=len(paths)) as pbar:
        for fp in paths:
            if not os.path.exists(fp):
                raise IOError("FieldSet file not found: %s" % str(fp))

            nc = netCDF4.Dataset(fp)

            vels = np.power(np.power(nc.variables["u"][:], 2) + np.power(nc.variables["v"][:], 2) + np.power(nc.variables["w"][:], 2), 0.5)
            if count==0:
                vmax = np.max(vels)

            nc.close()

            if count == 0:
                H, bin_edges = np.histogram(vels, bins=n_bins, range=(0, vmax))
            else:
                H += np.histogram(vels, bins=bin_edges)[0]

            count += 1
            pbar.update(1)

    if saveplot is not None:
        width = (bin_edges[1] - bin_edges[0])

        fig = plt.figure()

        # plot pdf
        plt_pdf = fig.add_subplot(1, 2, 1)
        plt_pdf.bar(bin_edges[1:], H, width=width)
        xlim = plt_pdf.get_xlim()

        # plot cdf
        plt_cdf = fig.add_subplot(1, 2, 2)
        plt_cdf.set_xlim(xlim)
        plt_cdf.set_ylim(0., 1.)
        n = sum(H)
        x = bin_edges
        y = np.append(np.zeros(1), np.cumsum(H) / n)
        plt_cdf.plot(x, y)

        # compute cutoffs and plot as vertical red lines on both plots
        cutoff_50 = np.argmax(y >= 0.5)
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
        plt.ylabel("Count", fontsize=18)
        plt_pdf.set_title("Histogram", fontsize=20)
        plt_pdf.set_xlabel("Velocity Magnitudes (m/s)", fontsize=18)
        plt_pdf.set_ylabel("Count", fontsize=18)

        plt_cdf.set_title("CDF", fontsize=20)
        plt_cdf.set_xlabel("Velocity Magnitudes (m/s)", fontsize=18)
        plt_cdf.set_ylabel("Fraction of Data", fontsize=18)

        plt.savefig(saveplot)

    return H, bin_edges

def plot_densities(filepath, savepath=None):
    """
    This method uses the mayavi library to produce 3D density plots of a particle simulation.
    :param filepath: string representing the path to netCDF file containing particle position data (EXCLUDING THE .nc)
    :param savepath: string representing where to save the density plots.
    """
    density = np.load(str(filepath + "_density.npy"))
    min = density.min()
    max = density.max()
    xmin, ymin, zmin = (0., 0., 0.)
    xmax, ymax, zmax = (720., 720., 360.)
    xi, yi, zi = density.shape[0:3]
    xi, yi, zi = np.mgrid[xmin:xmax:xi*1j, ymin:ymax:yi*1j, zmin:zmax:zi*1j]

    for t in tqdm(range(density.shape[3])):
        figure = mlab.figure('DensityPlot', bgcolor=(0., 0., 0.), size=(720, 720))
        grid = mlab.pipeline.scalar_field(xi, yi, zi, density[:, :, :, t])
        mlab.pipeline.volume(grid, vmin=min + .2 * (max - min), vmax=min + .8 * (max - min))
        mlab.axes()
        mlab.view(azimuth=45, elevation=235, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
         # mlab.view(azimuth=-45, elevation=315, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
         # mlab.show()
        if savepath is not None:
            mlab.savefig(savepath + "%03.0f" % t + "x.png")
        mlab.clf()


if __name__ == "__main__":
    filepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/trajectories_10000p_30s_0.01dt_0.1sdt_initunif_dead"
    savepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/dead/density_10000p_30s_0.01dt_0.1sdt_initunif_dead/"
    plot_densities(filepath, savepath)
    # save_plot_dir = "/home/alexander/Documents/QMEE/LSR/fig/velocity_dist.png"
    # H, bin_edges = histogram_cell_velocities("/media/alexander/AKC Passport 2TB/Maarten/sim022/F*.nc.022", 100, saveplot=save_plot_dir)
    # np.save("/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/H.npy", H)
    # np.save("/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/bin_edges.npy", bin_edges)
