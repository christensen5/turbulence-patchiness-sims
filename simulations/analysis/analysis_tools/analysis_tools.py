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
        plt.subplot(1, 2, 1)
        plt.bar(bin_edges[1:], H, width=width)
        plt.title("Histogram", fontsize=20)
        plt.xlabel("Velocity Magnitudes (m/s)", fontsize=18)
        plt.ylabel("Count", fontsize=18)

        plt.subplot(1, 2, 2)
        n = sum(H)
        x = bin_edges
        y = np.append(np.zeros(1), np.cumsum(H)/n)
        plt.title("CDF", fontsize=20)
        plt.xlim(0., x[-1])
        plt.ylim(0., 1.)
        plt.xlabel("Velocity Magnitudes (m/s)", fontsize=18)
        plt.ylabel("Fraction of Data", fontsize=18)
        plt.plot(x, y)
        cutoff_95 = np.argmax(y >= 0.95)
        cutoff_99 = np.argmax(y >= 0.99)
        plt.vlines([x[cutoff_95], x[cutoff_99]], ymin=0., ymax=1., colors=['r'])
        plt.text(x[cutoff_95], 0.05, "x=%0.2f" % x[cutoff_95], fontsize=18)
        plt.text(x[cutoff_99], 0.05, "x=%0.2f" % x[cutoff_99], fontsize=18)

        plt.savefig(saveplot)

    return H, bin_edges


def plot_densities(filepath, timestamps, scale, savepath=None):
    """
    This method uses the mayavi library to produce 3D density plots of a particle simulation.
    :param filepath: string representing the path to netCDF file containing particle position data (EXCLUDING THE .nc)
    :param timestamps: float, list of floats, or string "firstlast" representing the timestamps for which plots are
    desired.
    :param scale: float determining the size of plotted points.
    :param savepath: string representing where to save the density plots.
    :return:
    """
    def calc_kde(data):
        return kde(data.T)

    # verify timestamp format
    if isinstance(timestamps, int) or isinstance(timestamps, float):
        timestamps = [timestamps]

    # load data
    lons = np.load(str(filepath + "_lons.npy"))
    lats = np.load(str(filepath + "_lats.npy"))
    deps = np.load(str(filepath + "_deps.npy"))

    if timestamps == "firstlast":
        timestamps = [0, lons.shape[0]-1]

    # compute and plot densities
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = lons.max(), lats.max(), deps.max()
    for t in tqdm(range(len(timestamps))):
        x = lons[timestamps[t], :]
        y = lats[timestamps[t], :]
        z = deps[timestamps[t], :]
        xyz = np.vstack([x, y, z])
        kde = stats.gaussian_kde(xyz)

        # # pointwise density
        # density = kde(xyz)
        # figure = mlab.figure('DensityPlot', bgcolor=(0., 0., 0.))
        # pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=scale)

        # whole grid density
        # xmin, ymin, zmin = 0, 0, 0
        # xmax, ymax, zmax = x.max(), y.max(), z.max()
        xi, yi, zi = np.mgrid[xmin:xmax:60j, ymin:ymax:60j, zmin:zmax:30j]
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        density = kde(coords).reshape(xi.shape)
        figure = mlab.figure('DensityPlot', bgcolor=(0., 0., 0.), size=(720, 720))
        grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
        min = density.min()
        max = density.max()
        mlab.pipeline.volume(grid, vmin=min + .2 * (max - min), vmax=min + .8 * (max - min))
        mlab.axes()
        mlab.view(azimuth=45, elevation=235, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
        # mlab.view(azimuth=-45, elevation=315, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
        # mlab.show()
        if savepath is not None:
            mlab.savefig(savepath + "%03.0f" % t + "x.png")
        mlab.clf()


if __name__ == "__main__":
    filepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initblob/mot/trajectories_10000p_30s_0.01dt_0.05sdt_initblob_mot"
    savepath = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initblob/mot/density_10000p_30s_0.01dt_0.05sdt_initblob_mot/"
    #timestamps = np.linspace(0, 3000, 301, dtype=int).tolist()
    timestamps = np.linspace(0, 600, 31, dtype=int).tolist()
    plot_densities(filepath, timestamps, 1, savepath)
    # save_plot_dir = "/home/alexander/Documents/QMEE/LSR/fig/velocity_dist.png"
    # H, bin_edges = histogram_cell_velocities("/media/alexander/AKC Passport 2TB/Maarten/sim022/F*.nc.022", 100, saveplot=save_plot_dir)
    # np.save("/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/H.npy", H)
    # np.save("/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/bin_edges.npy", bin_edges)