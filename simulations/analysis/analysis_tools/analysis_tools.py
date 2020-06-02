import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
import vg
from scipy import stats
from mayavi import mlab
from math import ceil, floor
from simulations.util.util import cart2spher
# mlab.options.backend = 'envisage'

import sys
# sys.path.append('/home/alexander/Documents/surface_mixing/Analysis/')
# from ana_objects import ParticleData

all = ['reformat_for_animate', 'reformat_for_voronoi', 'compute_vertical_distance_travelled',
       'compute_particlewise_Veff', 'extract_particlewise_epsilon',
       'histogram_cell_velocities', 'plot_densities', 'plot_voro_concs', 'plot_polar_angles',
       'plot_polar_angles_superimposed', 'plot_trajectories', 'plot_particlewise_angles',
       'plot_particlewise_velocities', 'plot_particlewise_vorticities']

# Load matplotlib style file
# plt.style.use('/home/alexander/Documents/turbulence-patchiness-sims/simulations/analysis/analysis_tools/plotstyle.mplstyle')


def reformat_for_animate(filepath):
    """
    A method to take the native parcels netcdf4 output file and reformat the particle position data to a matplotlib
    3D scatterplot-friendly format.

    :param filepath: path to the netCDF file containing particle trajectories
    """
    particlefile = netCDF4.Dataset(str(filepath))

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

    filedir = os.path.dirname(filepath)
    np.save(os.path.join(filedir, "lons.npy"), lons)  #np.save(filepath + '_lons.npy', lons)
    np.save(os.path.join(filedir, "lats.npy"), lats)  #np.save(filepath + '_lats.npy', lats)
    np.save(os.path.join(filedir, "deps.npy"), deps)  #np.save(filepath + '_deps.npy', deps)

    particlefile.close()


def reformat_for_voronoi(filedir, timesteps):
    """
    A method to take the lon, lat, dep .npy files containing particle position data from a simulations and reformat for
    and reformat to a text file(s) of the format required by Voronoi tesselation library voro++ at each given timestep.

    voro++ requires a .txt file as an input. Each row in this file takes the form:

        <Numerical ID label> <x coordinate> <y coordinate> <z coordinate>

    representing a particle's position at the given time. The output is a .vol text file of the exact same format but
    with an additional column containing the Voronoi cell volume for each particle:

        <Numerical ID label> <x coordinate> <y coordinate> <z coordinate> <Voronoi cell volume>

    :param filedir: path to the directory containing particle lon.npy, lat.npy, dep.npy files.

    :param timesteps: List of timesteps for which to produce voro++ input files.
    """
    if not isinstance(timesteps, list):
        raise TypeError("timesteps must be provided as a list.")

    dx = 600/720
    print("dx = %f ENSURE THIS IS CORRECT FOR YOUR SIM AND YOUR CHOICE OF VOLUME UNITS" %dx)

    lons = np.load(os.path.join(filedir, "lons.npy"))
    lats = np.load(os.path.join(filedir, "lats.npy"))
    deps = np.load(os.path.join(filedir, "deps.npy"))

    n = lons.shape[1]

    for t in tqdm(timesteps):
        savepath = os.path.join(filedir, "vor/in/vor_mm_%03d.txt" % t)
        points = np.stack((np.arange(n), lons[t, :] * dx, lats[t, :] * dx, deps[t, :] * dx), axis=1)
        np.savetxt(savepath, points, fmt=["%d", "%.12f", "%.12f", "%.12f"])


def compute_vertical_distance_travelled(filepath):
    """
    A method to compute the vertical distance each particle has travelled during a simulation. Particles at the surface
    are not counted unless and until they move back below the surface.
    :param filepath: String representing the path to the directory containing simulation dep.npy files.
    :return: v_dist_array : A numpy array containing the vertical distance travelled for each particle.
    """
    scale_factor = 0.6/720
    print("WARNING: Scale factor set to %f. Ensure this is correct." %scale_factor)
    deps = np.load(os.path.join(filepath, "deps.npy"))
    nparticles = deps.shape[1]

    deps = np.clip(deps, a_min=None, a_max=360)  # reset depths above surface to the surface.
    d_v = np.sum(abs(deps[1:, :] - deps[:-1, :]), axis=0)

    d_v = d_v * scale_factor  # convert cells -> m

    return d_v


# def compute_particlewise_Veff(path_to_particle_trajectories, path_to_fluid_velocities):
#     """
#     A method to compute the effective velocity of each particle at each saved position during its trajectory in a
#     a completed simulation.
#     :param path_to_particle_trajectories: String representing the path to the .nc file containing particle trajectories.
#     :param path_to_fluid_velocities: String representing the path to the .nc file containing fluid velocities.
#     :return:
#     """
#     i = 0
#     raise NotImplementedError("compute_particlewise_Veff() method from analysis_tools not yet implemented.")


def extract_particlewise_epsilon(filepath, epsilon_csv_file="/media/alexander/AKC Passport 2TB/epsilon.csv"):
    """
    A method to extract the epsilon value at each particle's position during its trajectory in a completed simulation,
    given a .csv file containing the epsilon values at all depths and times.
    :param filepath: String representing the path to the .nc file containing particle trajectories.
    :param epsilon_csv_file: String representing the path to the .csv file containing epsilon values. The csv file
    should consist of 4 columns (time, zb, zc, epsilon), and be sorted first according to increasing time, then
    sub-sorted (i.e. within time-values) according to increasing zb.
    Each row then represents the epsilon value at the depth range defined by 'zb' & 'zc' and at the time defined by
    'time':
        'time' represents the number of seconds since the start of the DNS simulation (INCLUDING SPIN-UP!!)
        'zb' represents the z-coordinate of the 'bottom' of the depth range.
        'zc' represents the z-coordinate of the centre of the depth range.
        'epsilon' represents the value of the turbulent dissipation rate at the depth range and time specified.
    :return:
    """

    # load epsilon values
    print("REMINDER: Please ensure epsilon_csv_file is sorted first by increasing time, then subsorted by increasing zb")
    epsilon_array = np.genfromtxt(epsilon_csv_file, delimiter=",")
    time_offset = 30.0  # epsilon csv file includes DNS spin-up timesteps, which the Parcels simulation does not.
    print("Time offset set to %.1f seconds. Ensure this is correct." % time_offset)
    time_offset_index = np.searchsorted(epsilon_array[:, 0], time_offset)

    epsilon_array = epsilon_array[time_offset_index:, :]  # remove rows corresponding to spin-up timesteps
    epsilon_array[:, 0] -= time_offset  # align timestamps with the Parcels simulation.
    epsilon_array[:, 3] = -epsilon_array[:, 3]  # make epsilon column positive
    epsilon_timestamps = np.unique(epsilon_array[:, 0])  # extract timestamps at which we have epsilon data
    dz = epsilon_array[1, 1] - epsilon_array[0, 1]

    # load particle depths and timestamps
    nc = netCDF4.Dataset(filepath)
    depths = nc.variables['z'][:] * dz  # dz converts cells -> metres
    particle_timestamps = nc.variables['time'][:]  # extract timestamps at which we have particle data
    nc.close()

    # match epsilon values to particle depths.
    eps = np.zeros((depths.shape[0], epsilon_timestamps.size))
    ti = 0
    for t in tqdm(epsilon_timestamps):
        epsilon_array_t = epsilon_array[np.isclose(t, epsilon_array[:, 0])]
        depths_t = depths[np.isclose(t, particle_timestamps)]

        indicies = np.searchsorted(epsilon_array_t[:, 1], depths_t) - 1
        eps[:, ti] = epsilon_array_t[indicies, 3]
        ti += 1
    np.save(os.path.join(os.path.dirname(filepath), "eps.npy"), eps)


def extract_voronoi_epsilon(filepath, epsilon_csv_file="/media/alexander/AKC Passport 2TB/epsilon.csv"):
    """
    A method to extract the epsilon value at each particle's position during its trajectory in a completed simulation,
    given a .csv file containing the epsilon values at all depths and times.
    :param filepath: String representing the path to the .npy file containing particle depths POST-VORONOI ANALYSIS.
    :param epsilon_csv_file: String representing the path to the .csv file containing epsilon values. The csv file
    should consist of 4 columns (time, zb, zc, epsilon), and be sorted first according to increasing time, then
    sub-sorted (i.e. within time-values) according to increasing zb.
    Each row then represents the epsilon value at the depth range defined by 'zb' & 'zc' and at the time defined by
    'time':
        'time' represents the number of seconds since the start of the DNS simulation (INCLUDING SPIN-UP!!)
        'zb' represents the z-coordinate of the 'bottom' of the depth range.
        'zc' represents the z-coordinate of the centre of the depth range.
        'epsilon' represents the value of the turbulent dissipation rate at the depth range and time specified.
    :return:
    """

    # load epsilon values
    print("REMINDER: Please ensure epsilon_csv_file is sorted first by increasing time, then subsorted by increasing zb")
    epsilon_array = np.genfromtxt(epsilon_csv_file, delimiter=",")
    time_offset = 30.0  # epsilon csv file includes DNS spin-up timesteps, which the Parcels simulation does not.
    print("Time offset set to %.1f seconds. Ensure this is correct." % time_offset)
    time_offset_index = np.searchsorted(epsilon_array[:, 0], time_offset)

    epsilon_array = epsilon_array[time_offset_index:, :]  # remove rows corresponding to spin-up timesteps
    epsilon_array[:, 0] -= time_offset  # align timestamps with the Parcels simulation.
    epsilon_array[:, 3] = -epsilon_array[:, 3]  # make epsilon column positive
    epsilon_timestamps = np.unique(epsilon_array[:, 0])  # extract timestamps at which we have epsilon data

    # load particle depths and timestamps
    depths = np.load(filepath) * 0.001  # converts mm -> metres
    particle_timestamps = np.arange(0, 61)

    # match epsilon values to particle depths.
    eps = np.zeros((depths.shape[0], epsilon_timestamps.size))
    ti = 0
    for t in tqdm(epsilon_timestamps):
        epsilon_array_t = epsilon_array[np.isclose(t, epsilon_array[:, 0])]
        depths_t = depths[:, ti]

        indicies = np.searchsorted(epsilon_array_t[:, 1], depths_t) - 1
        eps[:, ti] = epsilon_array_t[indicies, 3]
        ti += 1
    np.save(os.path.join(os.path.dirname(filepath), "eps_vor.npy"), eps)


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
        raise IOError("FieldSet files notplot_entropies found: %s" % str(notfound_paths))
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
    timestamps = np.linspace(0, 300, 31, dtype=int).tolist()
    lons = np.load(os.path.join(filepath, "lons.npy"))
    lats = np.load(os.path.join(filepath, "lats.npy"))
    deps = np.load(os.path.join(filepath, "deps.npy"))
    density = np.load(os.path.join(filepath, "density.npy"))
    density = density[:, :, 0:180, :]
    min = density.min()
    max = density.max()
    xmin, ymin, zmin = (0., 0., 0.)
    xmax, ymax, zmax = (720., 720., 360.)
    xi, yi, zi = density.shape[0:3]
    xi, yi, zi = np.mgrid[xmin:xmax:xi*1j, ymin:ymax:yi*1j, zmin:zmax:zi*1j]

    # for t in tqdm(range(density.shape[3])):
    #     figure = mlab.figure('DensityPlot', bgcolor=(1., 1., 1.), fgcolor=(0.,0.,0.), size=(720, 720))
    #     grid = mlab.pipeline.scalar_field(xi, yi, zi, density[:, :, :, t])
    #     #mlab.pipeline.volume(grid, vmin=min + .2 * (max - min), vmax=min + .8 * (max - min))
    #     # vol_lowconc = mlab.pipeline.volume(grid, vmin=0., vmax=min + .25 * (max - min), color=(1., 0., 0.))
    #     vol_highconc = mlab.pipeline.volume(grid, vmin=min + .75 * (max - min), vmax=max, color=(0., 0., 1.))
    #     mlab.axes()
    #     mlab.view(azimuth=45, elevation=235, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
    #      # mlab.view(azimuth=-45, elevation=315, distance=2500, focalpoint=(xmax/2., ymax/2., zmax/2.))
    #      # mlab.show()
    #     if savepath is not None:
    #         mlab.savefig(savepath + "%03.0f" % t + "dead.png")
    #     mlab.clf()

    for t in tqdm(range(density.shape[3])):
        x = lons[timestamps[t], ~np.isnan(lons[timestamps[t], :])]
        y = lats[timestamps[t], ~np.isnan(lats[timestamps[t], :])]
        z = deps[timestamps[t], ~np.isnan(deps[timestamps[t], :])]

        xyz = np.vstack([x, y, z])
        kde = stats.gaussian_kde(xyz)
        density = kde(xyz)
        min=density.min()
        max=density.max()
        f = 0.5
        f_cutoff = min + 0.5*(max-min)
        colors = np.zeros((len(x), 4)).astype(np.uint8)
        for i in range(x.size):
            colors[i, 0] = 255 * (density[i] < f_cutoff)
            colors[i, 2] = 255 * (density[i] > f_cutoff)
            colors[i, 3] = int(255 * 0.2) * (density[i] > f_cutoff) + int(255 * 0.99) * (density[i] < f_cutoff)

        # Plot scatter with mayavi
        figure = mlab.figure('DensityPlot', bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.), size=(720, 720))
        pts = mlab.points3d(x, y, z, density, colormap='blue-red', scale_mode='none', scale_factor=2)
        pts.module_manager.scalar_lut_manager.lut.table = colors
        mlab.axes()
        mlab.draw()
        mlab.savefig(savepath + "%03.0f" % t + ".png")
        mlab.clf()


def plot_voro_concs(filepath, savepath=None):
    """
    This method uses the mayavi library to produce 3D density plots of a particle simulation based on the reciprocal of
    of the volume of the voronoi cells.
    :param filepath: string representing the path to netCDF file containing particle position data (EXCLUDING THE .nc)
    :param savepath: string representing where to save the density plots.
    """
    timestamps = np.linspace(0, 300, 31, dtype=int).tolist()
    lons = np.load(os.path.join(filepath, "lons.npy"))[timestamps, :]
    lats = np.load(os.path.join(filepath, "lats.npy"))[timestamps, :]
    deps = np.load(os.path.join(filepath, "deps.npy"))[timestamps, :]
    vols = np.load(os.path.join(filepath, "vols.npy"))
    concs = np.reciprocal(vols)

    for t in [26]:#tqdm(range(31)):
        not_nans = ~np.isnan(lons[t, :])
        x = lons[t, not_nans]
        y = lats[t, not_nans]
        z = deps[t, not_nans]
        c = concs[not_nans, t]
        x = x[z>1]
        y = y[z>1]
        c = c[z>1]
        z = z[z>1]

        # Plot scatter with mayavi
        figure = mlab.figure('DensityPlot', bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.), size=(720, 720))
        pts = mlab.points3d(x, y, z, np.log10(c), colormap='blue-red', scale_mode='scalar', scale_factor=1., transparent=True)
        # pts.module_manager.scalar_lut_manager.lut.table = colors
        # pts.mlab_source.dataset.point_data.scalars = c

        # s = np.ones_like(x)
        # pts = mlab.quiver3d(x, y, z, s, s, s, scalars=c, mode="sphere", scale_factor=.5)
        # pts.glyph.color_mode = 'color_by_scalar'
        # pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

        mlab.axes()
        mlab.draw()
        mlab.savefig(savepath + "%03.0f" % t + ".png")
        # mlab.clf()


# def plot_entropies(filepath):
#     tload = [0, -1]
#     # time_origin=datetime.datetime(2000,1,5)
#     # Times = [(time_origin + datetime.timedelta(days=t * 5)).strftime("%Y-%m") for t in tload]
#
#     # def reduce_particleset():
#     #     # Load particle data
#     #     pdir = datadir + 'MixingEntropy/'  # Data directory
#     #     fname = 'surfaceparticles_y2000_m1_d5_simdays3650_pos'  # F
#     #
#     #     # load data
#     #     pdata = ParticleData.from_nc(pdir=pdir, fname=fname, Ngrids=40, tload=tload)
#     #     pdata.remove_nans()
#     #
#     #     # Get those particles that start and end in the chosen basin
#     #     r = np.load(outdir_paper + "EntropyMatrix/Entropy_Clusters.npy")
#     #
#     #     for i_basin in range(1, 6):  # loop over basins as defined in figure 3a)
#     #
#     #         print('--------------')
#     #         print('BASIN: ', i_basin)
#     #         print('--------------')
#     #
#     #         # define basin region
#     #         basin = np.array([1 if r[i] == i_basin else 0 for i in range(len(r))])
#     #
#     #         # constrain to particles that start in the respective basin
#     #         l = {0: basin}
#     #         basin_data = pdata.get_subset(l, 2.)
#     #
#     #         # select particles that are in the basin each subsequent year
#     #         for t in range(len(tload)):
#     #             l[t] = basin
#     #         basin_data = pdata.get_subset(l, 2.)
#     #
#     #         lons = basin_data.lons.filled(np.nan)
#     #         lats = basin_data.lats.filled(np.nan)
#     #         times = basin_data.times.filled(np.nan)
#     #         np.savez(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin), lons=lons, lats=lats,
#     #                  times=times)
#
#     pdata = ParticleData.from_nc(filepath, "", tload)
#
#
#     def compute_transfer_matrix():
#         # deg_labels is the choice of square binning
#
#         for i_basin in range(1, 6):
#
#             # load reduced particle data for each basin
#             pdata = np.load(outdir_paper + 'EntropyMatrix/Reduced_particles_' + str(i_basin) + '.npz', 'r')
#             lons = pdata['lons']
#             lats = pdata['lats']
#             times = pdata['times']
#             del pdata
#             pdata_ocean = ParticleData(lons=lons, lats=lats, times=times)
#
#             # Define labels according to initial position
#             transfer_matrix = {}
#             pdata_ocean.set_labels(deg_labels, 0)
#             l0 = pdata_ocean.label
#             N = len(np.unique(l0))
#
#             # get existing labels and translate them into labels 0, ...., N-1
#             unique, counts = np.unique(l0, return_counts=True)
#             py_labels = dict(list(zip(unique, list(range(N)))))
#             original_labels = dict(list(zip(list(range(N)), unique)))
#
#             # compute transfer matrix
#             for t in range(0, len(lons[0])):
#                 n = np.zeros((N, N))
#                 pdata_ocean.set_labels(deg_labels, t)
#                 l = pdata_ocean.label
#
#                 for j in range(len(l)):
#                     if l[j] in l0:  # restrict to the existing labels (at t=0)
#                         n[py_labels[l0[j]], py_labels[l[j]]] += 1
#
#                 transfer_matrix[t] = n
#
#             np.savez(outdir_paper + 'EntropyMatrix/n_matrix_deg' + str(int(deg_labels)) + '/n_matrix_' + str(i_basin),
#                      n=transfer_matrix, original_labels=original_labels)
#
#     def plot_spatial_entropy():
#         # function to get the spatial entropy
#
#         Lons_edges = np.linspace(-180, 180, int(360 / deg_labels) + 1)
#         Lats_edges = np.linspace(-90, 90, int(180 / deg_labels) + 1)
#         Lons_centered = np.array([(Lons_edges[i] + Lons_edges[i + 1]) / deg_labels for i in range(len(Lons_edges) - 1)])
#         Lats_centered = np.array([(Lats_edges[i] + Lats_edges[i + 1]) / deg_labels for i in range(len(Lats_edges) - 1)])
#
#         fig = plt.figure(figsize=(12, 8))
#         gs1 = gridspec.GridSpec(2, 2)
#         gs1.update(wspace=0.15, hspace=0.)
#
#         labels = ['a) ', 'b) ', 'c) ', 'd) ']
#
#         for t, k in zip([1, 3, 6, 10], list(range(4))):
#             T = Times[t]
#
#             S_loc = np.zeros(len(Lons_centered) * len(Lats_centered))  # final entropy field
#
#             for i_basin in range(1, 6):
#                 # load data
#                 data = np.load(outdir_paper + 'EntropyMatrix/n_matrix_deg' + str(int(deg_labels)) + '/n_matrix_' + str(
#                     i_basin) + '.npz', 'r')
#                 n_matrix = data['n'].tolist()
#                 original_labels = data['original_labels'].tolist()
#                 n = n_matrix[t]
#
#                 # row-normalize n
#                 for i in range(len(n)):
#                     s = np.sum(n[i, :])
#                     if s != 0:
#                         n[i, :] /= s
#                     else:
#                         n[i, :] = 0
#
#                 # column-normalize
#                 for i in range(len(n)):
#                     s = np.sum(n[:, i])
#                     if s != 0:
#                         n[:, i] /= s
#                     else:
#                         n[:, i] = 0
#
#                 # Compute entropy for each location
#                 S = {}
#                 for j in range(len(n)):
#                     s = 0
#                     for i in range(len(n)):
#                         if n[i, j] != 0:
#                             s -= n[i, j] * np.log(n[i, j])
#
#                     S[original_labels[j]] = s
#
#                 # maximum entropy
#                 N = len(np.unique(list(original_labels.keys())))
#                 maxS = np.log(N)
#
#                 for i in range(len(S_loc)):
#                     if i in list(S.keys()):
#                         S_loc[i] = S[i] / maxS
#
#             plt.subplot(gs1[k])
#
#             S_loc = S_loc.reshape((len(Lats_centered), len(Lons_centered)))
#             S_loc = np.roll(S_loc, int(180 / deg_labels))
#             m = Basemap(projection='robin', lon_0=0, resolution='c')
#             m.drawparallels([-60, -30, 0, 30, 60], labels=[True, False, False, True], color='w', linewidth=1.2, size=9)
#             m.drawmeridians([-150, -60, 0, 60, 150], labels=[False, False, False, True], color='w', linewidth=1.2,
#                             size=9)
#             m.drawcoastlines()
#             m.fillcontinents(color='lightgrey')
#
#             lon_bins_2d, lat_bins_2d = np.meshgrid(Lons_edges, Lats_edges)
#             xs, ys = m(lon_bins_2d, lat_bins_2d)
#             assert (np.max(S_loc) <= 1)
#             p = plt.pcolormesh(xs, ys, S_loc, cmap='magma', vmin=0, vmax=1, rasterized=True)
#             plt.title(labels[k] + str(T), size=12, y=1.01)
#
#         # color bar on the right
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.822, 0.35, 0.015, 0.4])
#         cbar = fig.colorbar(p, cax=cbar_ax)
#         cbar.ax.tick_params(labelsize=11)
#         cbar.set_label(r'$S/S_{max}$', size=12)
#         fig.savefig(outdir_paper + figure_title, dpi=300, bbox_inches='tight')
#
#     reduce_particleset()
#     compute_transfer_matrix()
#     plot_spatial_entropy()


def plot_polar_angles(filepath, savepath_hist=None, savepath_timeseries=None):
    timestamps = np.arange(0, 300, 10)
    nc = netCDF4.Dataset(filepath + ".nc")
    dir_x = nc.variables["dir_x"][:][:, timestamps]
    dir_y = nc.variables["dir_y"][:][:, timestamps]
    dir_z = nc.variables["dir_z"][:][:, timestamps]

    up = np.array([0., 0., 1])

    theta = np.zeros_like(dir_x)
    mean = np.zeros(timestamps.size)

    for t in tqdm(range(timestamps.size)):
        fig = plt.figure(figsize=(12, 9))
        for p in range(dir_x.shape[0]):
            orientation = np.array((dir_x[p, t], dir_y[p, t], dir_z[p, t]))
            theta[p, t] = vg.angle(up, orientation)

        n = theta[:, t].size
        # x = np.sort(theta[:, t].flatten())
        # y = np.array(range(n)) / float(n)
        #mean[t] = x[np.argmax(y >= .5)]
        mean[t] = np.mean(theta[:, t])
        # ylims = [400, 1000]
        # text_x = [0.1, 0.2]
        # text_y = [200, 500]
        plt_hist = fig.add_subplot(111)#(1, 2, i + 1)
        plt_hist.hist(theta[:, t], 100, range=(0., 180.))
        plt_hist.set_title("Histogram of Particle Polar Angle at time t=%2.f s" % t, fontsize=25)
        plt_hist.set_xlim(0, 180)
        plt_hist.set_ylim(0., 5000)
        plt_hist.set_xlabel("Mean Polar angle (deg)", fontsize=25)
        plt_hist.set_ylabel("Count", fontsize=25)
        plt_hist.axvline(mean[t], ymin=0., ymax=plt_hist.get_ylim()[1], color='red')
        plt_hist.text(137, 120, "mean =%2.1f" % mean[t], fontsize=25, color='red')
        for tick in plt_hist.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in plt_hist.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        fig.savefig(savepath_hist + str(int(t)))
        plt.close('plt_hist')
    # np.save(filepath + '_theta.npy', theta)

    fig = plt.figure(figsize=(12, 9))
    plt_means = fig.add_subplot(111)
    plt_means.set_title("Mean Particle Polar Angle over time", fontsize=25)
    plt_means.set_xlabel("Time (s)", fontsize=25)
    plt_means.set_ylabel("Polar angle (deg)", fontsize=25)
    plt_means.set_xlim(0, 30)
    plt_means.set_ylim(100, 0)
    plt_means.plot(np.arange(0, 30), mean, '-bo', linewidth=2, markersize=3)
    for tick in plt_means.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_means.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    fig.savefig(savepath_timeseries)
    plt.close('plt_means')


def plot_trajectories(sample, orientations, filepath, savepath=None):
    from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d.art3d as art3d
    step = 1

    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1

    nc = netCDF4.Dataset(filepath + ".nc")
    x = nc.variables["lon"][:][sample][:, timestamps]
    y = nc.variables["lat"][:][sample][:, timestamps]
    z = nc.variables["z"][:][sample][:, timestamps]
    if orientations:
        dir_x = nc.variables["dir_x"][:][sample][:, timestamps]
        dir_y = nc.variables["dir_y"][:][sample][:, timestamps]
        dir_z = nc.variables["dir_z"][:][sample][:, timestamps]
    nc.close()

    fig = plt.figure(figsize=(15, 15))

    ax = plt.axes(projection='3d')
    ax.set_title("Particle Trajectories", fontsize=20)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.set_zlabel("Depth", fontsize=20)

    m = 1
    for p in tqdm(range(len(sample))):
        ax.scatter(x[p, 0], y[p, 0], -z[p, 0], 'c', c='k', s=6.0)  # mark start points
        ax.plot(x[p, :], y[p, :], -z[p, :], 'o', markersize=4)
        if orientations:
            ax.quiver(x[p, ::m], y[p, ::m], -z[p, ::m],
                  dir_x[p, ::m], dir_y[p, ::m], -dir_z[p, ::m],
                  length=7, color='k')

    # ax.set_xlim3d(0, 720)
    # ax.set_ylim3d(0, 720)
    # ax.set_zlim3d(-180, 0)
    # plt.subplots_adjust(top=0.9)

    fig.savefig(savepath)


def plot_polar_angles_superimposed(filepaths, colours, labels, savepath_timeseries=None):

    def extract_polar_angles(filepath):
        timestamps = np.arange(0, 300, 10)
        nc = netCDF4.Dataset(filepath + ".nc")
        dir_x = nc.variables["dir_x"][:][:, timestamps]
        dir_y = nc.variables["dir_y"][:][:, timestamps]
        dir_z = nc.variables["dir_z"][:][:, timestamps]

        up = np.array([0., 0., 1])

        theta = np.zeros_like(dir_x)
        mean = np.zeros(timestamps.size)

        for t in range(timestamps.size):
            for p in range(dir_x.shape[0]):
                orientation = np.array((dir_x[p, t], dir_y[p, t], dir_z[p, t]))
                theta[p, t] = vg.angle(up, orientation)

            mean[t] = np.mean(theta[:, t])

        return mean

    timeseries = []
    with tqdm(total=len(filepaths)) as pbar:
        for file in filepaths:
            timeseries.append(extract_polar_angles(file))
            pbar.update(1)


    fig = plt.figure(figsize=(12, 9))
    plt_means = plt.subplot(111)
    plt_means.set_title("Mean Particle Polar Angles over time", fontsize=25)
    plt_means.set_xlabel("Time (s)", fontsize=25)
    plt_means.set_ylabel("Polar angle (deg)", fontsize=25)
    plt_means.set_xlim(0, 30)
    plt_means.set_ylim(100, 0)
    plt_means.plot(np.arange(0, 30), timeseries[0], '-o', color=colours[:, 0], linewidth=2, markersize=3, label=labels[0])
    plt_means.plot(np.arange(0, 30), timeseries[1], '-o', color=colours[:, 1], linewidth=2, markersize=3, label=labels[1])
    plt_means.plot(np.arange(0, 30), timeseries[2], '-o', color=colours[:, 2], linewidth=2, markersize=3, label=labels[2])
    plt_means.plot(np.arange(0, 30), timeseries[3], '-o', color=colours[:, 3], linewidth=2, markersize=3, label=labels[3])
    plt_means.plot(np.arange(0, 30), timeseries[4], '-o', color=colours[:, 4], linewidth=2, markersize=3, label=labels[4])
    plt_means.legend()
    for tick in plt_means.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_means.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    fig.savefig(savepath_timeseries)
    plt.close('plt_means')


def plot_trajectories(sample, orientations, filepath, savepath=None):
    from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d.art3d as art3d
    step = 1

    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1

    nc = netCDF4.Dataset(filepath + ".nc")
    x = nc.variables["lon"][:][sample][:, timestamps]
    y = nc.variables["lat"][:][sample][:, timestamps]
    z = nc.variables["z"][:][sample][:, timestamps]
    if orientations:
        dir_x = nc.variables["dir_x"][:][sample][:, timestamps]
        dir_y = nc.variables["dir_y"][:][sample][:, timestamps]
        dir_z = nc.variables["dir_z"][:][sample][:, timestamps]
    nc.close()

    fig = plt.figure(figsize=(15, 15))

    ax = plt.axes(projection='3d')
    ax.set_title("Particle Trajectories", fontsize=20)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.set_zlabel("Depth", fontsize=20)

    m = 1
    for p in tqdm(range(len(sample))):
        ax.scatter(x[p, 0], y[p, 0], -z[p, 0], 'c', c='k', s=6.0)  # mark start points
        ax.plot(x[p, :], y[p, :], -z[p, :], 'o', markersize=4)
        if orientations:
            ax.quiver(x[p, ::m], y[p, ::m], -z[p, ::m],
                  dir_x[p, ::m], dir_y[p, ::m], -dir_z[p, ::m],
                  length=7, color='k')

    # ax.set_xlim3d(0, 720)
    # ax.set_ylim3d(0, 720)
    # ax.set_zlim3d(-180, 0)
    # plt.subplots_adjust(top=0.9)

    fig.savefig(savepath)


def animate_directions(p, filepath, savepath=None):
    from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d.art3d as art3d
    from matplotlib.animation import FuncAnimation

    step = 1
    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1
    nc = netCDF4.Dataset(filepath + ".nc")
    dir_x = nc.variables["dir_x"][:][p, timestamps]
    dir_y = nc.variables["dir_y"][:][p, timestamps]
    dir_z = nc.variables["dir_z"][:][p, timestamps]
    nc.close()

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.set_tight_layout(True)

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

    # Create a sphere
    r = .5
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Particle %d Orientation" %p, fontsize=25)
    ax.set_xlabel("", fontsize=25)
    ax.set_ylabel("", fontsize=25)
    ax.set_zlabel("", fontsize=25)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    for t in timestamps:
        arrow = ax.quiver(r*dir_x[t], r*dir_y[t], r*dir_z[t], dir_x[t], dir_y[t], dir_z[t], length=1, color='k')
        fig.savefig(savepath + "%03d.png" % t)
        arrow.remove()


def plot_particlewise_angles(sample, filepath, savepath=None):

    step = 1
    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1

    nc = netCDF4.Dataset(filepath + ".nc")
    dir_x = nc.variables["dir_x"][:][sample][:, timestamps]
    dir_y = nc.variables["dir_y"][:][sample][:, timestamps]
    dir_z = nc.variables["dir_z"][:][sample][:, timestamps]
    nc.close()

    fig = plt.figure(figsize=(12, 14))
    plt_p_rs = fig.add_subplot(311)
    plt_p_rs.set_title("Sample of Particle Orientation magnitudes over time", fontsize=25)
    # plt_vort_mags.set_xlabel("Time (s)", fontsize=25)
    plt_p_rs.set_ylabel("r", fontsize=25)
    plt_p_rs.set_ylim(0, 1)
    plt_p_elevs = fig.add_subplot(312)
    plt_p_elevs.set_title("Sample of Particle Elevation Angles over time", fontsize=25)
    # plt_p_elevs.set_xlabel("Timestep", fontsize=25)
    plt_p_elevs.set_ylabel("Phi (deg)", fontsize=25)
    plt_p_elevs.set_ylim(180, 0)
    plt_p_azims = fig.add_subplot(313)
    plt_p_azims.set_title("Sample of Particle Azimuthal angles over time", fontsize=25)
    # plt_p_azims.set_xlabel("Time (s)", fontsize=25)
    plt_p_azims.set_ylabel("Theta (deg)", fontsize=25)
    plt_p_azims.set_ylim(0, 360)

    for p in tqdm(range(len(sample))):
        r_p, phi_p, theta_p = cart2spher(dir_x[p, :], dir_y[p, :], dir_z[p, :])
        plt_p_rs.plot(r_p, '.-', linewidth=2)
        plt_p_elevs.plot(phi_p * 57.2958, '.-', linewidth=2)
        plt_p_azims.plot(theta_p * 57.2958 + 180, '.-', linewidth=2)
    for tick in plt_p_rs.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_p_rs.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_p_elevs.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_p_elevs.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_p_azims.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_p_azims.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(savepath)


def plot_particlewise_velocities(sample, filepath, savepath=None):
    step = 1
    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1

    nc = netCDF4.Dataset(filepath + ".nc")
    u = nc.variables["u"][:][sample][:, timestamps]#[sample, timestamps]
    v = nc.variables["v"][:][sample][:, timestamps]
    w = nc.variables["w"][:][sample][:, timestamps]
    nc.close()

    fig = plt.figure(figsize=(14, 7))
    plt_vel_mags = fig.add_subplot(111)
    plt_vel_mags.set_title("Sample of Fluid Velocity magnitudes over time", fontsize=25)
    plt_vel_mags.set_xlabel("Timestep", fontsize=25)
    plt_vel_mags.set_ylabel("V (m/s)", fontsize=25)

    for p in tqdm(range(len(sample))):
        v_p = np.sqrt(np.power(u[p, :], 2) + np.power(v[p, :], 2) + np.power(w[p, :], 2))
        plt_vel_mags.plot(v_p / 1200, '.-', linewidth=2)

    for tick in plt_vel_mags.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vel_mags.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    # for tick in plt_vort_phis.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    # for tick in plt_vort_phis.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    # for tick in plt_vort_thetas.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    # for tick in plt_vort_thetas.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
    fig.savefig(savepath)


def plot_particlewise_vorticities(sample, filepath, savepath=None):

    step = 1
    timestamps = np.arange(0, 300, step)
    timestamps[0] = 1

    nc = netCDF4.Dataset(filepath + ".nc")
    vort_x = nc.variables["vort_x"][:][sample][:, timestamps]
    vort_y = nc.variables["vort_y"][:][sample][:, timestamps]
    vort_z = nc.variables["vort_z"][:][sample][:, timestamps]
    nc.close()

    fig = plt.figure(figsize=(14, 16))
    plt_vort_mags = fig.add_subplot(311)
    plt_vort_mags.set_title("Sample of Vorticity magnitudes over time", fontsize=25)
    # plt_vort_mags.set_xlabel("Time (s)", fontsize=25)
    plt_vort_mags.set_ylabel("|Omega|", fontsize=25)
    # plt_vort_mags.set_ylim(180, 0)
    plt_vort_phis = fig.add_subplot(312)
    plt_vort_phis.set_title("Sample of Vorticity elevation angles over time", fontsize=25)
    # plt_vort_phis.set_xlabel("Time (s)", fontsize=25)
    plt_vort_phis.set_ylabel("Phi (deg)", fontsize=25)
    plt_vort_phis.set_ylim(180, 0)
    plt_vort_thetas = fig.add_subplot(313)
    plt_vort_thetas.set_title("Sample of Vorticity azimuthal angles over time", fontsize=25)
    plt_vort_thetas.set_xlabel("Timestep", fontsize=25)
    plt_vort_thetas.set_ylabel("Theta (deg)", fontsize=25)
    plt_vort_thetas.set_ylim(0, 360)

    for p in tqdm(range(len(sample))):
        r_p, phi_p, theta_p = cart2spher(vort_x[p, :], vort_y[p, :], vort_z[p, :])
        plt_vort_mags.plot(r_p, '.-', linewidth=2)
        plt_vort_phis.plot(phi_p * 57.2958, '.-', linewidth=2)
        plt_vort_thetas.plot(theta_p * 57.2958 + 180, '.-', linewidth=2)


    for tick in plt_vort_mags.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vort_mags.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vort_phis.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vort_phis.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vort_thetas.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt_vort_thetas.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(savepath)


if __name__ == "__main__":
    # superimposed polar angle plots
    B1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/B_expt/100000p_30s_0.01dt_0.1sdt_1.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_1.0B_initunif_mot_1.0vswim"
    # B2 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim"
    # B3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/B_expt/100000p_30s_0.01dt_0.1sdt_3.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_3.0B_initunif_mot_1.0vswim"
    # B5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/B_expt/100000p_30s_0.01dt_0.1sdt_5.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_5.0B_initunif_mot_1.0vswim"
    # B7 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/B_expt/100000p_30s_0.01dt_0.1sdt_7.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_7.0B_initunif_mot_1.0vswim"
    # filepaths_B = [B1, B2, B3, B5, B7]
    # V1 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.1vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.1vswim"
    # V2 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.5vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_0.5vswim"
    # V3 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim"
    # V4 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.5vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.5vswim"
    # V5 = "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/vswim_expt/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_2.0vswim/trajectories_100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_2.0vswim"
    # filepaths_V = [V1, V2, V3, V4, V5]
    #
    # colours_B = np.zeros((3, 5))
    # colours_B[0, :] = np.linspace(0, 1, 5)
    # labels_B = ["B=1", "B=2", "B=3", "B=5", "B=7"]
    # colours_V = np.zeros((3, 5))
    # colours_V[1, :] = np.linspace(0, 1, 5)
    # labels_V = ["V=0.1", "V=0.5", "V=1.0", "V=1.5", "V=2.0"]
    #
    # plot_polar_angles_superimposed(filepaths_B, colours_B, labels_B, "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/comparison/theta/mean_polar_timeseries_Bvar.png")
    # plot_polar_angles_superimposed(filepaths_V, colours_V, labels_V, "/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/comparison/theta/mean_polar_timeseries_vswimvar.png")
    #
    # plot_voro_concs(
    #     '/media/alexander/DATA/Ubuntu/Maarten/outputs/sim022/initunif/mot/100000p_30s_0.01dt_0.1sdt_2.0B_initunif_mot_1.0vswim',
    # #     './')
    # files = ["/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/dead/100000p_0-60s_0.01dt_0.1sdt_initunif_dead/trajectories_100000p_0-60s_0.01dt_0.1sdt_initunif_dead.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_10um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_100um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_500um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_1.0B_1000um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_1.0B_1000um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_10um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_100um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_500um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_3.0B_1000um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_3.0B_1000um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_10um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_100um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_500um_initunif_mot.nc",
    #          "/media/alexander/DATA/Ubuntu/Maarten/outputs/results123/initunif/mot/100000p_0-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot/trajectories_100000p_0-60s_0.01dt_0.1sdt_5.0B_1000um_initunif_mot.nc"]
    # timesteps = list(np.arange(0, 601, 10))
    # for file in files:
    #     # reformat_for_voronoi(os.path.dirname(file), timesteps)
    #     extract_voronoi_epsilon(filepath=os.path.join(os.path.dirname(file), "vols_d.npy"),
    #                                 epsilon_csv_file='/media/alexander/AKC Passport 2TB/epsilon.csv')
