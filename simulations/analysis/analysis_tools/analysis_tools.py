import netCDF4
import numpy as np

all = ['reformat_for_animate']


def reformat_for_animate(filepath):
    """
    A method to take the native parcels netcdf4 output file and reformat the particle position data to a matplotlib
    3D scatterplot-friendly format.

    :param filepath: path to the netCDF file containing particle trajectories WITHOUT THE .NC
    """

    particlefile = netCDF4.Dataset(str(filepath + ".nc"))

    num_particles = particlefile.variables["lon"].shape[0]
    num_frames = particlefile.variables["lon"].shape[1]

    lons = [[particlefile.variables["lon"][p][f] for p in range(num_particles)] for f in range(num_frames)]
    lats = [[particlefile.variables["lat"][p][f] for p in range(num_particles)] for f in range(num_frames)]
    deps = [[particlefile.variables["z"][p][f] for p in range(num_particles)] for f in range(num_frames)]

    # Prefer the ugly explicit loop method for tqdm.
    # lons = [[particlefile.variables["lon"][p][0] for p in range(num_particles)]]
    # lats = [[particlefile.variables["lat"][p][0] for p in range(num_particles)]]
    # deps = [[particlefile.variables["z"][p][0] for p in range(num_particles)]]
    # for f in tqdm(range(num_frames-1)):
    #     lons_f = []
    #     lats_f = []
    #     deps_f = []
    #     for p in range(num_particles):
    #         lons_f.append(particlefile.variables["lon"][p][f+1])
    #         lats_f.append(particlefile.variables["lat"][p][f+1])
    #         deps_f.append(particlefile.variables["z"][p][f+1])
    #     lons.append(lons_f)
    #     lats.append(lats_f)
    #     deps.append(deps_f)

    particlefile.close()

    np.save(filepath + '_lons.npy', lons)
    np.save(filepath + '_lats.npy', lats)
    np.save(filepath + '_deps.npy', deps)