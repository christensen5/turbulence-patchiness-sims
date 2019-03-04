__all__ = ['DeleteParticle', 'TopBottomBoundary', 'periodicBC', 'AdvectionRK4_3D_withTemp']

def DeleteParticle(particle, fieldset, time):  # delete particles who run out of bounds.
    print("Particle %d deleted at (%f, %f, %f)" % (particle.id, particle.lon, particle.lat, particle.depth))
    particle.delete()


def TopBottomBoundary(particle, fieldset, time):  # delete particles who run out of bounds.
    if particle.depth < 0:
        particle.depth = particle.diameter/2 if particle.diameter is not None else 0.1
    elif particle.depth > fieldset.U.grid.depth[-1]:
        print("Out-of-depth particle %d deleted at (%f, %f, %f)" % (particle.id, particle.lon, particle.lat, particle.depth))
        particle.delete()
    else:
        print("Particle %d escaped xy-periodic halo at (%f, %f, %f) and was deleted." % (particle.id, particle.lon, particle.lat, particle.depth))


def periodicBC(particle, fieldset, time):
    # longitudinal/X boundary
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west
    # latitudinal/Y boundary
    if particle.lat < fieldset.halo_south:
        particle.lat += fieldset.halo_north - fieldset.halo_south
    elif particle.lat > fieldset.halo_north:
        particle.lat -= fieldset.halo_north - fieldset.halo_south


def AdvectionRK4_3D_withTemp(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    Function needs to be converted to Kernel object before execution"""
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt
    particle.temp = fieldset.Temp[time + particle.dt, particle.depth, particle.lat, particle.lon]
