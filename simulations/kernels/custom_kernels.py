import math
import warnings

__all__ = ['DeleteParticle', 'TopBottomBoundary', 'periodicBC', 'AdvectionRK4_3D_withTemp', 'GyrotaxisEE_3D_withTemp',
           'GyrotaxisRK4_3D_withTemp']

warnings.simplefilter('once', UserWarning)


def DeleteParticle(particle, fieldset, time):  # delete particles who run out of bounds.
    print("Particle %d deleted at (%f, %f, %f), t=%f" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()


def TopBottomBoundary(particle, fieldset, time, margin=1):  # delete particles who run out of bounds.
    if particle.diameter is None or math.isnan(particle.diameter):
        warnings.warn("Particle %d diameter not specified. TopBottomBoundary margin reverting to default!" % particle.id)
        particle.diameter = margin

    if particle.depth < particle.diameter/2.:
        particle.depth = particle.diameter/2.
        print("Particle %d breached surface at (%f, %f, %f), t=%f  and was resubmerged." % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    elif particle.depth > fieldset.U.grid.depth[-1]:
        print("Out-of-depth particle %d deleted at (%f, %f, %f), t=%f." % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
        particle.delete()
    else:
        print("Particle %d escaped xy-periodic halo at (%f, %f, %f), t=%f and was deleted." % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
        particle.delete()


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

    # Update temp at new position.
    particle.temp = fieldset.Temp[time + particle.dt, particle.depth, particle.lat, particle.lon]


def GyrotaxisEE_3D_withTemp(particle, fieldset, time):
    """Gyrotactic alignment of particles and consequent advection using Euler Forward integration. Function needs to
    be converted to Kernel object before execution."""
    (u, v, w) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    (vort_x, vort_y, vort_z) = (fieldset.vort_X[time, particle.depth, particle.lat, particle.lon],
                                fieldset.vort_Y[time, particle.depth, particle.lat, particle.lon],
                                fieldset.vort_Z[time, particle.depth, particle.lat, particle.lon])

    # Re-align the particle
    if not particle.B == 0.0:
        di = 0.5 * ((1 / particle.B * particle.dir_x * particle.dir_z) + (vort_y * particle.dir_z) - (
                vort_z * particle.dir_y))
        dj = 0.5 * ((1 / particle.B * particle.dir_y * particle.dir_z) + (vort_z * particle.dir_x) - (
                vort_x * particle.dir_z))
        dk = 0.5 * ((1 / particle.B * (-1 + particle.dir_z * particle.dir_z)) + (vort_x * particle.dir_y) - (
                vort_y * particle.dir_x))
    else:
        di = 0.5 * ((vort_y * particle.dir_z) - (vort_z * particle.dir_y))
        dj = 0.5 * ((vort_z * particle.dir_x) - (vort_x * particle.dir_z))
        dk = 0.5 * ((vort_x * particle.dir_y) - (vort_y * particle.dir_x))

    newdir = [particle.dir_x + (di * particle.dt), particle.dir_y + (dj * particle.dt),
              particle.dir_z + (dk * particle.dt)]
    newnorm = (((newdir[0] ** 2) + (newdir[1] ** 2) + (newdir[2] ** 2)) ** 0.5)

    # alignment vector must be unit-length
    particle.dir_x = newdir[0] / newnorm
    particle.dir_y = newdir[1] / newnorm
    particle.dir_z = newdir[2] / newnorm

    # Update position
    particle.lon += (u * particle.dt) + (particle.dir_x * particle.v_swim * particle.dt)
    particle.lat += (v * particle.dt) + (particle.dir_y * particle.v_swim * particle.dt)
    particle.depth += (w * particle.dt) + (particle.dir_z * particle.v_swim * particle.dt)

    # Update temp
    particle.temp = fieldset.Temp[time + particle.dt, particle.depth, particle.lat, particle.lon]


def GyrotaxisRK4_3D_withTemp(particle,fieldset, time):
    """Gyrotactic alignment of particles and consequent advection using fourth-order Runge-Kutta integration,
    including vertical velocity. Function needs to be converted to Kernel object before execution"""
    # Compute k1
    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
    (vort_x1, vort_y1, vort_z1) = (fieldset.vort_X[time, particle.depth, particle.lat, particle.lon],
                                   fieldset.vort_Y[time, particle.depth, particle.lat, particle.lon],
                                   fieldset.vort_Z[time, particle.depth, particle.lat, particle.lon])
    dir_x1 = particle.dir_x
    dir_y1 = particle.dir_y
    dir_z1 = particle.dir_z

    k1_lon = (u1 + particle.v_swim * dir_x1) * particle.dt
    k1_lat = (v1 + particle.v_swim * dir_y1) * particle.dt
    k1_dep = (w1 + particle.v_swim * dir_z1) * particle.dt
    
    # Compute k2
    lon1 = particle.lon + k1_lon / 2.
    lat1 = particle.lat + k1_lat / 2.
    dep1 = particle.depth + k1_dep / 2.
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
    # New vort not needed unless we switch to computing p(t_n+1,x_n+1) using "midpoint" rather than Euler.
    # (vort_x2, vort_y2, vort_z2) = (fieldset.vort_X[time, dep1, lat1, lon1],
    #                                    fieldset.vort_Y[time, dep1, lat1, lon1],
    #                                    fieldset.vort_Z[time, dep1, lat1, lon1])
    if not particle.B == 0.0:
        di = 0.5 * (((1/particle.B) * (dir_x1*dir_z1)) + (vort_y1*dir_z1) - (vort_z1*dir_y1))
        dj = 0.5 * (((1/particle.B) * (dir_y1*dir_z1)) + (vort_z1*dir_x1) - (vort_x1*dir_z1))
        dk = 0.5 * (((1/particle.B) * (-1 + dir_z1*dir_z1)) + (vort_x1*dir_y1) - (vort_y1*dir_x1))
    else:
        di = 0.5 * ((vort_y1 * dir_z1) - (vort_z1 * dir_y1))
        dj = 0.5 * ((vort_z1 * dir_x1) - (vort_x1 * dir_z1))
        dk = 0.5 * ((vort_x1 * dir_y1) - (vort_y1 * dir_x1))
    newdir1 = [dir_x1 + (di * particle.dt * 0.5), dir_y1 + (dj * particle.dt * 0.5),
              dir_z1 + (dk * particle.dt * 0.5)]  # dt * 0.5 because we're at t + dt/2
    newnorm1 = (((newdir1[0] ** 2) + (newdir1[1] ** 2) + (newdir1[2] ** 2)) ** 0.5)

    dir_x2 = newdir1[0] / newnorm1
    dir_y2 = newdir1[1] / newnorm1
    dir_z2 = newdir1[2] / newnorm1

    k2_lon = (u2 + particle.v_swim * dir_x2) * particle.dt
    k2_lat = (v2 + particle.v_swim * dir_y2) * particle.dt
    k2_dep = (w2 + particle.v_swim * dir_z2) * particle.dt

    # Compute k3
    lon2 = particle.lon + k2_lon / 2.
    lat2 = particle.lat + k2_lat / 2.
    dep2 = particle.depth + k2_dep / 2.
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
    # New vort not needed unless we switch to computing p(t_n+1,x_n+1) using "midpoint" rather than Euler.
    # (vort_x3, vort_y3, vort_z3) = (fieldset.vort_X[time + .5 * particle.dt, dep2, lat2, lon2],
    #                                    fieldset.vort_Y[time + .5 * particle.dt, dep2, lat2, lon2],
    #                                    fieldset.vort_Z[time + .5 * particle.dt, dep2, lat2, lon2])
    # d_ijk remain the same for k_234 under Euler scheme for computing p(t_n+1,x_n+1). So don't
    # recompute unless we switch to "midpoint" scheme (in which case change below to dir_x2, vort_x2 etc...).
    # if not particle.B == 0.0:
    #     di = 0.5 * (((1/particle.B) * (dir_x1*dir_z1)) + (vort_y1*dir_z1) - (vort_z1*dir_y1))
    #     dj = 0.5 * (((1/particle.B) * (dir_y1*dir_z1)) + (vort_z1*dir_x1) - (vort_x1*dir_z1))
    #     dk = 0.5 * (((1/particle.B) * (-1 + dir_z1*dir_z1)) + (vort_x1*dir_y1) - (vort_y1*dir_x1))
    # else:
    #     di = 0.5 * ((vort_y1*dir_z1) - (vort_z1*dir_y1))
    #     dj = 0.5 * ((vort_z1*dir_x1) - (vort_x1*dir_z1))
    #     dk = 0.5 * ((vort_x1*dir_y1) - (vort_y1*dir_x1))
    # newdir & newnorm (and hence dir_xyz) remain the same for k_23 as well. Again must rewrite below if we switch to "midpoint".
    # newdir2 = [dir_x1 + (di * particle.dt * 0.5), dir_y1 + (dj * particle.dt * 0.5), dir_z1 + (dk * particle.dt * 0.5)] # dt x 0.5 because we're at t + dt/2
    # newnorm2 = (((newdir2[0] ** 2) + (newdir2[1] ** 2) + (newdir2[2] ** 2)) ** 0.5)

    dir_x3 = dir_x2
    dir_y3 = dir_y2
    dir_z3 = dir_z2

    k3_lon = (u3 + particle.v_swim * dir_x3) * particle.dt
    k3_lat = (v3 + particle.v_swim * dir_y3) * particle.dt
    k3_dep = (w3 + particle.v_swim * dir_z3) * particle.dt

    # Compute k4
    lon3 = particle.lon + k3_lon
    lat3 = particle.lat + k3_lat
    dep3 = particle.depth + k3_dep
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
    # New vort not needed unless we switch to computing p(t_n+1,x_n+1) using "midpoint" rather than Euler.
    # (vort_x4, vort_y4, vort_z4) = (fieldset.vort_X[time + particle.dt, dep3, lat3, lon3],
    #                                fieldset.vort_Y[time + particle.dt, dep3, lat3, lon3],
    #                                fieldset.vort_Z[time + particle.dt, dep3, lat3, lon3])
    # d_ijk remain the same for k_234 under Euler scheme for computing p(t_n+1,x_n+1). So don't
    # recompute unless we switch to "midpoint" scheme (in which case change below to dir_x3, vort_i3 etc...).
    # if not particle.B == 0.0:
    #     di = 0.5 * (((1/particle.B) * (dir_x1*dir_z1)) + (vort_y1*dir_z1) - (vort_z1*dir_y1))
    #     dj = 0.5 * (((1/particle.B) * (dir_y1*dir_z1)) + (vort_z1*dir_x1) - (vort_x1*dir_z1))
    #     dk = 0.5 * (((1/particle.B) * (-1 + dir_z1*dir_z1)) + (vort_x1*dir_y1) - (vort_y1*dir_x1))
    # else:
    #     di = 0.5 * ((vort_y1*dir_z1) - (vort_z1*dir_y1))
    #     dj = 0.5 * ((vort_z1*dir_x1) - (vort_x1*dir_z1))
    #     dk = 0.5 * ((vort_x1*dir_y1) - (vort_y1*dir_x1))
    # newdir & newnorm (and hence dir_xyz) are different for k4 since time increment is dt not dt/2. Still may need to
    # change if we switch to computing p(t_n+1,x_n+1) using "midpoint" rather than Euler.
    newdir3 = [dir_x1 + (di * particle.dt), dir_y1 + (dj * particle.dt), dir_z1 + (dk * particle.dt)]
    newnorm3 = (((newdir3[0] ** 2) + (newdir3[1] ** 2) + (newdir3[2] ** 2)) ** 0.5)

    dir_x4 = newdir3[0] / newnorm3
    dir_y4 = newdir3[1] / newnorm3
    dir_z4 = newdir3[2] / newnorm3

    k4_lon = (u4 + particle.v_swim * dir_x4) * particle.dt
    k4_lat = (v4 + particle.v_swim * dir_y4) * particle.dt
    k4_dep = (w4 + particle.v_swim * dir_z4) * particle.dt

    particle.lon += (k1_lon + 2*k2_lon + 2*k3_lon + k4_lon) / 6.
    particle.lat += (k1_lat + 2*k2_lat + 2*k3_lat + k4_lat) / 6.
    particle.depth += (k1_dep + 2*k2_dep + 2*k3_dep + k4_dep) / 6.

    # CHANGE IF WE SWITCH TO COMPUTING p(t_n+1, x_n+1) IN SOME OTHER WAY THAN EULER
    particle.dir_x = dir_x4
    particle.dir_y = dir_y4
    particle.dir_z = dir_z4

    # Update temp at new position.
    particle.temp = fieldset.Temp[time + particle.dt, particle.depth, particle.lat, particle.lon]
