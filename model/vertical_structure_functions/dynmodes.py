# -*- coding: utf-8 -*-
"""Ocean vertical dynamic modes exercise code for Mathematical Modelling of
Geophysical Fluids MPE2013 Workshop at African Institute for Mathematical Sciences.
Adopted from https://bitbucket.org/douglatornell/aims-workshop
by Doug Latornell.
"""

from __future__ import print_function
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg as la

def dynmodes(Nsq, depth, nmodes, normalize=True):
    """dynmodes(Nsq, depth, nmodes, normalize)
    Calculate the 1st nmodes ocean dynamic vertical modes
    given a profile of Brunt-Vaisala (buoyancy) frequencies squared.

    Based on
    http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
    by John Klinck, 1999.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int

    :arg normalize: whether the modal structures should be normalized. Default is True
    :type normalize: boolean

    :returns: :obj:`(wmodes, pmodes, rmodes), ce, (dz_w, dz_p)` (vertical velocity modes,
              horizontal velocity modes, vertical density modes), modal speeds,
              (box size of vertical velocity grid, box size of pressure grid)
    :rtype: tuple of :class:`numpy.ndarray`
    """
    if np.all(depth >= 0.):
        z = -depth
    else:
        z = depth

    nmodes = min((nmodes, len(z)-2))
    # 2nd derivative matrix plus boundary conditions
    d2dz2_w, dz_w = build_d2dz2_matrix_w(z)
    # N-squared diagonal matrix
    Nsq_mat = np.diag(Nsq)
    # Solve generalized eigenvalue problem for eigenvalues and vertical
    # velocity modes
    eigenvalues_w, wmodes = la.eigs(d2dz2_w, k=nmodes, M=Nsq_mat,
                                    which='SM')#, maxiter=10000*len(depth),
                                    #ncv=4*nmodes+1)
    eigenvalues_w, wmodes = clean_up_modes(eigenvalues_w, wmodes, nmodes)
    # Horizontal velocity modes
    d2dz2_p, dz_p = build_d2dz2_matrix_p(z, Nsq)
    eigenvalues_p, pmodes = la.eigs(d2dz2_p, k=nmodes,
                                    which='SM')#, maxiter=10000*len(depth),
                                    #ncv=4*nmodes+1)
    eigenvalues_p, pmodes = clean_up_modes(eigenvalues_p, pmodes, nmodes)
    nmodes = min(pmodes.shape[1], wmodes.shape[1])
    eigenvalues_p, eigenvalues_w, pmodes, wmodes = (
        eigenvalues_p[:nmodes], eigenvalues_w[:nmodes], pmodes[:, :nmodes].T, wmodes[:, :nmodes].T)
    
    # Modal speeds
    ce = 1 / np.sqrt(eigenvalues_p)
    print("Mode speeds do correspond: %s" % np.allclose(ce * np.sqrt(eigenvalues_w), 1.))
    
    # Normalze mode structures to satisfy \int_{-H}^0 \hat{p}^2 dz = 1
    # wmodes are normalised with the same facto
    if normalize:
        norm = normalize_mode(pmodes, dz_p)
        pmodes *= norm
        wmodes *= norm
    
    # unify sign, that pressure modes are alwas positive at the surface
    wmodes, pmodes = unify_sign(wmodes, pmodes)

    # Vertical desity modes
    rmodes = wmodes * Nsq

    return (wmodes, pmodes, rmodes), (ce, 1 / np.sqrt(eigenvalues_w)), (dz_w, dz_p)


def normalize_mode(mode, dz):
    """Normalize the modal function to satisfy \int_{-H}^0 \hat{p}^2 dz = 1 """
    factor = np.sqrt(np.abs(1. / ((mode ** 2.) * dz).sum(axis=1)))
    return factor[:, np.newaxis]

# stattdessen: Das Mittel der Mode ueber die Tiefe = 1



def unify_sign(wmodes, pmodes):
    """unify_sign(wmodes, pmodes)
    Changes sign of the mode that, at the surface,  either the first derivative is \
    positive (wmodes) or the mode itself is positive (pmode)

    returns tupel of modes (wmodes, pmodes) with corrected sign
    """
    sig_p = np.sign(pmodes[:, 0])
    sig_p[sig_p == 0.] = 1.
    sig_w = np.sign(wmodes[:, 0] - wmodes[:, 1])
    sig_w[sig_w == 0.] = 1.
    pmodes = sig_p[:, np.newaxis] * pmodes
    wmodes = sig_w[:, np.newaxis] * wmodes
    return wmodes, pmodes

def build_d2dz2_matrix_w(z):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for
    w-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    # Size (in [m]) of vertical coordinate grid steps
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)

    d0 = np.r_[-1., (1. / dz[:-1] + 1. / dz[1:]) / dz_mid, -1.]
    dm1 = np.r_[-1. / dz[:-1] / dz_mid, 0., 0.]
    d1 = np.r_[0., 0., -1. / dz[1:] / dz_mid]
    diags = np.vstack((d0, d1, dm1))

    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(z), len(z)))
#    pcolor(d2dz2.toarray()[:10, :10]); colorbar(); title('d2dz2')
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    return d2dz2, dz_mid

def build_d2dz2_matrix_p(z, Nsq):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for 
    p-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    
    Ndz = Nsq * dz_mid

    d0 = np.r_[1. / Ndz[1] / dz[0],
               (1. / Ndz[2:-1] + 1. / Ndz[1:-2]) / dz[1:-1],
               1. / Ndz[-2] / dz[-1]]
    d1 = np.r_[0., -1. / Ndz[1:-1] / dz[:-1]]
    dm1 = np.r_[-1. / Ndz[1:-1] / dz[1:], 0.]

    diags = np.vstack((d0, d1, dm1))
    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(z)-1, len(z)-1))
    return d2dz2, dz


def clean_up_modes(eigenvalues, wmodes, nmodes):
    """Exclude complex-valued and near-zero/negative eigenvalues and their
    modes. Sort the eigenvalues and mode by increasing eigenvalue magnitude,
    truncate the results to the number of modes that were requested,
    and convert the modes from complex to real numbers.

    :arg eigenvalues: Eigenvalues
    :type eigenvalues: :class:`numpy.ndarray`

    :arg wmodes: Modes
    :type wmodes: :class:`numpy.ndarray`

    :arg nmodes: Number of modes requested
    :type nmodes: int

    :returns: :obj:`(eigenvalues, wmodes)`
    :rtype: tuple of :class:`numpy.ndarray`
    """
    # Filter out complex-values and small/negative eigenvalues
    # and corresponding modes
    mask = (eigenvalues.imag == 0) & (eigenvalues >= 1e-10)
    eigenvalues = eigenvalues[mask]
    wmodes = wmodes[:, mask]

    # Sort eigenvalues and modes and truncate to number of modes requests
    index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[index[:nmodes]]
    wmodes = wmodes[:, index[:nmodes]]
    return eigenvalues.real, wmodes.real


def plot_modes(Nsq, depth, nmodes, wmodes, pmodes, rmodes):
    """Plot Brunt-Vaisala (buoyancy) frequency profile and 3 sets of modes
    (vertical velocity, horizontal velocity, and vertical density) in 4 panes.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg wmodes: Vertical velocity modes
    :type wmodes: :class:`numpy.ndarray`

    :arg pmodes: Horizontal velocity modes
    :type pmodes: :class:`numpy.ndarray`

    :arg rmodes: Vertical density modes
    :type rmodes: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    # Nsq
    ax.plot(Nsq, -depth)
    ax.ticklabel_format(style='sci', scilimits=(2, 2), axis='x')
    ax.set_ylabel('z')
    ax.set_xlabel('N^2')
    # modes
    mode_sets = [
        # (values, subplot number, x-axis title)
        (wmodes, 2, 'wmodes'),
        (pmodes, 3, 'pmodes'),
        (rmodes, 4, 'rmodes'),
    ]
    for mode_set in mode_sets:
        modes, subplot, title = mode_set
        ax = fig.add_subplot(2, 2, subplot)
        for i in range(nmodes):
            ax.plot(modes[i], -depth, label='mode {}'.format(i + 1))
        ax.ticklabel_format(style='sci', scilimits=(3, 3), axis='x')
        ax.set_ylabel('z')
        ax.set_xlabel(title)
        ax.legend(loc='best')


def read_density_profile(filename):
    """Return depth and density arrays read from filename.

    :arg filename: Name of density profile file.
    :type filename: string

    :returns: :obj:`(depth, density)` depths, densities
    :rtype: tuple of :class:`numpy.ndarray`
    """
    depth = []
    density = []
    with open(filename) as f:
        for line in interesting_lines(f):
            deep, rho = map(float, line.split())
            depth.append(deep)
            density.append(rho)
    return np.array(depth), np.array(density)


def interesting_lines(f):
    for line in f:
        if line and not line.startswith('#'):
            yield line


def density2Nsq(depth, density, rho0=1028):
    """Return the Brunt-Vaisala (buoyancy) frequency (Nsq) profile
    corresponding to the given density profile.
    The surface Nsq value is set to the value of the 1st calculated value
    below the surface.
    Also return the depths for which the Brunt-Vaisala (buoyancy) frequencies squared
    were calculated.

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg density: Densities in [kg/m^3]
    :type density: :class:`numpy.ndarray`

    :arg rho0: Reference density in [kg/m^3]; defaults to 1028
    :type rho0: number

    :returns: :obj:`(Nsq_depth, Nsq)` depths for which the Brunt-Vaisala
              (buoyancy) frequencies squared were calculated,
              Brunt-Vaisala (buoyancy) frequencies squared
    :rtype: tuple of :class:`numpy.ndarray`
    """
    grav_acc = 9.8  # m / s^2
    Nsq = np.zeros_like(density)
    Nsq[1:] = np.diff(density) * grav_acc / (np.diff(depth) * rho0)
    Nsq[0] = Nsq[1]
    Nsq[Nsq < 0] = 0
    Nsq_depth = np.zeros_like(depth)
    Nsq_depth[1:] = (depth[:depth.size - 1] + depth[1:]) / 2
    return Nsq_depth, Nsq
