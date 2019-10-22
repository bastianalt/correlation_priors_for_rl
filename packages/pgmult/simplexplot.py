import os
import tempfile

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import scipy.spatial as spa
import scipy.stats as sta
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import interp1d
from scipy.special import expit, logit
from sklearn.utils.extmath import softmax

from packages.utils.utils import xy2bc, bc2xy
from packages.pgmult.utils import stickBreaking, reverseStickBreaking


def jacobianDet(probabilities, axis=0, keepdims=False, short=False):
    """
    Computes the determinants of the Jacobian of the stick breaking transformation for a given collection of
    probability vectors.

    Parameters
    ----------
    probabilities: array of probability vectors (points on the simplex)
    axis: axis along the probability vectors are oriented
    keepdims: if true, the produced singleton dimension is kept
    short: if true, then it is assumed that the last (dependent) entry of each vector is already dropped

    Returns
    -------
    array containing the determinants computed along the specified dimension
    """
    # check arguments
    pi = np.array(probabilities)
    assert np.all(np.logical_and(pi >= 0, pi <= 1))
    assert np.ndim(pi) >= axis

    # temporarily permute axis
    pi = np.swapaxes(pi, 0, axis)

    # if complete probability vectors are provided
    if not short:
        assert np.allclose(np.sum(pi, axis=0), 1)
        pi = pi[0:-1, ...]

    # construct product terms for all probability vectors
    csum = pi.cumsum(axis=0)
    num = 1 - np.concatenate((np.zeros((1, *pi.shape[1:])), csum[0:-1, ...]))
    den = pi * (1 - csum)
    frac = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    # undo permutation
    frac = np.swapaxes(frac, 0, axis)

    # final product
    res = frac.prod(axis=axis, keepdims=keepdims)
    return res


def realsamples2simplex(samples, method='stick'):
    """
    Transforms samples from R^n to the simplex.

    Parameters
    ----------
    samples: [N x M] array representing N samples. M depends on the number of categories and the transformation method.

    method: defines the transformation from R^n to the simplex
        'stick':    logistic transformation + stick breaking
                    -->  M = #categories - 1

        'logistic': logistic transformation + renormalization
                    -->  M = #categories

        'softmax':  softmax transformation
                    -->  M = #categories

    Returns
    -------
    [N x K] array representing the corresponding points on the probability simplex, where K = #categories
    """
    method = method.lower()
    if method == 'stick':
        # logistic transformation
        x = expit(samples)

        # stick breaking
        return stickBreaking(x, axis=1)

    elif method == 'logistic':
        # logistic transformation
        x = expit(samples)

        # normalization
        return x / x.sum(axis=1, keepdims=True)

    elif method == 'softmax':
        # softmax transformation
        return softmax(samples)

    else:
        raise ValueError('unknown transformation method')


def gaussian2simplex(mean, cov, pi, short=False):
    """
    Maps a Gaussian density to the probability simplex via logistic + stick breaking transformation.

    Parameters
    ----------
    mean: 1D array of length M, representing the mean of the Gaussian distribution

    cov: 2D array of shape (M,M), representing the covariance matrix of the Gaussian distribution

    pi: [N x K] array containing the N vectors on the probability simplex where the density should be computed,
        K=M if short=True, K=M+1 if short=False

    short: if true it is assumed that the last (dependent) entry of each vector has already been dropped

    Returns
    -------
    vector of length N containing the density values
    """
    psi = logit(reverseStickBreaking(pi, axis=1, short=short))
    density = sta.multivariate_normal.pdf(psi, mean, cov) * jacobianDet(pi, axis=1, short=short)
    density = np.nan_to_num(density)
    return density


def plot(distribution, method='stick', short=False, lingrid=None, bins=100, subdiv=6, levels=100):
    """
    Visualizes a distribution on the 1D or 2D probability simplex. The distribution can be either provided
    parametrically as a Gaussian or in form of a collection of samples.

    Parameters
    ----------
    distribution:   Either an [N x M] array representing N samples from the distribution or a tuple (mean, cov)
        containing the parameters of the Gaussian. The dimensionality of the Gaussian / the value M depend on the number
        of categories and/or the specified transformation method, see 'gaussian2simplex' and 'realsamples2simplex'.

    method: Either 'stick', 'logistic', 'softmax' or 'direct'. If 'direct', the methods are assumed to be provided
        directly on the simplex. For the other options, see 'realsamples2simplex'. If the distribution is specified
        parametrically, 'stick' is assumed.

    short: Only relevant for method='direct'. If true it is assumed that the last (dependent) entry of each sample
        has already been dropped.

    lingrid: grid for plotting 1D densities computed from parametric Gaussians

    bins: number of bins for plotting 1D densities computed from samples

    subdiv: grid granularity for plotting 2D densities

    levels: value granularity for plotting 2D densities
    """
    method = method.lower()

    if isinstance(distribution, tuple):
        parametric = True
        mean, cov = map(lambda x: np.array(x), distribution)
        dim = mean.shape[0] + 1
        assert method == 'stick'
    else:
        parametric = False
        samples = np.array(distribution)

        # type of samples
        if method == 'direct':
            # samples are directly provided on probability simplex
            if short:
                p = np.c_[samples, 1 - samples.sum(axis=1)]
            else:
                p = samples
        else:
            # map samples to probability simplex
            p = realsamples2simplex(samples, method)
        dim = p.shape[1]

    # two categories
    if dim == 2:
        # if distribution is parametrized as Gaussian
        if parametric:
            # plotting grid
            if lingrid is None:
                lingrid = np.linspace(0, 1, 100)

            # corresponding points on two-dimensional simplex (last entry dropped)
            pshort = lingrid[:, None]

            # evaluate density on simplex points
            density = gaussian2simplex(mean, cov, pshort, short=True)

            # visualization
            plt.plot(lingrid, density)

        # otherwise (distribution given in form of samples)
        else:
            # visualization
            plt.hist(p[:, 0], bins=bins, range=(0, 1), density=True)

    # three categories
    elif dim == 3:
        # define simplex triangle
        corners = np.array([[0, 0], [1, 0], [0.5, np.sin(np.pi / 3)]])
        triangle = tri.Triangulation(*corners.T)

        # get simplex grid
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        grid = np.c_[trimesh.x, trimesh.y]

        # compute Delaunay triangulation
        D = spa.Delaunay(grid)

        # if distribution is parametrized as Gaussian
        if parametric:
            p = xy2bc(grid, corners)
            density = gaussian2simplex(mean, cov, p)
            visualizationMesh = trimesh

        # otherwise (distribution given in form of samples)
        else:
            # get xy-Cartesian coordinates of simplex points
            xy = bc2xy(p, corners)

            # compute histogram on the triangulation simplexes
            simplexInds = D.find_simplex(xy)
            density = np.bincount(simplexInds, minlength=D.nsimplex) / samples.shape[0]

            # compute center points of triangulation
            centerPoints = np.zeros((D.nsimplex, 2))
            for vInd, v in enumerate(D.vertices):
                centerPoints[vInd] = D.points[D.vertices[vInd]].mean(axis=0)

            # mesh for visualization, defined on the center points of the original triangulation
            visualizationMesh = tri.Triangulation(centerPoints[:, 0], centerPoints[:, 1])

        # visualization
        plt.tricontourf(visualizationMesh, density, levels)
        plt.text(*corners[0], '(1,0,0)', verticalalignment='top', horizontalalignment='right')
        plt.text(*corners[1], '(0,1,0)', verticalalignment='top', horizontalalignment='left')
        plt.text(*corners[2], '(0,0,1)', verticalalignment='bottom', horizontalalignment='center')
        plt.title(method, loc='right')

    else:
        raise ValueError('wrong input dimension')


def linearGaussianInterpolator(startDist, endDist, steps):
    """
    Generator to interpolate linearly between the parameters of two Gaussian distributions.

    Parameters
    ----------
    startDist: tuple (mean, cov) specifying the first distribution
    endDist: tuple (mean, cov) specifying the second distribution
    steps: number of interpolation steps

    Returns
    -------
    Sequence of interpolated parameters.
    """
    # extract parameters
    mean1, cov1 = startDist
    mean2, cov2 = endDist

    # create interpolator objects
    meanInt = interp1d([0, 1], np.c_[mean1, mean2])
    covInt = interp1d([0, 1], np.c_[cov1.ravel(), cov2.ravel()])

    # interpolation grid
    t = np.linspace(0, 1, steps)

    # return interpolated values
    for i in t:
        yield (meanInt(i), covInt(i).reshape(cov1.shape))


def exportDistributionSweep(distributions, fps=30, dpi=100, path=None):
    """
    Creates (and plays) a video of a sequence of distributions on the simplex.

    Parameters
    ----------
    distributions: an iterable containing the different distributions passed to the plot function
    fps: framerate of the video
    dpi: resolution of the video
    path: path to the video. If 'None', a temporary file is used.
    """
    # create temporary file
    if path is None:
        _, path = tempfile.mkstemp(suffix='.mp4')

    # create writer object and export frames to video
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure()
    with writer.saving(fig, path, dpi=dpi):
        for param in distributions:
            fig.clear()
            plot(param)
            writer.grab_frame()

    # show video
    os.system('open ' + path)


if __name__ == '__main__':
    # specify one-dimensional Gaussian distribution
    mean1d = [0]
    var = [[1]]

    # specify two-dimensional Gaussian distribution
    mean2d = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    # number of samples
    nSamples = int(1e6)

    # draw Gaussian samples
    X1d = np.random.multivariate_normal(mean1d, var, nSamples)
    X2d = np.random.multivariate_normal(mean2d, cov, nSamples)

    # Case 1: one-dimensional via sampling
    plt.figure(1), plot(X1d), plt.show()

    # Case 2: one-dimensional via inverse transform
    plt.figure(1), plot((mean1d, var)), plt.show()

    # Case 3: two-dimensional via sampling
    plt.figure(1), plot(X2d), plt.show()

    # Case 4: two-dimensional via inverse transform
    plt.figure(1), plot((mean2d, cov)), plt.show()

    # Parameter Sweep
    mean1, mean2 = np.array([-1, 0]), np.array([1, 0])
    cov1, cov2 = np.eye(2), np.eye(2)
    distributions = linearGaussianInterpolator((mean1, cov1), (mean2, cov2), 3)
    exportDistributionSweep(distributions)
