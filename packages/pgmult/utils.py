import numpy as np
import pypolyagamma
from scipy.special._ufuncs import expit, logit
from scipy.linalg import solve_triangular


def multivariate_multinomial(N_m, Pi):
    """
            Draws count data X given Pi.

           Parameters
           ----------
           N_m: M dimensional vector with total number of samples per M
           Pi: Probability matrix used for creating the data

           Returns
           -------
           X: [M x K] M dimensional number of counts per category


           """
    N_m = np.array(N_m)
    Pi = np.array(Pi)
    M = N_m.__len__()
    X = np.array([np.random.multinomial(N_m[m], Pi[m, :]) for m in range(M)])
    return X


def stick_breaking(Psi):
    """
    Calculates the stickbreaking construction of the gaussian variables Psi
    Parameters
    ----------
    Psi: [M x K-1] Gaussian Variables used for the stick breaking

    Returns
    -------
    Pi: [M x K] Probability matrix using the logistic function and stick breaking
    """
    Pi = np.zeros((np.shape(Psi)[0], np.shape(Psi)[1] + 1))
    Pi[:, 0] = expit(Psi[:, 0])
    Pi[:, 1:-1] = expit(Psi[:, 1:]) * np.cumprod(1 - expit(Psi[:, 0:-1]), axis=1)
    Pi[:, -1] = 1 - np.sum(Pi[:, 0:-1], axis=1)
    # Check for numerical instability
    if np.any(Pi[:, -1] < 0):
        Pi[Pi[:, -1] < 0, -1] = 0  # Set last weight to 0
        Pi /= np.sum(Pi, axis=1)[:, None]  # Normalize last weight to 0

    return Pi


def inv_stick_breaking(Pi):
    """
    Calculates the inverse stick breaking construction for the probability matrix Pi

    Parameters
    ----------
    Pi: [M x K] Probability matrix using the logistic function and stick breaking


    Returns
    -------
    Psi: [M x K-1] Gaussian Variables used for the stick breaking
    """
    Psi = np.zeros((np.shape(Pi)[0], np.shape(Pi)[1] - 1))
    Psi[:, 0] = logit(Pi[:, 0])
    Psi[:, 1:] = logit(Pi[:, 1:-1] / (1 - np.cumsum(Pi[:, 0:-2], axis=1)))
    Psi[np.isnan(Psi)] = -np.Inf

    return Psi


def moment_matching(alpha, N_MC=False):
    """
    Finds the parameters for the Gaussian (mu, Sigma) by matching the moments
    of a dirichlet distribution with parameters alpha
    Parameters
    ----------
    alpha: [MxK] parameters of the M dirichlet distributions
    N_MC: Number of Monte Carlo samples used for estimating the expectations

    Returns
    -------
    mu: [MxK-1] M mean vectors of the Gaussians
    Sigma: [MxM] Correlation matrix of the M histograms

    """
    M, K = np.shape(alpha)
    if isinstance(N_MC, bool):
        N_MC = int(1e5)

    Pi_samples = np.array([np.random.dirichlet(alpha[m, :], N_MC) for m in range(M)]).swapaxes(0, 1)
    Psi_samples = np.array([inv_stick_breaking(Pi_samples[n, :, :]) for n in range(N_MC)])
    mu_k = np.zeros((M, K - 1))
    for k in range(K - 1):
        mu_k[:, k] = np.mean(Psi_samples[:, :, k], axis=0)

    Sigma = np.cov(Psi_samples.swapaxes(1, 2).reshape(-1, M).T)

    return mu_k, Sigma


def suff_stats_mult(X):
    """
    Transforms the matrix X into a matrix T which corresponds to the counting matrix presented in
    https://arxiv.org/abs/1506.05843

    Parameters
    ----------
    X: [M x K] count data for K categories

    Returns
    -------
    T: [M x K-1] sufficient statistics
    """
    N_m = np.sum(X, axis=1)
    T = np.c_[N_m, N_m[:, None] - np.cumsum(X[:, 0:-2], axis=1)]
    return T


def poly_gamma_rand(n, Psi):
    """
    returns Polyagamma random variables

    Parameters
    ----------
    pg: polya gamma object
    n: [M x K-1] count matrix
    Psi: [M x K-1] Gaussian variables

    Returns
    -------
    omega: [MxK-1] polya gamma variables conditioned on Psi and data n (sufficient statistics of X)

    """

    # f = np.vectorize(pg.pgdraw)
    # return f(n, Psi)
    pg = pypolyagamma.PyPolyaGamma(np.random.randint(0, 2 ** 63, 1))
    return np.reshape([pg.pgdraw(i, j) for i, j in zip(n.ravel(), Psi.ravel())], n.shape)


def stickBreaking(fractions, axis=0):
    """
    Performs stick breaking for the given collection of breaking fractions.

    Parameters
    ----------
    fractions: array of numbers in [0,1]
    axis: axis along which the sticks are oriented

    Returns
    -------
    array of probability vectors (points on the simples) oriented along the same axis
    """
    # check arguments
    f = np.array(fractions)
    assert np.all(np.logical_and(f >= 0, f <= 1))
    assert np.ndim(f) >= axis

    # temporarily permute axis
    f = np.swapaxes(f, 0, axis)

    # stick breaking
    pi = np.zeros((f.shape[0] + 1, *f.shape[1:]))
    pi[0, ...] = f[0, ...]
    pi[1:-1] = f[1:, ...] * np.cumprod(1 - f[0:-1, ...], axis=0)
    pi[-1, ...] = 1 - np.sum(pi[0:-1, ...], axis=0)

    # undo permutation
    pi = np.swapaxes(pi, 0, axis)
    return pi


def reverseStickBreaking(probabilities, axis=0, short=False):
    """
    Converts a given collection of probability vectors into the corresponding stick breaking representations.

    Parameters
    ----------
    probabilities: array of probability vectors (points on the simplex)
    axis: axis along which the probability vectors are oriented
    short: if true it is assumed that the last (dependent) entry of each vector has already been dropped

    Returns
    -------
    array of stick breaking fractions (oriented along the same axis) encoding the probability vectors
    """
    # check arguments
    pi = np.array(probabilities)
    assert np.all(np.logical_and(pi >= 0, pi <= 1))
    assert np.ndim(pi) >= axis

    # temporarily permute axis
    pi = np.swapaxes(pi, 0, axis)

    # if complete probability vectors are provided, drop last entry
    if not short:
        assert np.allclose(np.sum(pi, axis=0), 1)
        pi = pi[0:-1, ...]

    # reverse stick breaking
    den = 1 - np.r_[np.zeros((1, *pi.shape[1:])), np.cumsum(pi[0:-1, ...], axis=0)]
    fracs = np.divide(pi, den, out=np.zeros_like(pi), where=den != 0)
    # fractions following a 1 in the stick breaking representation can not be recovered from the probability
    # vector since they all map to the same result:
    # e.g. stickBreaking([0.5, 1, a, b, c, ...]) yields [0.5, 0.5, 0, 0, 0, 0, ...] for all a, b, c, ...
    # hence, if more than K trailing zeros are provided (K=1 if short=True, K=0 if short=False), all undetermined
    # values are set to np.nan

    # undo permutation
    fracs = np.swapaxes(fracs, 0, axis)
    return fracs


def cholesky_normal(mu, U=None, V=None):
    """
    Generates samples from multivariate normal N(mu, Sigma) using a precomputed Cholesky decomposition of the covariance
    matrix or of the inverse of the covariance matrix:

    Sig = (U^*) U           U = scipy.linalg.cholesky(Sigma))
    inv(Sig) = (V^*) V      V = scipy.linalg.cholesky(inv(Sigma))

    :param mu: mean of the multivariate Gaussian
    :param U:precomputed Cholesky matrix of Sigma
    :param V: precomputed Cholesky matrix of inverse of Sigma
    :return:
    """
    assert (U is None) ^ (V is None)

    x = np.random.randn(len(mu))
    if V is None:
        y = U.T @ x + mu
    else:
        # y = np.linalg.solve(V, x) + mu
        y = solve_triangular(V, x) + mu
    return y
