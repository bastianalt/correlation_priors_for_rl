import numpy as np
from scipy.special import expit
from scipy.special import logit

def stick_breaking(Psi):
    """

    Parameters
    ----------
    Psi: [M x K-1] Gaussian Variables used for the stick breaking

    Returns
    -------
    Pi: [M x K] Probability matrix using the logistic function and stick breaking
    """
    Pi = np.zeros((np.shape(Psi)[0], np.shape(Psi)[1]+ 1))
    Pi[:,0] = expit(Psi[ :,0])
    Pi[ :,1:-1] = expit(Psi[ :,1:]) * np.cumprod(1 - expit(Psi[ :,0:-1]), axis=1)
    Pi[:,-1] = 1 - np.sum(Pi[ :,0:-1], axis=1)

    return Pi

def inv_stick_breaking(Pi):
    """

    Parameters
    ----------
    Pi: [M x K] Probability matrix using the logistic function and stick breaking


    Returns
    -------
    Psi: [M x K-1] Gaussian Variables used for the stick breaking
    """
    M,K=np.shape(Pi)
    Psi=np.zeros((np.shape(Pi)[0], np.shape(Pi)[1]- 1))
    Psi[:,0]=logit(Pi[ :,0])
    Psi[:, 1:]=logit(Pi[ :,1:-1]/(1-np.cumsum(Pi[ :,0:-2],axis=1)))


    return Psi

def suff_stats_mult(X):
    """

    Parameters
    ----------
    X: [M x K] count data for K categorires

    Returns
    -------
    T: [M x K-1] sufficient statistics
    """
    M, K = np.shape(X)
    N_m = np.sum(X, axis=1)
    T = np.zeros((M, K - 1))
    T[:, 0] = N_m
    T[:, 1:] = (np.tile(N_m, (1, K - 2)).reshape(M, K - 2) - np.cumsum(X[:, 0:-2], axis=1))
    return T

def poly_gamma_rand(pg, n, Psi):
    """

    Parameters
    ----------
    pg: polya gamma object
    n: [M x K] count matrix
    Psi: [M x K] Gaussian variables

    Returns
    -------

    """
    M, K = np.shape(n)
    omega = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            omega[m, k] = pg.pgdraw(n[m, k], Psi[m, k])

    return omega
