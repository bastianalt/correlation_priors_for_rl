import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import inv, cholesky, cho_solve, solve_triangular
from numpy.linalg import slogdet
from scipy.optimize import minimize
import cvxpy as cp


class Covariance(object):
    """Base class for covariance objects"""

    def __init__(self):
        # covariance matrix as np.array
        self.val = None
        # lower cholesky factorization of the matrix
        self.cho_lower = None
        # inverse covariance matrix as np.array
        self.inv = None
        # log determinant matrix as np.array
        self.logdet = None
        # gradient of the matrix w.r.t. hyperparameters
        self.grad = None
        pass

    @abstractmethod
    def get_params(self):
        """returns hyperparameters as np array"""
        pass

    @abstractmethod
    def set_params(self, params):
        """
        Sets internal parameters for given hyperparameters
        Parameters
        ----------
        params - hyper parameters as np array
        """
        pass

    def optimize(self, mu, lambda_, V, maxiter=None):
        """
        Generic optimizer for hyper parameters given variational parameters of the ELBO
        Parameters
        ----------
        mu - prior mean
        lambda_ - fitted variational mean
        V - fitted variational covariance
        maxiter - maximum steps in the optimizer

        Returns
        -------
        grad - np array of gradients
        """

        (M, K_prime) = mu.shape
        mu_tilde = mu - lambda_

        def neg_ELBO_hyper(hyperparams):
            """
            Calculates objective function for hyperparams
            Parameters
            ----------
            hyperparams - np array of hyper parameters

            Returns
            -------
            negative ELBO w.r.t. hyperparameters

            """
            # Check if params changed
            if not np.all(self.get_params() == hyperparams):
                self.set_params(hyperparams)

            # Calculate objective value
            objective = K_prime / 2 * self.logdet
            for k in range(K_prime):
                objective += 1 / 2 * np.sum(self.inv * V[:, :, k])  # tr(A.T@B)=sum(A*B)
            for k in range(K_prime):
                objective += 1 / 2 * mu_tilde[:, k].T @ self.inv @ mu_tilde[:, k]

            return objective

        def grad_neg_ELBO_hyper(hyperparams):
            """
            Calculates gradient of the objective function
            Parameters
            ----------
            hyperparams - np array of hyper parameters

            Returns
            -------
            gradient of the negative ELBO w.r.t. hyperparameters
            """

            # Check if params changed
            if not np.all(self.get_params() == hyperparams):
                self.set_params(hyperparams)

            nParams = len(hyperparams)
            grad = np.zeros(nParams)
            # Calculate gradient
            for n in range(nParams):
                gamma = self.inv.T * self.grad[n, :, :]
                grad[n] = K_prime / 2 * np.sum(gamma)
                for k in range(K_prime):
                    alpha_k = self.inv @ mu_tilde[:, k]
                    beta_k = self.inv @ V[:, :, k] @ self.inv
                    grad[n] += -1 / 2 * (np.sum(alpha_k @ alpha_k.T * self.grad[n, :, :] + beta_k * self.grad[n, :, :]))
            return grad

        # Optimize hyper parameters
        res = minimize(neg_ELBO_hyper, self.get_params(), jac=grad_neg_ELBO_hyper,
                       method='BFGS', options={'maxiter': maxiter})
        # Set the parameters
        if not np.all(self.get_params() == res.x):
            self.set_params(res.x)


class NegExponential(Covariance):
    """kernel with a^2 exp(-d)"""

    def __init__(self, distance, amplification=1):
        super().__init__()
        self.distance = distance
        self.amplification = amplification
        self.val = amplification ** 2 * np.exp(-distance) + 1e-6 * np.eye(distance.shape[0])
        Sigma_inv, logdet, L = inv_possemidef(self.val)
        self.inv = Sigma_inv
        self.logdet = logdet
        self.cho_lower = L
        self.grad = 2 * amplification * np.array(np.exp(-distance))

    def get_params(self):
        return np.array(self.amplification)

    def set_params(self, params):
        amplification = params
        self.val = amplification ** 2 * self.val / self.amplification ** 2
        self.inv = self.amplification ** 2 * self.inv / amplification ** 2
        N = self.val.shape[0]
        self.logdet = self.logdet - N * np.log(self.amplification ** 2) + N * np.log(amplification ** 2)
        self.cho_lower = self.amplification * self.cho_lower / amplification
        self.grad = amplification * self.grad / self.amplification
        self.amplification = amplification

    def optimize(self, mu, lambda_, V, maxiter=None):
        # Closed form optimization
        (M, K_prime) = mu.shape
        mu_tilde = mu - lambda_
        amplification2 = 0
        amplitude_old = self.get_params()
        for k in range(K_prime):
            amplification2 += np.sum(amplitude_old ** 2 * self.inv * (
                    mu_tilde[:, k] @ mu_tilde[:, k].T + V[:, :, k]))
        amplification2 = amplification2 / (M * K_prime)
        self.set_params(np.sqrt(amplification2))

def inv_possemidef(Sigma):
    """
    inverts a positive semidefinite matrix using cholesky decomposition
    Parameters
    ----------
    Sigma -  Positive semi definite matrix (Covariance matrix)

    Returns
    -------
    Sigma_inv, -Inverse of Sigma
    logdet,  - log determinant of Sigma
    L, - lower triangular chelseky factorization of Sigma
    """
    L = cholesky(Sigma, lower=True)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    Sigma_inv = L_inv.T @ L_inv
    logdet = 2 * np.sum(np.log(L.diagonal()))
    return Sigma_inv, logdet, L
