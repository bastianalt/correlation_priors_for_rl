import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import cholesky, cho_solve
from numpy.linalg import inv
from packages.pgmult.utils import multivariate_multinomial, stick_breaking, inv_stick_breaking, suff_stats_mult, \
    poly_gamma_rand, cholesky_normal
from scipy.stats import invwishart
from abc import ABC, abstractmethod
from tabulate import tabulate
from packages.pgmult.pg_cov import Covariance, NegExponential
from copy import deepcopy

class PgMultModel(ABC):
    """Class for Logit Stick Breaking Multinomial Models, with inference using Polygamma augmentation"""

    def __init__(self, M, K_prime, Psi_init=None, X_train=None):
        """

        Parameters
        ----------
        M: number of histograms
        K_prime: K_prime=K-1, where K= number of categories
        Psi_init: [Mx K_prime] Initial value for Gibbs sampling used in posterior inference
        """

        self.M = M
        self.K_prime = K_prime
        self.Psi_init = Psi_init
        self.X_train = X_train

    # Setters and Getters for properties
    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, x):
        try:
            x = int(x)
        except ValueError:
            raise ValueError('Number of Histograms has to be an integer')
        try:
            assert x > 0
        except AssertionError:
            raise ValueError('Number of Histograms has to be positive')

        self._M = x

    @property
    def K_prime(self):
        return self._K_prime

    @K_prime.setter
    def K_prime(self, x):
        try:
            x = int(x)
        except ValueError:
            raise ValueError('K_prime has to be an integer')
        try:
            assert x > 0
        except AssertionError:
            raise ValueError('K_prime has to be positive')

        self._K_prime = x

    @property
    def Psi_init(self):
        return self._Psi_init

    @Psi_init.setter
    def Psi_init(self, x):
        # initialize latent Gaussian random variables
        if x is None:
            x = np.random.randn(self.M, self.K_prime)
        try:
            assert x.shape == (self.M, self.K_prime)
        except AssertionError:
            raise ValueError('dimensions mismatch')
        self._Psi_init = x

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, x):

        # compute sufficient statistics from data set
        self._N_mat = None
        self._kappa = None
        if x is not None:
            self._N_mat = suff_stats_mult(x)
            self._kappa = x[:, 0:-1] - self._N_mat / 2
        self._X_train = x

    @abstractmethod
    def sample_prior(self, proj_2_prob=False):
        pass

    @staticmethod
    def sample_Psi(kappa, omega_sam, mu, Sigma, Sigma_inv=None, nonInformative=False):
        """
        Gibbs step for normal distributed variables conditioned on PG auxiliary variables

        Parameters
        ----------
        kappa: [M x K] data vector containing the sufficient statistics
        omega_sam: [M x K] sampled value from the PG distribution
        mu: [MxK-1] K-1 means for M dimensional multivariate Gaussian
        Sigma: [MxM] correlation matrix of the M histograms

        Returns
        -------

        """
        (M, K_prime) = kappa.shape
        Psi_sam = np.zeros_like(mu)

        if nonInformative:
            # assume C -> inf, i.e., Sigma_k_tilde = (omega_k)^(-1)

            sigma_mat = 1 / (omega_sam + 1e-6)
            mu_mat = sigma_mat * kappa
            Psi_sam = mu_mat + np.sqrt(sigma_mat) * np.random.randn(M, K_prime)
            # for k in range(K_prime):
            #     Psi_sam[:, k] = mu_mat[:, k] + np.sqrt(sigma_mat[:, k]) * np.random.randn(M)
        else:
            if Sigma_inv is None:
                Sigma_inv = inv(Sigma)

            for k in range(K_prime):
                # Exploiting cholesky factorization
                Sigma_k_tilde_inv = Sigma_inv + np.diag(omega_sam[:, k])
                U = cholesky(Sigma_k_tilde_inv)
                mu_k_tilde = cho_solve((U, False), kappa[:, k] + Sigma_inv @ mu[:, k])
                Psi_sam[:, k] = cholesky_normal(mu_k_tilde, V=U)
        return Psi_sam

    def create_prior_data(self, N_m):
        """
        Samples a Pi from the prior and then draws data X from it.

        Parameters
        ----------
        N_m: M dimensional vector with total number of samples per M


        Returns
        -------
        X: [M x K] M dimensional number of counts per category
        Pi: Probability matrix used for creating the data

        """
        prior_sam = self.sample_prior(proj_2_prob=True)
        X = multivariate_multinomial(N_m, prior_sam['Pi'])
        return X, prior_sam['Pi']

    @abstractmethod
    def sample_posterior(self, N_samples, X, Psi_init=None, burnin=None, thinning=None, proj_2_prob=False, disp=True):
        pass

    @abstractmethod
    def gibbs_step(self, X=None, Psi_init=None, proj_2_prob=False):
        pass


class PgMultNormal(PgMultModel):
    def __init__(self, M=None, K_prime=None, mu=None, Sigma=None, nonInformative=False, Psi_init=None, lambda_init=None,
                 V_init=None, X_train=None):

        super().__init__(M, K_prime, Psi_init, X_train)
        self.hyper = {'mu': mu, 'Sigma': Sigma, 'nonInformative': nonInformative}
        self.var_param = {'lambda_': None, 'V': None}
        self.lambda_init = lambda_init
        self.V_diag_init = None
        self.V_init = V_init

    @property
    def lambda_init(self):
        return self._lambda_init

    @lambda_init.setter
    def lambda_init(self, x):
        if x is None:
            x = np.zeros((self.M, self.K_prime))
        self._lambda_init = x

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, x):
        if x is None:
            V_diag = np.ones((self.M, self.K_prime))
            x = np.repeat(np.eye(self.M)[:, :, np.newaxis], self.K_prime, axis=2)
        else:
            V_diag = np.zeros((self.M, self.K_prime))
            for k in range(self.K_prime):
                V_diag[:, k] = x[:, :, k].diagonal()
                # np.array([V_mat.diagonal() for V_mat in x]).T

        self._V_init = x
        self.V_diag_init = V_diag

    @property
    def hyper(self):
        return self._hyper

    @hyper.setter
    def hyper(self, x):
        if x['nonInformative']:
            x['mu'] = np.zeros((self.M, self.K_prime))  # sp.csr_matrix((self.M, self.K_prime + 1))
            x['Sigma'] = np.diag(np.full(self.M, fill_value=np.inf))  # sp.eye(self.M) * np.inf
            x['Sigma_inv'] = np.zeros((self.M, self.M))  # sp.csr_matrix((self.M, self.M))
        else:
            try:
                # Vague normal sigmoid stick breaking prior
                if x['mu'] is None:
                    p = np.ones((self.M, self.K_prime + 1)) / (self.K_prime + 1)
                    x['mu'] = inv_stick_breaking(p)

                if x['Sigma'] is None:
                    L = np.random.randn(self.M, self.M)
                    x['Sigma'] = 1e6 * L @ L.T

            except TypeError:
                raise TypeError('Invalid Prior')

            try:
                assert x['mu'].shape == (self.M, self.K_prime)
                if isinstance(x['Sigma'], Covariance):
                    x['Sigma_obj'] = deepcopy(x['Sigma'])
                else:
                    if np.all(x['Sigma'] != np.inf):
                        assert x['Sigma'].shape == (self.M, self.M)
            except AssertionError:
                raise TypeError('Invalid Prior')

            if isinstance(x['Sigma'], Covariance):
                x['Sigma'] = x['Sigma_obj'].val
                x['Sigma_inv'] = x['Sigma_obj'].inv
            else:
                x['Sigma_inv'] = 0 if (np.any(x['Sigma'] == np.inf)) else inv(x['Sigma'])

        self._hyper = deepcopy(x)

    def sample_prior(self, proj_2_prob=False):
        """
        Samples Probability matrix from the prior

        Parameters
        ----------
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------
            x - Dict: keys: Psi, Pi (optional)
            Psi: [MxK_prime] Normal sample from the prior distribution
            Pi: [M x K] Probability matrix over K categories
        """

        Psi = np.array(
            [multivariate_normal(self.hyper['mu'][:, k], self.hyper['Sigma']) for k in range(self.K_prime)]).T
        if proj_2_prob:
            Pi = stick_breaking(Psi)
            x = {'Psi': Psi, 'Pi': Pi}
        else:
            x = {'Psi': Psi}

        return x

    def gibbs_step(self, X=None, Psi_init=None, proj_2_prob=False):
        """
        calculates one gibbs step
        Parameters
        ----------
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex
        X: [MxK] data matrix if None previously saved training data is used
        Psi_init: [MxK_prime] starting value of the Gibbs sampler for the normal distributed variables
            (if None last value is used)

        Returns
        -------
        x - Dict: keys: Psi, omega Pi (optional)
        Psi: [MxK_prime] One normal sample from the posterior distribution
        omega: [MxK_prime] One PG sample from the posterior distribution
        Pi: [M x K] One sample from the probability matrix over K categories
        """

        if X is not None:
            self.X_train = X
        if Psi_init is not None:
            self.Psi_init = Psi_init

        # --------------- update polya-gamma variables --------------- #
        # polya-gamma auxiliary variable
        omega_sam = poly_gamma_rand(self._N_mat, self.Psi_init)

        # --------------- update latent Gaussian variables --------------- #
        # sample new latent Gaussian variables
        Psi_sam = self.sample_Psi(self._kappa, omega_sam, self.hyper['mu'], self.hyper['Sigma'],
                                  Sigma_inv=self.hyper['Sigma_inv'], nonInformative=self.hyper['nonInformative'])

        self.Psi_init = Psi_sam

        if proj_2_prob:
            Pi = stick_breaking(Psi_sam)
            x = {'Psi': Psi_sam, 'omega': omega_sam, 'Pi': Pi}
        else:
            x = {'Psi': Psi_sam, 'omega': omega_sam}

        return x

    def sample_posterior(self, N_samples, X, Psi_init=None, burnin=0, thinning=0, disp=True, proj_2_prob=False):
        """
        Samples N_samples from the Posterior given data X
        Parameters
        ----------
        N_samples: Number of samples to draw from the posterior
        X: [M x K] Count data used to do posterior inference
        Psi_init: initial value for the gibbs sampler
        burnin: int number of burnin samples
        thinning: number of discarded values in the chain between two samples
        disp: output steps of the MCMC sampler in the console
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------
        post_samples - Dict: keys: Psi, omega, Pi (optional)
        Psi: List of posterior samples for the Gaussians
        omega: List of posterior samples for the Poly Gamma variables
        Pi: List of posterior samples for the probability matrix over K categories
        """

        # container for the posterior samples
        if proj_2_prob:
            post_samples = {'Pi': [], 'Psi': [], 'omega': []}
        else:
            post_samples = {'Psi': [], 'omega': []}

        # initialize latent Gaussian random variables
        if Psi_init is not None:
            self.Psi_init = Psi_init

        self.X_train = X  # Save input and compute sufficient stats

        enough_samples = False
        n = 0
        while not enough_samples:
            n += 1
            # Print results
            if disp:
                print(tabulate([[n]], headers=['MCMC sampling step']))

            sam = self.gibbs_step()

            # add samples to list
            if n > burnin and np.mod(n, thinning + 1) == 0:
                if proj_2_prob:
                    post_samples['Pi'].append(stick_breaking(sam['Psi'].copy()))

                post_samples['Psi'].append(sam['Psi'].copy())
                post_samples['omega'].append(sam['omega'].copy())

            if post_samples['Psi'].__len__() == N_samples:
                enough_samples = True

        return post_samples

    def fit(self, X, iter_max=100, iter_tol=1e-3, lambda_init=None, V_init=None):
        """
        Fits the variational model to the data

        Parameters
        ----------
        X - [MxK] count data
        iter_max: Maximimum number of iterations in the CAVI ELBO update
        iter_tol: tolerance for the CAVI ELBO update of the normalized change in parameter lambda_
        lambda_init: [MxK-1] Init value for the Variational parameter for the mean
        V_init: K-1 dim list [MXM] Init value for the Variational parameter for the variance

        Returns
        -------
        var_param - dictionary with fitted variational parameters

        """

        self.X_train = X  # Save input and compute sufficient stats
        if lambda_init is not None:
            self.lambda_init = lambda_init
        if V_init is not None:
            self.V_init = V_init

        lambda_ = self._lambda_init
        V = self.V_init
        V_diag = self.V_diag_init

        v = np.zeros((self.M, self.K_prime))
        u = self._N_mat

        converged = False
        iter = 0
        while not converged:
            lambda_old = lambda_.copy()

            # CAVI Update PG
            if self.hyper['nonInformative']:
                v = np.sqrt(V_diag + lambda_ ** 2)
            else:
                for k in range(self.K_prime):
                    v[:, k] = np.sqrt(np.diag(V[:, :, k]) + lambda_[:, k] ** 2)

            # CAVI Update Gaussian
            omega_mean = u / (2 * v) * np.tanh(v / 2)
            if self.hyper['nonInformative']:
                V_diag = 1 / (omega_mean + 1e-6)  # numerics
                lambda_ = V_diag * self._kappa
            else:
                for k in range(self.K_prime):
                    V[:, :, k] = inv(self.hyper['Sigma_inv'] + np.diag(omega_mean[:, k]))
                    lambda_[:, k] = V[:, :, k] @ (self._kappa[:, k] + self.hyper['Sigma_inv'] @ self.hyper['mu'][:, k])

            # Optimize hyper parameters
            if 'Sigma_obj' in self.hyper:
                self.hyper['Sigma_obj'].optimize(self.hyper['mu'], lambda_, V)
                self.hyper['Sigma'] = self.hyper['Sigma_obj']
                # optimize mu
                self.hyper['mu']= lambda_.copy()


            iter += 1
            with np.errstate(invalid='ignore', divide='ignore'):
                rel_change = np.linalg.norm(lambda_old - lambda_) / np.linalg.norm(lambda_old)
            if iter > iter_max or rel_change < iter_tol:
                converged = True

        self.var_param['lambda_'] = lambda_

        if self.hyper['nonInformative']:
            for k in range(self.K_prime):
                V[:, :, k] = np.diag(V_diag[:, k])
            self.var_param['V'] = V
            self.var_param['V_diag'] = V_diag
        else:
            self.var_param['V'] = V
            # Reset stored cholesky
            self.var_param['V_cho'] = None

        return self.var_param

    def sample_var_posterior(self, N_samples, proj_2_prob=False):
        """
        Creates N samples from the variational posterior
        Parameters
        ----------
        N_samples - number of posterior samples
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------

        """
        # container for the posterior samples
        if proj_2_prob:
            post_samples = {'Pi': [], 'Psi': []}
        else:
            post_samples = {'Psi': []}

        Psi_sam = np.zeros((N_samples, self.M, self.K_prime))
        if self.hyper['nonInformative']:
            Psi_sam = self.var_param['V_diag'][None, :, :] * np.random.randn(N_samples, self.M, self.K_prime) + \
                      self.var_param['lambda_'][None, :, :]
        else:

            # Precompute cholesky if not already calculated
            if self.var_param['V_cho'] is None:
                V_cho = np.zeros_like(self.var_param['V'])
                for k in range(self.K_prime):
                    V_cho[:, :, k] = cholesky(self.var_param['V'][:, :, k])
                self.var_param['V_cho'] = V_cho

            # sample using cholesky
            for n in range(N_samples):
                for k in range(self.K_prime):
                    Psi_sam[n, :, k] = cholesky_normal(self.var_param['lambda_'][:, k],
                                                       U=self.var_param['V_cho'][:, :, k])

        post_samples['Psi'] = [sam for sam in Psi_sam]
        if proj_2_prob:
            post_samples['Pi'] = [stick_breaking(sam.copy()) for sam in post_samples['Psi']]

        return post_samples

    def mean_variational_posterior(self, proj_2_prob=False):
        """
        Returns the mean of the variational posterior
        Parameters
        ----------
         proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------
        mean - dictonary containing the variational mean
        """
        if proj_2_prob:
            mean = {'Pi': None, 'Psi': None}
        else:
            mean = {'Psi': None}

        mean['Psi'] = self.var_param['lambda_']
        if proj_2_prob:
            mean['Pi'] = stick_breaking(self.var_param['lambda_'])
        return mean

class PgMultNIW(PgMultModel):
    def __init__(self, M=None, K_prime=None, mu_0=None, lambda_0=None, W_0=None, nu_0=None, Psi_init=None,
                 X_train=None):
        super().__init__(M, K_prime, Psi_init, X_train)
        self.hyper = {'mu_0': mu_0, 'lambda_0': lambda_0, 'W_0': W_0, 'nu_0': nu_0}

    @property
    def hyper(self):
        return self._hyper

    @hyper.setter
    def hyper(self, x):
        try:
            # Vague hierarchical normal sigmoid stick breaking prior
            if x['mu_0'] is None:
                p = np.ones((self.M, self.K_prime + 1)) / (self.K_prime + 1)
                x['mu_0'] = inv_stick_breaking(p)
            if x['lambda_0'] is None:
                x['lambda_0'] = 1e-6 * np.ones(self.K_prime)
            if x['W_0'] is None:
                L = np.random.randn(self.M, self.M)
                x['W_0'] = 1e-6 * L @ L.T
            if x['nu_0'] is None:
                x['nu_0'] = self.M + 1e-6
        except TypeError:
            raise TypeError('Invalid Prior')

        try:
            assert x['mu_0'].shape == (self.M, self.K_prime)
            assert x['lambda_0'].__len__() == self.K_prime
            assert x['W_0'].shape == (self.M, self.M)
            assert x['nu_0'] >= self.M
        except AssertionError:
            raise TypeError('Invalid Prior')

        # Set help variables
        self._lambda_bar = x['lambda_0'] + 1
        self._lambda_ratio = x['lambda_0'] / self._lambda_bar
        self._nu_bar = x['nu_0'] + self.K_prime

        self._hyper = x

    def sample_prior(self, proj_2_prob=False):
        """
        Samples Probability matrix from the prior

        Parameters
        ----------
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------
            x - Dict: keys: mu, Sigma, Psi, Pi (optional)
            mu: [MxK_prime] Normal sample from the prior
            Sigma: [MxM] Inverse Wishart sample from the prior
            Psi: [MxK_prime] Normal sample from the prior distribution
            Pi: [M x K] Probability matrix over K categories
        """
        Psi = []
        Sigma = invwishart.rvs(self.hyper['nu_0'], inv(self.hyper['W_0']))
        mu = np.zeros((self.M, self.K_prime))
        for k in range(self.K_prime):
            mu[:, k] = multivariate_normal(self.hyper['mu_0'][:, k], 1 / self.hyper['lambda_0'][k] * Sigma)
            Psi.append(multivariate_normal(mu[:, k], Sigma))

        # Pi = stick_breaking(np.array(Psi))

        if proj_2_prob:
            Pi = stick_breaking(np.array(Psi))
            x = {'Psi': Psi, 'Pi': Pi, 'mu': mu, 'Sigma': Sigma}
        else:
            x = {'Psi': Psi, 'mu': mu, 'Sigma': Sigma}

        return x

    def gibbs_step(self, X=None, Psi_init=None, proj_2_prob=False):
        """
        calculates one gibbs step
        Parameters
        ----------
        X: [MxK] data matrix if None previously saved training data is used
        Psi_init: [MxK_prime] starting value of the Gibbs sampler for the normal distributed variables
            (if None last value is used)
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex


        Returns
        -------
        x - Dict: keys: Psi, omega, mu, Sigma, Pi (optional)
        Psi: [MxK_prime] One normal sample from the posterior distribution
        omega: [MxK_prime] One PG sample from the posterior distribution
        mu: [MxK_prime] One mean parameter sample from the posterior distribution
        Sigma: [MxM] One covariance parameter sample from the posterior distribution
        Pi: [M x K] One sample from the probability matrix over K categories
        """
        if X is not None:
            self.X_train = X
        if Psi_init is not None:
            self.Psi_init = Psi_init

        # --------------- update Gaussian covariance matrix --------------- #
        # posterior covariance hyperparameter
        Psi_centered = self.Psi_init - self.hyper['mu_0']
        W_bar = self.hyper['W_0'] + (self._lambda_ratio * Psi_centered) @ Psi_centered.T

        # sample new covariance matrix
        Sigma_sam = invwishart.rvs(self._nu_bar, inv(W_bar))

        # --------------- update Gaussian means --------------- #
        # posterior mean hyperparameters
        mu_bar = (self.hyper['mu_0'] * self.hyper['lambda_0'] + self.Psi_init) / self._lambda_bar

        # sample new mean values
        mu_sam = np.zeros((self.M, self.K_prime))
        U = cholesky(Sigma_sam)
        for k in range(self.K_prime):
            mu_sam[:, k] = cholesky_normal(mu_bar[:, k], U=1 / np.sqrt(self._lambda_bar[k]) * U)

        # --------------- update polya-gamma variables --------------- #
        # polya-gamma auxiliary variable
        omega_sam = poly_gamma_rand(self._N_mat, self.Psi_init)

        # --------------- update latent Gaussian variables --------------- #
        # sample new latent Gaussian variables
        Psi_sam = self.sample_Psi(self._kappa, omega_sam, mu_sam, Sigma_sam)

        self.Psi_init = Psi_sam

        if proj_2_prob:
            Pi = stick_breaking(Psi_sam)
            x = {'Psi': Psi_sam, 'omega': omega_sam, 'Pi': Pi, 'mu': mu_sam, 'Sigma': Sigma_sam}
        else:
            x = {'Psi': Psi_sam, 'omega': omega_sam, 'mu': mu_sam, 'Sigma': Sigma_sam}

        return x  # Psi_sam, omega_sam, mu_sam, Sigma_sam

    def sample_posterior(self, N_samples, X, Psi_init=None, burnin=0, thinning=0, disp=True, proj_2_prob=False):
        """
        Samples N_samples from the Posterior given data X
        Parameters
        ----------
        N_samples: Number of samples to draw from the posterior
        X: [M x K] Count data used to do posterior inference
        Psi_init: initial value for the gibbs sampler
        burnin: int number of burnin samples
        thinning: number of discarded values in the chain between two samples
        disp: output steps of the MCMC sampler in the console
        proj_2_prob: Bool: if TRUE  returns the projection to the probability simplex

        Returns
        -------
        post_samples - Dict: keys: Psi, omega, mu, Sigma, Pi (optional)
        Psi: List of posterior samples for the Gaussians
        omega: List of posterior samples for the Poly Gamma variables
        Sigma: List of posterior samples for covariance matrix
        mu: List of posterior samples for the means
        Pi: List of posterior samples for the probability matrix over K categories
        """

        # container for the posterior samples
        post_samples = {'Pi': [], 'Psi': [], 'omega': [], 'mu': [], 'Sigma': []}

        # initialize latent Gaussian random variables
        if Psi_init is not None:
            self.Psi_init = Psi_init

        self.X_train = X  # Save input and compute sufficient stats

        # generate Gibbs samples
        enough_samples = False
        n = 0
        while not enough_samples:
            n += 1
            # Print results
            if disp:
                print(tabulate([[n]], headers=['MCMC sampling step']))

            sam = self.gibbs_step()
            # add samples to list
            if n > burnin and np.mod(n, thinning + 1) == 0:
                if proj_2_prob:
                    post_samples['Pi'].append(stick_breaking(sam['Psi'].copy()))

                post_samples['Psi'].append(sam['Psi'].copy())
                post_samples['omega'].append(sam['omega'].copy())
                post_samples['mu'].append(sam['mu'].copy())
                post_samples['Sigma'].append(sam['Sigma'].copy())

            if post_samples['Psi'].__len__() == N_samples:
                enough_samples = True

        return post_samples


def PGinfer(data, mu, Sigma, nonInformative, nSamples, nBurnin):
    """
    Helper function for inference of correlated probabilities
    Parameters
    ----------
    data - [M x K] - counts of K categories for a set of M histograms
    nSamples - Number of posterior samples
    mu - [MxK-1] Mean parameter for the prior variables in the normal sigmoid multinomial model
    Sigma- [MxM] Covariance parameter for the prior variables in the normal sigmoid multinomial model

    Returns
    -------
    nSamples of the [MxK] probability matrices
    """
    nDists, nCats = data.shape
    PgMultObj = PgMultNormal(M=nDists, K_prime=nCats - 1, mu=mu, Sigma=Sigma, nonInformative=nonInformative)
    samples = PgMultObj.sample_posterior(nSamples, data, disp=False, burnin=nBurnin)
    Psi = samples['Psi']
    Pi_samples = np.array([stick_breaking(sample) for sample in Psi])
    return Pi_samples


def PG_infer_var(data, Sigma, nSamples=None):
    """
    Helper function for inference of correlated probabilities
    Parameters
    ----------
    data - [M x K] - counts of K categories for a set of M histograms
    nSamples - Number of posterior samples
    mu - [MxK-1] Mean parameter for the prior variables in the normal sigmoid multinomial model
    Sigma- [MxM] Covariance parameter for the prior variables in the normal sigmoid multinomial model

    Returns
    -------
    nSamples of the [MxK] probability matrices
    """
    nDists, nCats = data.shape
    PgMultObj = PgMultNormal(M=nDists, K_prime=nCats - 1, mu=None, Sigma=Sigma, nonInformative=False)
    PgMultObj.fit(data)
    if nSamples is None:
        Pi = PgMultObj.mean_variational_posterior(proj_2_prob=True)['Pi']
    else:
        Pi = np.array(PgMultObj.sample_var_posterior(nSamples, proj_2_prob=True)['Pi']).mean(axis=0)
    return Pi
