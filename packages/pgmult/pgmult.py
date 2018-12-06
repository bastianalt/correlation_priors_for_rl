import numpy as np
import pypolyagamma
from scipy.linalg import inv
from packages.utils.utils import stick_breaking, suff_stats_mult, poly_gamma_rand


class pgmult(object):
    def __init__(self, mu_0_k, Sigma_0):
        """
        Parameters
        ----------
        mu_0_k: [M x K-1] prior Mean for each K-1 categories
        Sigma_0: [M x M] Correlation among the M dimensions of the coupled multinomials
        """

        self.mu_0_k = mu_0_k
        self.Sigma_0 = Sigma_0
        self.Sigma_0_inv = inv(Sigma_0)
        self.M, self.K_prime = np.shape(mu_0_k)
        self.pg = pypolyagamma.PyPolyaGamma(seed=0)

    def sample_prior(self):
        """
        Samples Probability matrix from the prior

        Returns
        -------
            Pi: [M x K] Probability matrix over K categories
        """

        Psi = np.zeros((self.M, self.K_prime)) #init
        if self.M == 1:
            Psi = Psi.reshape(1, -1)

        #Draw K_prime samples
        for k in range(self.K_prime):
            Psi[:, k] = np.random.multivariate_normal(self.mu_0_k[:, k], self.Sigma_0)

        Pi = stick_breaking(Psi)
        return Pi

    def create_prior_data(self, N_m):
        """

        Parameters
        ----------
        N_m: M dimensional vector with total number of samples per M


        Returns
        -------
        X: [M x K] M dimesional number of counts per category
        Pi: Probability matrix used for creating the data

        """
        Pi = self.sample_prior()
        X = self.create_model_data(N_m,Pi)
        return X, Pi

    def create_model_data(self,N_m, Pi):
        """

               Parameters
               ----------
               N_m: M dimensional vector with total number of samples per M
               Pi: Probability matrix used for creating the data

               Returns
               -------
               X: [M x K] M dimesional number of counts per category


               """
        X = np.array([np.random.multinomial(N_m[m], Pi[m, :]) for m in range(self.M)])
        return X

    def sample_posterior(self, N_samples, X, Psi_init=False):
        """

        Parameters
        ----------
        N_samples: Number of samples to draw from the posterior
        X: [M x K] Count data used to do posterior inference
        Psi_init: initinal value for the gibbs sampler

        Returns
        -------
        Psi_sam: List of posterior samples for the Gaussians
        omega_sam: List of posterior samples for the Poly Gamma variables
        """

        #Init
        Psi_sam = np.zeros((N_samples,self.M, self.K_prime))
        omega_sam = np.zeros((N_samples,self.M, self.K_prime))

        # TODO Choose clever initalization
        if not (Psi_init):
            Psi_init = np.random.randn(self.M, self.K_prime)
        Psi = Psi_init

        #Transform data
        N_mat = suff_stats_mult(X)
        kappa = X[:, 0:-1] - N_mat / 2

        #Gibbs Sampling!
        for n in range(N_samples):
            omega = poly_gamma_rand(self.pg, N_mat, Psi)

            for k in range(self.K_prime):
                Sigma_k_tilde = inv(self.Sigma_0_inv + np.diag(omega[:, k]))
                mu_k_tilde = (Sigma_k_tilde.dot(
                    kappa[:, k].reshape(-1, 1) + self.Sigma_0_inv.dot(self.mu_0_k[:, k].reshape(-1, 1)))).squeeze()
                Psi[:, k] = np.random.multivariate_normal(mu_k_tilde, Sigma_k_tilde)

            #Add samples to list
            Psi_sam[n,:,:]=Psi
            omega_sam[n,:,:]=omega
        return Psi_sam, omega_sam



