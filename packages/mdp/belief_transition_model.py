from scipy.stats import dirichlet
import numpy as np
from abc import ABC, abstractmethod
from packages.pgmult.pg_mult_model import PgMultNormal, PgMultNIW
from packages.pgmult.utils import stick_breaking


class BeliefTransitionModel(ABC):
    """Abstract base class for a belief transition models"""

    def __init__(self, nStates, nActions):
        """

        Parameters
        ----------
        nStates: int - number of states of the MDP
        nActions: int - number of actions of the MDP
        """
        self.nStates = nStates
        self.nActions = nActions
        self.data = np.zeros((nStates, nStates, nActions))

    @abstractmethod
    def draw(self, N_samples=1):
        """
        Abstract method that returns a draw of the belief transition model
        Returns
        -------
        sample of the transition probability matrix T[nextState,curState,action]

        """
        pass

    def update(self, nextStates, curStates, actions):
        """
        Abstract method that updates the posterior using the obtained data

        Parameters
        ----------
        nextStates: new states of the MDP
        curStates: previous states of the MDP
        actions: actions taken

        """

        self.data[nextStates, curStates, actions] += 1


class DirichletTransitionModel(BeliefTransitionModel):
    """Transition probability model with independent dirichlet distributed rows of the transition matrix"""

    def __init__(self, nStates, nActions, alpha=None):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        alpha: [SxSxA]: parameter vector of the dirichlet prior (pseudo counts) or
                scalar: prior pseudo count number for all states and actions

        """
        super().__init__(nStates=nStates, nActions=nActions)
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, x):

        if x is None:
            x = np.ones_like(self.data)
        elif np.isscalar(x):
            x = np.ones_like(self.data) * x

        try:
            assert x.shape[0] == x.shape[1]
            assert np.all(x > 0)
        except AssertionError:
            raise ValueError('invalid alpha hyper parameter model')

        # check consistency with remaining properties
        try:
            assert x.shape[0] == self.nStates
            assert x.shape[2] == self.nActions

        except AssertionError:
            raise ValueError('inconsistent number of states')

        self._alpha = x

    def draw(self, N_samples=1):
        """

        Returns
        -------
        Posterior draw from the transition model
        """
        # generate transition model
        if N_samples == 1:
            T = np.zeros_like(self.data)
            alpha_post = self.alpha + self.data
            for curState in range(self.nStates):
                for action in range(self.nActions):
                    T[:, curState, action] = dirichlet.rvs(alpha_post[:, curState, action])
        else:
            T = [np.zeros_like(self.data) for _ in range(N_samples)]
            alpha_post = self.alpha + self.data
            for curState in range(self.nStates):
                for action in range(self.nActions):
                    samples = np.split(dirichlet.rvs(alpha_post[:, curState, action], size=N_samples), N_samples)
                    for s, sample in enumerate(samples):
                        T[s][:, curState, action] = sample
        return T

    def mean(self):
        alpha_post = self.alpha + self.data
        T = alpha_post / np.sum(alpha_post, axis=0)[None, :, :]
        return T



class PGStateTransitionModel(BeliefTransitionModel):
    """Transition probability model with correlated rows of the transition matrix using a
     normal sigmoid multinomial model """

    def __init__(self, nStates, nActions, mu=None, Sigma=None, nonInformative=True):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        mu: [S x S-1] Prior mean parameter for the normal distributed variables
        Sigma: [SxS]  Prior Covariance matrix parameter for the normal distributed variables
        nonInformative: bool - if True nonInformative prior is used --> faster inference
        """
        super().__init__(nStates=nStates, nActions=nActions)
        self.PGMult = [PgMultNormal(M=self.nStates, K_prime=self.nStates - 1, mu=mu, Sigma=Sigma,
                                    nonInformative=nonInformative) for _ in range(nActions)]
        self.Psi_init = [None for _ in range(self.nActions)]

    def draw(self, N_samples=1):
        """

        Returns
        -------
        Posterior draw from the transition model

        """
        # generate transition model
        if N_samples == 1:
            T = np.zeros_like(self.data)
            for action in range(self.nActions):
                X = self.data[:, :, action].T
                samples = self.PGMult[action].sample_posterior(1, X, burnin=2, disp=False,
                                                               Psi_init=self.Psi_init[action],
                                                               proj_2_prob=True)
                Pi = np.array(samples['Pi'][-1])  # np.array(stick_breaking(samples['Psi'][-1]))
                T[:, :, action] = Pi.T
                self.Psi_init[action] = samples['Psi'][-1]
        else:
            T = [np.zeros_like(self.data) for _ in range(N_samples)]
            for action in range(self.nActions):
                X = self.data[:, :, action].T
                samples = self.PGMult[action].sample_posterior(N_samples, X, burnin=2, thinning=2, disp=False,
                                                               Psi_init=self.Psi_init[action], proj_2_prob=True)
                for s, sample in enumerate(samples['Pi']):
                    T[s][:, :, action] = np.array(sample).T

                self.Psi_init[action] = samples['Psi'][-1]

        return T

class PGVarStateTransitionModel(BeliefTransitionModel):
    """Transition probability model with correlated rows of the transition matrix using a
     normal sigmoid multinomial model fitting with variational inference"""

    def __init__(self, nStates, nActions, mu=None, Sigma=None, nonInformative=True):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        mu: [S x S-1] Prior mean parameter for the normal distributed variables
        Sigma: [SxS]  Prior Covariance matrix parameter for the normal distributed variables
        nonInformative: bool - if True nonInformative prior is used --> faster inference
        """
        super().__init__(nStates=nStates, nActions=nActions)

        self.PGMult = [PgMultNormal(M=self.nStates, K_prime=self.nStates - 1, mu=mu, Sigma=Sigma,
                                    nonInformative=nonInformative) for _ in range(nActions)]
        self.lambda_ = [None for _ in range(self.nActions)]
        self.V = [None for _ in range(self.nActions)]
        self.isFitted = False  # Flag if the current model is fitted to the data

    def draw(self, N_samples=1):
        """

        Returns
        -------
        Posterior draw from the transition model

        """
        if not self.isFitted:
            self.fit()

        # generate transition model
        if N_samples == 1:
            T = np.zeros_like(self.data)
            for action in range(self.nActions):
                samples = self.PGMult[action].sample_var_posterior(1, proj_2_prob=True)
                Pi = np.array(samples['Pi'][-1])  # np.array(stick_breaking(samples['Psi'][-1]))
                T[:, :, action] = Pi.T
        else:
            T = [np.zeros_like(self.data) for _ in range(N_samples)]
            for action in range(self.nActions):
                samples = self.PGMult[action].sample_var_posterior(N_samples, proj_2_prob=True)
                for s, sample in enumerate(samples['Pi']):
                    T[s][:, :, action] = np.array(sample).T
        return T

    def mean(self):
        """

        Returns
        -------
        the stickbreaking transformation of the posterior mean of the transition model
        """
        if not self.isFitted:
            self.fit()

        T = np.zeros_like(self.data)
        for action in range(self.nActions):
            Pi = stick_breaking(self.lambda_[action])  # np.array(stick_breaking(samples['Psi'][-1]))
            T[:, :, action] = Pi.T
        return T

    def update(self, nextStates, curStates, actions):
        """
        Updates the data of the posterior
        Parameters
        ----------
        nextStates: int next state of the MDP
        curStates: int current state of the MDP
        actions: int action taken
        """
        super().update(nextStates, curStates, actions)
        self.isFitted = False  # New data --> not fitted yet

    def fit(self):
        """

        Returns
        -------
        Fits the posterior using variational inference using the current data

        """
        for action in range(self.nActions):
            X = self.data[:, :, action].T
            self.PGMult[action].fit(X, iter_max=100, iter_tol=1e-3, lambda_init=self.lambda_[action],
                                    V_init=self.V[action])
            # Warm start
            self.lambda_[action] = self.PGMult[action].var_param['lambda_'].copy()
            self.V[action] = self.PGMult[action].var_param['V'].copy()

        self.isFitted = True


class PGStateTransitionModelNIW(BeliefTransitionModel):
    """Transition probability model with correlated rows of the transition matrix using a
     normal sigmoid multinomial model with hierarchical normal inverse Wishart prior"""

    def __init__(self, nStates, nActions, mu_0=None, lambda_0=None, W_0=None, nu_0=None):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        mu_0: [S x S-1] Prior mean parameter for the normal hyper prior distributed variables
        lambda_0: [S-1] precision parameter for the normal hyper prior distributed variables
        W_0: [SxS] Precision matrix for the IW prior
        nu_0: float - degrees of freedom for the IW prior
        """
        super().__init__(nStates=nStates, nActions=nActions)
        self.PGMult = [PgMultNIW(M=self.nStates, K_prime=self.nStates - 1, mu_0=mu_0, lambda_0=lambda_0, W_0=W_0,
                                 nu_0=nu_0) for _ in range(nActions)]
        self.Psi_init = [None for _ in range(self.nActions)]
        # self.data = np.zeros((self.nStates, self.nStates, self.nActions))

    def draw(self, N_samples=1):
        """

        Returns
        -------
        Posterior draw from the transition model
        """
        # generate transition model
        if N_samples == 1:
            T = np.zeros_like(self.data)
            for action in range(self.nActions):
                X = self.data[:, :, action].T
                samples = self.PGMult[action].sample_posterior(1, X, burnin=2, disp=False,
                                                               Psi_init=self.Psi_init[action], proj_2_prob=True)
                Pi = np.array(samples['Pi'][-1])
                T[:, :, action] = Pi.T
                self.Psi_init[action] = samples['Psi'][-1]
        else:
            T = [np.zeros_like(self.data) for _ in range(N_samples)]
            for action in range(self.nActions):
                X = self.data[:, :, action].T
                samples = self.PGMult[action].sample_posterior(N_samples, X, burnin=2, thinning=2, disp=False,
                                                               Psi_init=self.Psi_init[action], proj_2_prob=True)
                for s, sample in enumerate(samples['Pi']):
                    T[s][:, :, action] = np.array(sample).T

                self.Psi_init[action] = samples['Psi'][-1]

        return T
