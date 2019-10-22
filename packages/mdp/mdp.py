import numpy as np
import cvxpy as cvx
import warnings

from packages.policy.simple_policy import Policy
from packages.mdp.simulator import Simulator
from packages.utils.utils import sampleDiscrete, det2stoch, configurator, randargmax
from packages.utils.utils import parallel as parallelMC
from numpy_groupies import aggregate
from scipy.stats import entropy, dirichlet, binom
from pyemd import emd
from itertools import product
from copy import deepcopy


class MDP(configurator):
    """
    Basis class for discrete-time discounted Markov decision processes (MDPs).

    In the following, S denotes the number of states and A denotes the number of actions.
    """

    def __init__(self, T=None, R=None, discount=None, pi=None):
        """
        :param T: [S x S x A] array representing the transition model (next-states oriented along 0th axis)
                - a simulator object
        :param R:
            - [S] array representing a state dependent reward model
            - [S x A] array representing a state-action dependent reward model
            - [S x S x A] array representing a  state-action-nextState dependent reward model
        :param discount: discount factor in [0, 1)
        :param pi: parameter representing the policy
            - [S] array containing integers from 0 to A-1 representing a static deterministic state-to-action mapping
            - [S x A] array representing a collection of distributions over actions
            - a policy object
        """
        # properties
        self._T = None
        self._R = None
        self._discount = None
        self._opt_V = None
        self._opt_Q = None
        self._opt_pi = None

        # store input values
        self.T = T
        self.R = R
        self.discount = discount
        self.pi = pi

        # computation results
        self._goal_models = {}

    @classmethod
    def DirichletMDP(cls, nStates, nActions, alpha=1, connected=True, nNeighbors=None, linkProbability=None):
        """
        Creates a random MDP with Dirichlet-distributed transition probabilities.

        By default, each state (i.e., node in the graph) is connected to itself. If "connected" is True, each state
        gets additionally connected to the next state (i.e., higher integer state index modulo "nStates"),
        creating a circular graph, to ensure that the MDP is connected. Additional neighbors are added randomly,
        either by adding links according to the specified link probability or until the specified number of neighbors
        "nNeighbors" is reached for all nodes. Hence, the minimum number of Neighbors is 1 if connected=True and 2
        otherwise.

        Note: Either nNeighbors or linkProbability must be None. If both are None, a fully connected MDP is generated.

        :param nStates: number of states
        :param nActions: number of actions
        :param alpha: symmetric Dirichlet concentration parameter for the transition probabilities to neighboring states
        :param connected: if True, the MDP is generated such that it is connected
        :param nNeighbors: integer specifying a fixed number of neighboring states for each state
        :param linkProbability: value in [0, 1] indicating the probability of two selected states getting connected

        :return: MDP object
        """
        # assert that only one of the two connection parameters is provided
        assert nNeighbors is None or linkProbability is None

        # if neither of the two is provided, generate a fully connected MDP
        if nNeighbors is None and linkProbability is None:
            nNeighbors = nStates

        # construct functions for generating numbers of additional links

        if nNeighbors:
            assert 0 <= nNeighbors <= nStates

            # additional number of links to be generated per state
            addLinks = nNeighbors - 2 if connected else nNeighbors - 1

            # return constant number of neighbors
            linkGen = lambda: addLinks
        else:
            # maximum possible number of additional links that can be generated
            maxAddLinks = nStates - 2 if connected else nStates - 1

            # draw random number of neighbors
            linkGen = lambda: binom.rvs(maxAddLinks, linkProbability)

        # generate transition model
        T = np.zeros((nStates, nStates, nActions))
        for s in range(nStates):
            # default links (positions yet to be shifted for actual state)
            l = np.r_[True, np.full(nStates - 1, False)]
            if connected:
                l[1] = True

            # number of additional links to add
            nAddLinks = linkGen()

            # randomly add additional links
            low = 2 if connected else 1
            inds = np.random.choice(range(low, nStates), nAddLinks, replace=False)
            l[inds] = True

            # shift indices to get actual neighbor indices
            neighbors = np.roll(l, s)
            nStateNeighbors = neighbors.sum()

            # generate random transition probabilities to neighbors
            T[neighbors, s, :] = dirichlet.rvs(alpha * np.ones(nStateNeighbors), nActions).T

        return MDP(T=T)

    @property
    def nStates(self):
        # extract number of states from transition model or reward model, if available
        if self.T is not None:
            # Check type of transition model
            if isinstance(self.T, Simulator):
                return self.T.nStates

            return self.T.shape[0]
        elif self.R is not None:
            return self.R.shape[0]
        else:
            return None

    @property
    def nActions(self):
        # extract number of actions from transition model or reward model, if available
        if self.T is not None:
            # Check type of transition model
            if isinstance(self.T, Simulator):
                return self.T.nActions

            return self.T.shape[2]
        elif self.R is not None and self.R.ndim == 2:
            return self.R.shape[1]
        else:
            return None

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, x):
        # in case of empty transition model
        if x is None:
            self._T = None
            return

        # Check if x is a Simulator
        if isinstance(x, Simulator):
            # check consistency with remaining properties
            try:
                if self.nStates is not None:
                    assert x.nStates == self.nStates
                if self.nActions is not None:
                    assert x.nActions == self.nActions
            except AssertionError:
                raise ValueError('inconsistent number of states')

        else:
            # represent transition model as numpy array
            x = np.array(x)

            # check if input satisfies distribution properties
            try:
                assert x.shape[0] == x.shape[1]
                assert np.allclose(x.sum(axis=0), 1)
                assert np.all(x >= 0)
            except AssertionError:
                raise ValueError('invalid transition model')

            # check consistency with remaining properties
            try:
                if self.nStates is not None:
                    assert x.shape[0] == self.nStates
                if self.nActions is not None:
                    assert x.shape[2] == self.nActions
            except AssertionError:
                raise ValueError('inconsistent number of states')

        # store variable and clear dependencies
        self._T = x
        self._opt_V = None
        self._opt_Q = None
        self._opt_pi = None
        self._goal_models = {}

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, x):
        # in case of empty reward model
        if x is None:
            self._R = None
            return

        # represent reward model as numpy array
        x = np.array(x)

        # check if input is valid state reward model
        if x.ndim not in (1, 2, 3):
            raise ValueError('invalid reward model')

        # check consistency with remaining properties
        try:
            if x.ndim == 1 and self.nStates is not None:
                assert len(x) == self.nStates
            if x.ndim == 2 and self.nStates is not None and self.nActions is not None:
                assert x.shape == (self.nStates, self.nActions)
            if x.ndim == 3 and self.nStates is not None and self.nActions is not None:
                assert x.shape == (self.nStates, self.nStates, self.nActions)
        except AssertionError:
            raise ValueError('inconsistent number of states/actions')

        try:
            if x.ndim == 3:
                assert self.T is not None and not isinstance(self.T, Simulator)
                x = np.sum(self.T * x, axis=0)
        except AssertionError:
            raise ValueError('transition model not specified')

        # store variable and clear dependencies
        self._R = x
        self._opt_V = None
        self._opt_Q = None
        self._opt_pi = None

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, x):
        # in case of empty discount factor
        if x is None:
            self._discount = None
            return

        # assert that the discount is in [0,1)
        if not 0 <= x < 1:
            raise ValueError('discount must be in [0,1)')

        # store variable and clear dependencies
        self._discount = x
        self._opt_V = None
        self._opt_Q = None
        self._opt_pi = None

    @property
    def opt_V(self):
        return self._opt_V

    @property
    def opt_Q(self):
        return self._opt_Q

    @property
    def opt_pi(self):
        return self._opt_pi

    @property
    def goal_models(self):
        return self._goal_models

    def _bellmanOperator(self, V):
        """Computes Q-values for all state-action pairs based on the given value estimate."""
        if self.R.ndim == 1:
            return self.R[:, None] + self.discount * np.einsum('ijk,i->jk', self.T, V)
        elif self.R.ndim == 2:
            return self.R + self.discount * np.einsum('ijk,i->jk', self.T, V)

    def linearProgramming(self):
        """
        Linear programming algorithm for MDPs.

        :return: see valueIteration
        """
        # value variables
        V = cvx.Variable(self.nStates)

        # optimization objective
        objective = cvx.Minimize(cvx.sum(V))

        # constraints
        constraints = []
        for s, a in product(range(self.nStates), range(self.nActions)):
            constraints += [V[s] >= self.R[s] + self.discount * self.T[:, s, a] * V]

        # solve problem and extract value variables
        problem = cvx.Problem(objective, constraints)
        problem.solve()
        V = problem.solution.primal_vars[0]

        # get Q-values and policy via lookahead step and greedy selection
        Q = self._bellmanOperator(V)
        pi = randargmax(Q, axis=1)

        # store and return result
        self._opt_V, self._opt_Q, self._opt_pi = V, Q, pi
        return V, Q, pi

    def valueIteration(self, epsilon=1e-6, maxIter=int(1e4)):
        """
        Value iteration algorithm for MDPs.

        :param epsilon: minimum necessary change of MDP values between two steps to keep the iteration running
        :param maxIter: integer specifying the maximum number of iterations
        :return:
            V: [S] array containing the state-values of the MDP
            Q: [S x A] array containing the state-action values of the MDP
            pi: [S] array containing optimal actions for all states
        """
        # initialize value function

        if self.R.ndim == 1:
            V = self.R
        elif self.R.ndim == 2:
            V = np.max(self.R, axis=1)

        # value iteration
        for _ in range(maxIter):
            # keep copy of current value function
            V_old = V.copy()

            # one-step lookahead
            Q = self._bellmanOperator(V)
            V = np.max(Q, axis=1)

            # stop if difference below threshold
            if np.max(np.abs(V - V_old)) < epsilon:
                break

        # extract optimal actions
        pi = randargmax(Q, axis=1)

        # store and return result
        self._opt_V, self._opt_Q, self._opt_pi = V, Q, pi
        return V, Q, pi

    def policyIteration(self, maxIter=int(1e4)):
        """
        Policy iteration algorithm for MDPs.

        Parameters and return values: see valueIteration
        """
        # initialize policy
        pi = np.random.randint(0, self.nActions, self.nStates)
        V = self.policyEvaluation(pi)

        # policy iteration
        for _ in range(maxIter):
            # keep copy of current values
            V_old = V.copy()

            # one-step lookahead
            Q = self._bellmanOperator(V)

            # update policy
            pi = randargmax(Q, axis=1)

            # get value of current policy
            V = self.policyEvaluation(pi)

            # stop if values have converged
            if np.allclose(V, V_old):
                break

        # store and return result
        self._opt_V, self._opt_Q, self._opt_pi = V, Q, pi
        return V, Q, pi

    def policyEvaluation(self, pi=None, discount=None):
        """
        Computes the state-values of the MDP for a given policy.

        :param pi: Parameter containing the policy. Can be either
            - [S] array containing actions for all states of the MDP
            - [S x A] array, where each row represents a discrete distribution over actions
        :param discount: Discount used for the evaluation. When zero, the average reward is computed.
        :return:
            - for nonzero discount: [S] array containing the corresponding MDP values
            - for zero discount: float representing the average reward of the MDP
        """
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        # by default, use discount stored in MDP
        if discount is None:
            discount = self.discount

        # compute marginal transition and reward model
        Tmarg = self._marginalTransitionModel(pi)
        Rmarg = self._marginalRewardModel(pi)

        if discount == 0:
            # compute average reward via stationary state distribution
            u = self._stationaryDistribution(pi)
            return Rmarg @ u
        else:
            # compute discounted value via matrix inversion
            try:
                A = np.eye(self.nStates) - self.discount * Tmarg.T
                return np.linalg.solve(A, Rmarg)
            except np.linalg.LinAlgError:
                # fix for non-reachable states in undiscounted MDPs
                A = np.eye(self.nStates) - 0.999 * Tmarg.T
                V = np.linalg.solve(A, Rmarg)
                return V

    def _stationaryDistribution(self, pi=None):
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        # get eigenvalues and -vectors of marginal transition model
        Tmarg = self._marginalTransitionModel(pi)
        [S, U] = np.linalg.eig(Tmarg)

        # find eigenvector to the eigenvalue 1
        ind = np.argmax(S)
        assert np.isclose(S[ind], 1)
        u = U[:, ind]

        # return positive eigenvector
        return u if np.all(u >= 0) else -u

    def _marginalTransitionModel(self, pi=None):
        """
        Computes the marginal transition model \tilde{T}(s'|s) = \sum_a T(s'|s,a)*pi(a|s) of the MDP for a given policy.

        :param pi: Parameter containing the policy. Can be either
            - [S] array containing actions for all states of the MDP
            - [S x A] array, where each row represents a discrete distribution over actions
        :return: [S x S] array representing the marginal transition model (next-states oriented along 0th axis).
        """
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        if pi.ndim == 1:
            # deterministic policy
            return np.take_along_axis(self.T, pi[None, :, None], axis=2).squeeze(axis=2)
        elif pi.ndim == 2:
            # stochastic policy
            return np.einsum('ijk,jk->ij', self.T, pi)
        else:
            raise ValueError('invalid policy')

    def _marginalRewardModel(self, pi=None):
        """
        Computes the marginal reward model \tilde{R}(s) = \sum_a R(s,a)*pi(a|s) of the MDP for a given policy.

        :param pi: Parameter containing the policy. Can be either
            - [S] array containing actions for all states of the MDP
            - [S x A] array, where each row represents a discrete distribution over actions
        :return: [S] array representing the marginal reward model
        """
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        if self.R.ndim == 1:
            return self.R

        if pi.ndim == 1:
            # deterministic policy
            return np.take_along_axis(self.R, pi[:, None], axis=1).squeeze(axis=1)
        elif pi.ndim == 2:
            # stochastic policy
            return np.sum(pi * self.R, axis=1)
        else:
            raise ValueError('invalid policy')

    def sampleTrajectories(self, nTrajs, nSteps, pi=None, initialStates=None, initialDistribution=None,
                           dropLastState=False, computeRewards=False, parallel=True, callback=None, verbose=False):
        """
        Samples a specified number of trajectories of a given length from the MDP.

        :param pi: Parameter containing the policy. Can be either
            - [S] array containing actions for all states of the MDP
            - [S x A] array, where each row represents a discrete distribution over actions
            - policy object
        :param nTrajs: Integer specifying the number of trajectories to be generated
        :param nSteps: Integer specifying the number of state transitions per trajectory (see return values)
        :param initialStates: Parameter to specify fixed starting states for the trajectories.
            - 'None': the starting states are sampled from the specified initial distribution
            - integer: all trajectories start from the specified state
            - array of length nTrajs: each trajectory starts from its own specified state
        :param initialDistribution: Initial distribution to generate the starting states
            - 'None' and also initialStates is 'None': a uniform distribution over states is assumed
            - 'None' but initialStates is not 'None': the given initial states are used
            - [S] array: treated as distribution over states
        :param dropLastState: Boolean indicating if last state of each trajectory should be dropped
        :param computeRewards: Boolean indicating if rewards should be computed along the trajectories (Only for
        stationary policies. For policy objects, the rewards are anyway computed for the policy update.)
        :param parallel: Boolean to enable parallel generation of trajectories for policy objects in separate
            threads. (Stationary policies are processed in parallel by default, via vectorization.)
        :param callback: optional callback object whose callback method is executed after each policy update
        :param verbose: boolean to print progress
        :return: dictionary with keys 'states', 'actions' (optionally: 'rewards', 'callbacks')
            states: [nTrajs, L] array containing the generated state sequences.
                L = nSteps + 1 if dropLastState == False
                L = nSteps if dropLast State == True
            actions: [nTrajs, nSteps] array containing the generated action sequences.
            rewards: [nTrajs, nSteps] array containing the collected rewards
            callbacks: [nTrajs, L] array containing the callback results (L: see states)
        """
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        # assert that either the initial states or the initial distribution is specified, but not both
        try:
            assert (initialStates is None) or (initialDistribution is None)
        except AssertionError:
            raise ValueError("either 'initialStates' or 'initialDistribution' must be 'None'")

        # if initial states are not specified, use the specified initial distribution or a uniform distribution
        if initialStates is None:
            if initialDistribution is None:
                initialDistribution = np.full(self.nStates, 1 / self.nStates)
            initialStates = sampleDiscrete(initialDistribution, size=nTrajs)

        # check type of policy
        if isinstance(pi, Policy):
            if self.R is None:
                raise ValueError('a strategy policy requires a valid reward model')
            baseStrategy = pi
            isStrategy = True
        else:
            isStrategy = False

        # create arrays to store the trajectories and initialize first states
        states = np.zeros((nTrajs, nSteps + 1), dtype=int)
        states[:, 0] = initialStates
        actions = np.zeros((nTrajs, nSteps), dtype=int)
        if computeRewards:
            rewards = np.zeros((nTrajs, nSteps))
        if callback:
            cb_results = np.empty((nTrajs, nSteps + 1), dtype=object)

        # ----- trajectory generation -----#

        # stationary policy
        if not isStrategy:
            for t in range(nSteps):
                # print output
                if verbose:
                    print(f"time step: {t}/{nSteps}")

                # extract current state and (randomly) select action according to the specified policy
                s = states[:, t]
                a = self.selectActions(s, pi)

                # collect rewards
                if computeRewards:
                    rewards[:, t] = self.earnRewards(s, a)

                # sample next states and store new time slice in arrays
                # Check type of transition model
                if isinstance(self.T, Simulator):
                    states[:, t + 1] = self.T.simulate(curStates=s, curActions=a)
                else:
                    states[:, t + 1] = sampleDiscrete(self.T[:, s, a])
                actions[:, t] = a

        # strategy policy
        else:
            # parallel execution via pool
            if parallel:
                def f(initState, id):
                    np.random.seed(id)
                    return self.executeStrategy(pi, nSteps, initState, callback, id, verbose)

                result = parallelMC(f, [[i] for i in states[:, 0]])
                states, actions, rewards, cb_results = map(np.vstack, zip(*result))
            # serial execution
            else:
                for traj in range(nTrajs):
                    pi = deepcopy(baseStrategy)
                    states[traj, :], actions[traj, :], rewards[traj, :], cb_results[traj, :] = \
                        self.executeStrategy(pi, nSteps, states[traj, 0], callback, traj, verbose)

        # ----- end of trajectory generation -----#

        # if desired, drop last states
        if dropLastState:
            states = states[:, :-1]
            if callback:
                cb_results = cb_results[:, :-1]

        # construct output dictionary
        result = {'states': states, 'actions': actions}
        if computeRewards:
            result['rewards'] = rewards
        if callback:
            result['callbacks'] = cb_results

        return result

    def executeStrategy(self, pi, nSteps, initialState, callback=None, id=0, verbose=False):
        # create arrays to store the trajectories and initialize first state
        states = np.zeros(nSteps + 1, dtype=int)
        states[0] = initialState
        actions = np.zeros(nSteps, dtype=int)
        rewards = np.zeros(nSteps)
        if callback:
            cb_results = np.empty(nSteps + 1, dtype=object)
            cb_results[0] = callback.callback(self, pi)

        # generate the trajectory
        for t in range(nSteps):
            # print output
            if verbose:
                print(f"trajectory: {id}   time step: {t}/{nSteps}")

            # extract current state and (randomly) select action according to the specified policy
            s1 = states[t]
            a = pi.selectActions(s1)

            # evaluate reward and sample next state
            r = self.earnRewards(s1, a)
            # Check type of transition model
            if isinstance(self.T, Simulator):
                s2 = self.T.simulate(curStates=s1, curActions=a)
            else:
                s2 = sampleDiscrete(self.T[:, s1, a])

            # update strategy
            pi.update(s1, a, r, s2)

            # run callback function
            if callback:
                cb_results[t + 1] = callback.callback(self, pi)

            # store values in arrays
            # Check type of transition model
            if isinstance(self.T, Simulator):
                states[t + 1] = self.T.simulate(curStates=s1, curActions=a)
            else:
                states[t + 1] = sampleDiscrete(self.T[:, s1, a])
            actions[t] = a
            rewards[t] = r

        return states, actions, rewards, cb_results

    def earnRewards(self, states, actions):
        return self.R[states] if self.R.ndim == 1 else self.R[states, actions]

    def selectActions(self, states, pi=None):
        # by default, use policy stored in MDP
        if pi is None:
            pi = self.pi

        return pi.selectActions(states) if isinstance(pi, Policy) else selectActions(pi, states)

    def _makeTerminal(self, terminalStates):
        """
        Returns a modified copy of the MDP's transition model in which the selected states are terminal states.

        :param terminalStates: list of terminal states
        :return: modified transition model
        """
        T = self.T.copy()
        T[:, terminalStates, :] = 0
        return T

    def _tunnelState(self, start, end):
        """
        Returns a modified copy of the MDP's transition model in which the given start state points deterministically
        to the given end state for all actions.
        """
        T = self.T.copy()
        T[:, start, :] = 0
        T[end, start, :] = 1
        return T

    def goalPlanning(self, terminalGoals=False, discount=None, useCosts=False):
        """
        Computes optimal strategies for each possible goal in the state space.

        :param terminalGoals: if true, the goals are assumed terminal (episode ends after the goal is reached, meaning
                that no further rewards/costs are obtained)
        :param discount: discount factor used for planning
        :param useCosts: if true, goal model is -1 reward everywhere until goal is reached; otherwise 0 reward
                everywhere except at goal state, which yields +1

        :return: tuple of [S x G], [S x A x G] and [S x G] arrays which represent the optimal V, Q and pi arrays for
                all possible goals (G = S)
        """
        if discount == 1 and terminalGoals and not useCosts:
            raise ValueError('terminal reward of 1 makes no sense without discounting')

        # use MDP discount as default discount
        if discount is None:
            discount = self.discount

        # goal planning setting
        setting = (terminalGoals, discount, useCosts)

        # check if result is already stored
        if setting in self._goal_models.keys():
            return self._goal_models[setting]

        # number of goals = number of states (for readability)
        nGoals = self.nStates

        # container variables
        V = np.zeros((self.nStates, nGoals))
        Q = np.zeros((self.nStates, self.nActions, nGoals))
        pi = np.zeros((self.nStates, nGoals), dtype=int)

        for goal in range(self.nStates):
            # transition model for goal planning
            T = self._makeTerminal(goal) if terminalGoals else self.T

            if useCosts:
                # set reward to -1 at all states, except at goal state
                R = -np.ones(self.nStates)
                R[goal] = 0
            else:
                # set reward to 0 at all states except goal state
                R = np.zeros(self.nStates)
                R[goal] = 1

            # construct MDP and set the transition model and discount manually, to avoid checks by setter methods
            # <-- is OK due to terminal state
            mdp = MDP(R=R)
            mdp._T = T
            mdp._discount = discount

            # planning for the current goal
            V[:, goal], Q[:, :, goal], pi[:, goal] = mdp.policyIteration()

        # store and return result
        self._goal_models[setting] = (V, Q, pi)
        return V, Q, pi

    def minimumTravelTimes(self):
        """
        Computes the minimum expected travel times between all pairs of states.

        :return: [S x S] array, where the (i,j)th element contains the travel time to get from state i to state j
        """
        # the minimum travel times can be expressed as negative MDP values using the following setup
        V, *_ = self.goalPlanning(terminalGoals=True, discount=1, useCosts=True)
        return -V

    def trajs2dems(self, states, actions):
        """
        Converts state and action trajectories into a demonstration data set.

        :param states: [M x N] array containing M state trajectories of length N
        :param actions: [M x N] array containing M action trajectories of length N
        :return: [S x A] array representing S histograms over actions observed at the different states of the MDP
        """
        return aggregate([states.ravel(), actions.ravel()], 1, size=(self.nStates, self.nActions))


def actionLikelihoods(data, policies, aggregateStates=True, logIn=False, logOut=False):
    """
    Computes the action likelihoods of a demonstration data set for a given set of stochastic policies.

    :param data: [D x 2] array containing D state-action pairs
    :param policies: [S x A x P] array containing P stochastic policies
    :param aggregateStates: flag to indicate if the likelihoods should be computed per demonstration pair or if they
            should be aggregated per state
    :param logIn: flag to indicate if the policies are provided in the log domain
    :param logOut: flag to indicate if the result should be return in the log domain

    :return: [D x P] or [S x P] array (depending on the aggregateStates flag) containing the action likelihoods
    """
    # transform policies to log domain
    logGoalPolicies = policies if logIn else np.log(policies)

    # evaluate likelihoods for each demonstration pair
    L = logGoalPolicies[data[:, 0], data[:, 1], :]

    # if desired, aggregate all action likelihoods per state
    if aggregateStates:
        L = aggregate(data[:, 0], L, axis=0, size=policies.shape[0])

    # convert back to linear domain
    if not logOut:
        L = np.exp(L)

    return L


def policyDivergence(pi1, pi2, method, mdp=None, distMat=None):
    """
    Computes the divergence between two policies in various ways.

    Each of the policies can be either deterministic ([S] array) or stochastic ([S x A] array). If only one policy is
    deterministic, it is converted into a stochastic representation, if necessary.

    :param pi1: first policy
    :param pi2: second policy
    :param method: defines the divergence measure
        - 'PL' (policy loss): only for deterministic actions
                compute an [S] boolean array indicating at which states the assigned actions disagree
        - 'KL' (Kullback-Leibler): compute an [S] array containing the Kullback-Leiber divergences of the action
                distributions at all states
        - 'EMD' (earth mover's distance): compute an [S] array containing the EMDs of the local action distributions
                based on the specified distance matrix
        - 'VL' (value loss) compute a single number representing the normalized value loss (Euclidean distance between
                the two value vectors, normalized by the L2-norm of the value vector of pi1)
    :param mdp: MDP object required if method=='VL'
    :param distMat: [A x A] matrix specifying the "distances" between action for method='EMD'
    :return: divergence measure
    """
    # convert policies to numpy arrays
    pi1 = np.array(pi1)
    pi2 = np.array(pi2)

    # divergence based on stochastic or deterministic representations
    stochastic = method in ('VL', 'EMD', 'KL')

    # check policy types and get number of actions
    det1 = (pi1.ndim == 1)
    det2 = (pi2.ndim == 1)
    if stochastic:
        nActions = pi1.shape[1] if det2 else pi2.shape[1]

    # policy loss only possible for determinstic policies
    if (method == 'PL') and (not det1 or not det2):
        raise ValueError('policy loss can be computed only for deterministic policies')

    # if necessary, convert the deterministic policy to a stochastic representation
    if stochastic and det1:
        pi1 = det2stoch(pi1, nActions)
    if stochastic and det2:
        pi2 = det2stoch(pi2, nActions)

    # default distance matrix for EMD
    if (method == 'EMD') and (distMat is None):
        distMat = np.ones((nActions, nActions))

    # policy loss
    if method == 'PL':
        return pi1 != pi2

    # Kullback-Leibler divergence
    elif method == 'KL':
        # interchange policies too obtain meaningful result
        if det2:
            pi2, pi1 = pi1, pi2
            warnings.warn('policies have been interchanged for proper KL computation')
        return np.array([entropy(p1, p2) for p1, p2 in zip(pi1, pi2)])

    # earth mover's distance
    elif method == 'EMD':
        return np.array([emd(dist1.astype(np.float64), dist2.astype(np.float64), distMat.astype(np.float64))
                         for dist1, dist2 in zip(pi1, pi2)])

    # value loss
    elif method == 'VL':
        V1 = mdp.policyEvaluation(pi1)
        V2 = mdp.policyEvaluation(pi2)
        return np.linalg.norm(V1 - V2) / np.linalg.norm(V1)

    else:
        raise ValueError('method not implement')


def normalizeQ(Q):
    """
    Normalized a table of Q-values per state to the range [0,1], producing a set of (normalized) advantage values.
    If all actions at a state yield the same return, the dummy value 1 is assigned to all actions of the state.

    :param Q: [S x A] array representing Q-values
    :return: [S x A] array containing the normalized advantage values
    """
    max = Q.max(axis=1, keepdims=True)
    min = Q.min(axis=1, keepdims=True)
    return np.divide(Q - min, max - min, out=np.ones_like(Q), where=(max != min))


def selectActions(decisionTable, states):
    """
    Selects actions at the specified states for a given (deterministic or stochastic) policy.

    :param decisionTable: Tabular representation of the policy. Can be either
        - [S] array containing actions for all states of the MDP
        - [S x A] array, where each row represents a discrete distribution over actions
    :param states: array of arbitrary length specifying the query states
    :return: array of the same size as the query array containing the selected actions
    """
    # convert input
    decisionTable = np.asarray(decisionTable)
    states = np.atleast_1d(states)

    if decisionTable.ndim == 1:
        # deterministic policy
        return decisionTable[states]
    else:
        # stochastic policy
        return sampleDiscrete(decisionTable[states, :], axis=1)


