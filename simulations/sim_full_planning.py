import sys
sys.path.append('..')

import os
import numpy as np
import xyzpy
from packages.mdp.belief_transition_model import DirichletTransitionModel, PGVarStateTransitionModel
from packages.mdp.mdp import MDP
from packages.policy.exploration import epsilonGreedy
from packages.pgmult.pg_cov import NegExponential
from packages.policy.simple_policy import TabularPolicy, PlanningPolicy
from packages.mdp.standard_domains import DiagonalTunnelWorld
from packages.utils.utils import positions2distmat, normalize01, init_Dataset
from funcy import rpartial


class ValueReporter:
    """Callback class that allows to track the greedy value of a policy during its execution in an MDP via callbacks."""
    def __init__(self):
        # last values to avoid recomputation for unchanged policy
        self.lastInput = (None, None)
        self.lastResult = ()

    def callback(self, mdp: MDP, policy: TabularPolicy):
        """Computes the value of the greedy strategy according to the given policy object."""
        pi = policy.get_greedy_policy()
        return self.computeValue(mdp, pi)

    def computeValue(self, mdp: MDP, pi):
        """Returns the value of a given policy by computation or lookup."""
        if self.lastInput[0] != mdp or np.any(self.lastInput[1] != pi):
            self.lastInput = (mdp, pi)
            self.lastResult = mdp.policyEvaluation(pi).mean()
        return self.lastResult


def constructPlanner(R, discount):
    """For a given reward vector and discount, generates a planner object that computes the optimal Q-values for a
    specified transition model."""
    def planner(T):
        mdp = MDP(T, R, discount)
        V, Q, pi = mdp.policyIteration()
        return Q
    return planner


def experiment(env, policies, seed, **params):
    """
    Evaluates different (strategy) policies in a given MDP.

    :param env: string specifying the test environment
    :param policies: list of strings specifying the policies to be evaluated
    :param seed: integer specifying the random seed of the experiment
    :param params: various additional simulation parameters
    :return: xarray containing the cumulative rewards and (greedy) values of each policy tracked over time
    """
    # set random seed
    np.random.seed(seed)

    # initialize empty xarray to store results
    dims = ['policies', 'time']
    coords = dict(policies=policies, time=np.r_[0:params['nSteps']])
    XR = init_Dataset(vars=['value', 'rewards'], dims=dims, coords=coords)

    # generate environment, random initial state, and planner object
    mdp = generateEnvironment(env)
    init_state = 0
    planner = constructPlanner(mdp.R, mdp.discount)

    # iterate over all policies
    for policy in policies:
        # generate policy and value reporter object
        pi = generatePolicy(policy, mdp, planner, params)
        vr = ValueReporter()

        # execute policy once from the initial state
        trajs = mdp.sampleTrajectories(nTrajs=1, nSteps=params['nSteps'], pi=pi, parallel=False,
                                       initialStates=init_state, computeRewards=True, callback=vr)

        # store cumulative rewards and tracked values
        XR['rewards'].loc[dict(policies=policy)] = trajs['rewards'][0].cumsum()
        XR['value'].loc[dict(policies=policy)] = trajs['callbacks'][0, 0:-1].astype(float)

    return XR


def generateEnvironment(env):
    """Generates an MDP based on its name."""
    if env == '5x5_TunnelWorldCorner':
        return DiagonalTunnelWorld(length=5, corner=True, discount=0.9)
    if env == '10x10_TunnelWorldCorner':
        return DiagonalTunnelWorld(length=10, corner=True, discount=0.9)
    else:
        raise ValueError('unknown environment')


def generatePolicy(policy, mdp, planner, params):
    """Generates a (strategy) policy based on its name."""

    # decision strategy for exploration policy
    decisionStrategy = rpartial(epsilonGreedy, params['epsilon'])

    # common parameters for all policies
    pi_params = dict(nStates=mdp.nStates, nActions=mdp.nActions, decisionStrategy=decisionStrategy, planner=planner,
                     updateFreq=params['planningUpdateFreq'])

    # switch between variational sampling and variational mean
    if policy in ['dir_sparse_sampling', 'dir_uniform_sampling', 'pg_sampling']:
        pi_params['nSamples'] = params['nModelSamples']
    elif policy in ['dir_sparse_mean', 'dir_uniform_mean', 'pg_mean']:
        pi_params['nSamples'] = None
    else:
        raise ValueError('unknown policy')

    # sparse Dirichlet model
    if policy in ['dir_sparse_sampling', 'dir_sparse_mean']:
        pi_params['beliefTransitionModel'] = \
            DirichletTransitionModel(mdp.nStates, mdp.nActions, alpha=1e-3 * np.ones_like(mdp.T))

    # uniform Dirichlet model
    elif policy in ['dir_uniform_sampling', 'dir_uniform_mean']:
        pi_params['beliefTransitionModel'] = DirichletTransitionModel(mdp.nStates, mdp.nActions)

    # PG model
    elif policy in ['pg_sampling', 'pg_mean']:
        distmat = positions2distmat(mdp.statePositions)
        distmat_kernel = normalize01(distmat + distmat.T) ** 2
        Sigma = NegExponential(distmat_kernel)
        pi_params['beliefTransitionModel'] = \
            PGVarStateTransitionModel(mdp.nStates, mdp.nActions, nonInformative=False, Sigma=Sigma)

    return PlanningPolicy(**pi_params)


if __name__ == '__main__':

    # path to store result
    name = 'result.h5'
    file = os.path.join('../results/fullplanning/', name)
    assert os.path.isdir(os.path.dirname(file))

    # number of Monte Carlo runs
    MC = 50

    # "outer" parameter sweep (performed via xyzpy)
    combos = dict(
        seed=np.r_[0:MC],
        planningUpdateFreq=[10, 50],
    )

    # "inner" parameter sweep (manually implemented to share quantities across loops)
    resources = dict(
        policies=['dir_sparse_sampling',
                  'dir_sparse_mean',
                  'dir_uniform_sampling',
                  'dir_uniform_mean',
                  'pg_sampling',
                  'pg_mean',
                  ],
    )

    # other simulation parameters
    constants = dict(
        env='10x10_TunnelWorldCorner',
        epsilon=0.0,
        nModelSamples=10,
        nSteps=2000,
    )

    # run experiments
    runner = xyzpy.Runner(experiment, var_names=None, constants=constants, resources=resources)
    harvester = xyzpy.Harvester(runner, file)
    crop = harvester.Crop(name=name, batchsize=5)
    crop.sow_combos(combos)
    crop.grow_missing(parallel=True, num_workers=4)
    crop.reap()
