import sys
sys.path.append('..')

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import packages.subgoal.inference as sg
from packages.mdp.gridworld import Gridworld
from packages.mdp.mdp import MDP, policyDivergence, normalizeQ, det2stoch
from packages.policy.exploration import softmaxPolicy
from packages.pgmult.pg_mult_model import PGinfer
from packages.utils.utils import parallel, distmat2covmat, positions2distmat


def generateImitationTask(mdp: MDP, R, beta, nTrajs, nSteps):
    # set reward function
    mdp.R = R

    # compute optimal policy, Q-values and corresponding advantage values
    _, Q, pi_opt = mdp.policyIteration()
    Q_norm = normalizeQ(Q)

    # compute softmax policy
    pi = softmaxPolicy(Q_norm, beta)

    # generate demonstration set
    trajs = mdp.sampleTrajectories(nTrajs, nSteps, pi, dropLastState=True)
    S, A = trajs['states'], trajs['actions']

    return pi, pi_opt, S, A


def imitationLearning(mdp, S, A, method, params):
    if method == 'act_dir':
        # dirichlet mean estimate
        D = mdp.trajs2dems(S, A)
        alpha = params[0]
        return (D + alpha) / (D + alpha).sum(axis=1, keepdims=True)

    elif method in ('act_pg_default', 'act_pg_eucl', 'act_pg_traveltime', 'act_pg_noninf'):
        # polya-gamma policy mean estimate
        D = mdp.trajs2dems(S, A)
        return PGinfer(D, *params).mean(axis=0)

    elif method in ('sg_dir', 'sg_pg_default', 'sg_pg_eucl', 'sg_pg_traveltime', 'sg_pg_noninf'):
        # polya-gamma mean estimate via subgoal modeling
        D = np.c_[S.ravel(), A.ravel()]
        return sg.estimatePolicy(D, mdp, *params)

    raise ValueError('unknown method')


def experiment(c_mdp, c_gw, c_rand, c_dir, c_pg, c_sweep, id=0):
    # set random seed
    np.random.seed(id)

    # container to store the results
    result = np.full([len(c_sweep['envs']), len(c_sweep['methods']), len(c_sweep['divergences']),
                      len(c_sweep['nTrajs'])], fill_value=np.nan)

    # sweep number of trajectories
    for tr, nTrajs in enumerate(c_sweep['nTrajs']):

        # sweep environments
        for env, envName in enumerate(c_sweep['envs']):
            if envName == 'randomMDP':
                mdp = MDP.DirichletMDP(nStates=c_rand['nStates'],
                                       nActions=c_rand['nActions'],
                                       linkProbability=c_rand['linkProbability'],
                                       alpha=c_rand['alpha'])
                env_params = c_rand
                mdp.discount = c_mdp['discount']

            elif envName == 'gridworld':
                # create gridworld once (to avoid recomputation of its properties)
                if 'gw' not in locals():
                    gw = Gridworld(shape=c_gw['shape'],
                                    motionPatterns=c_gw['motionPatterns'],
                                    discount=c_mdp['discount'])
                env_params = c_gw
                mdp = gw

            else:
                raise ValueError('unknown environment')

            # generate random reward function
            R = np.zeros(mdp.nStates)
            rewardStates = np.random.choice(range(mdp.nStates), c_mdp['nRewards'], replace=False)
            R[rewardStates] = np.random.random(c_mdp['nRewards'])

            # generate task
            pi, pi_opt, S, A = generateImitationTask(mdp, R, c_mdp['beta'], nTrajs, c_mdp['nSteps'])

            # sweep inference methods
            for m, method in enumerate(c_sweep['methods']):

                # covariance matrix for polya-gamma inference
                if method in ('act_pg_traveltime', 'act_subgoal_traveltime'):
                    travelTimes = mdp.minimumTravelTimes()
                    Sigma = distmat2covmat(travelTimes)

                elif method in ('act_pg_eucl', 'sg_pg_eucl'):
                    if envName == 'randomMDP':
                        continue
                    distmat = positions2distmat(mdp.statePositions)
                    Sigma = distmat2covmat(distmat)

                # inference parameters
                if method == 'act_dir':
                    params = (c_dir['alpha_act'],)
                elif method == 'act_pg_default':
                    params = (c_pg['mu'], None, False, c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'act_pg_eucl':
                    params = (c_pg['mu'], Sigma, False, c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'act_pg_traveltime':
                    params = (c_pg['mu'], Sigma, False, c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'act_pg_noninf':
                    params = (c_pg['mu'], None, True, c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'sg_dir':
                    subgoalModel = ('dir', dict(alpha=c_dir['alpha_sg']))
                    params = (subgoalModel, c_pg['beta'], c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'sg_pg_default':
                    subgoalModel = ('pg', dict(mu=c_pg['mu'], Sigma=None, nonInformative=False))
                    params = (subgoalModel, c_pg['beta'], c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'sg_pg_eucl':
                    subgoalModel = ('pg', dict(mu=c_pg['mu'], Sigma=Sigma, nonInformative=False))
                    params = (subgoalModel, c_pg['beta'], c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'sg_pg_traveltime':
                    subgoalModel = ('pg', dict(mu=c_pg['mu'], Sigma=Sigma, nonInformative=False))
                    params = (subgoalModel, c_pg['beta'], c_pg['nSamples'], c_pg['nBurnin'])
                elif method == 'sg_pg_noninf':
                    subgoalModel = ('pg', dict(mu=c_pg['mu'], Sigma=None, nonInformative=True))
                    params = (subgoalModel, c_pg['beta'], c_pg['nSamples'], c_pg['nBurnin'])

                # perform inference
                pi_hat = imitationLearning(mdp, S, A, method, params)

                # compute MAP action assignment
                # pi_hat = np.argmax(pi_hat, axis=1)
                # pi_hat = det2stoch(pi_hat, mdp.nActions)

                # sweep divergence measures
                for d, div in enumerate(c_sweep['divergences']):
                    result[env, m, d, tr] = policyDivergence(pi_opt, pi_hat, div,
                                                             mdp=mdp, distMat=env_params['actionDistances']).mean()
    return result


def simulate(file):

    # ---------- experimental setup ---------- #
    conf_gridworld = {'shape': (15, 15),
                      'motionPatterns': [np.rot90([[0, 0.7, 0], [0.1, 0, 0.1], [0, 0.1, 0]], r) for r in range(4)],
                      # 'actionDistances': np.array([np.roll([0, 1, 2, 1], i) for i in range(4)]),
                      'actionDistances': None,
                      }

    conf_randomMDP = {'nStates': np.prod(conf_gridworld['shape']),
                      'nActions': 10,
                      'linkProbability': 0.5,
                      'actionDistances': None,
                      'alpha': 0.01}

    conf_sweep = {'nTrajs': np.logspace(0, 4, 10, dtype=int),
                  'divergences': [
                      # 'KL',
                      'VL',
                      'EMD'
                  ],
                  'MC': 30,
                  'envs': [
                      'randomMDP',
                      'gridworld'
                  ],
                  'methods': [
                      'act_dir',
                      # 'act_pg_default',
                      'act_pg_eucl',
                      'act_pg_traveltime',
                      'act_pg_noninf',
                      # 'sg_dir',
                      # 'sg_pg_default',
                      # 'sg_pg_eucl',
                      # 'sg_pg_traveltime',
                      # 'sg_pg_noninf'
                  ]}

    conf_polyagamma = {'nSamples': 200,
                       'nBurnin': 100,
                       'mu': None,
                       'Sigma': [],
                       'beta': 3}

    conf_dir = {'alpha_sg': 0.1,
                'alpha_act': 0.1}

    conf_mdp = {'nSteps': 1,
                'discount': 0.99,
                'beta': 3,
                'nRewards': 1}

    setup = (conf_mdp, conf_gridworld, conf_randomMDP, conf_dir, conf_polyagamma, conf_sweep)

    # ---------- run experiment ---------- #
    # results = experiment(*setup)
    results = parallel(experiment, setup, conf_sweep['MC'])

    # ---------- store results ---------- #
    with open(file, 'wb') as f:
        pickle.dump([setup, results], f)


if __name__ == '__main__':

    file = '../results/imitationlearning/result.pkl'
    assert os.path.isdir(os.path.dirname(file))
    simulate(file)

