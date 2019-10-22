import sys
sys.path.append('..')

import os
import numpy as np
import xyzpy as xyz
from packages.utils.utils import positions2distmat, normalize01, hellinger, init_Dataset, emd
from packages.mdp.standard_domains import DiagonalTunnelWorld
from packages.mdp.belief_transition_model import PGVarStateTransitionModel
from packages.pgmult.pg_cov import NegExponential
from funcy import rpartial


def estimation_error(T, T_hat, metric, params):
    """
    Computes the error between an estimated transition model and the corresponding ground truth based on a given
    evaluation metric (with optional parameters).

    :param T: [S x S x A] ground truth transition model
    :param T_hat: [S x S x A] estimated transition model
    :param metric: string specifying the evaluation metric
    :param params: parameters of the evaluation metric
    :return: metric value
    """
    # select metric
    if metric == 'emd':
        fun = rpartial(emd, params)
    elif metric == 'hellinger':
        fun = hellinger

    # MDP dimensions
    _, nStates, nActions = T.shape

    # evaluate the metric for each state and action
    E = np.zeros([nStates, nActions])
    for s, a in np.ndindex(nStates, nActions):
        E[s, a] = fun(T[:, s, a], T_hat[:, s, a])

    # return mean and std metric values (computed over the state space of the MDP)
    return E.mean(), E.std()


def experiment(env, nData, methods, metrics, seed):
    """
    Evaluates different methods to estimate the transition model of an MDP.

    :param env: string specifying the test environment
    :param nData: list of integers specifying the training dataset sizes to be tested (number of observed transitions)
    :param methods: list of strings specifying the inference methods to be considered
    :param metrics: list of strings specifying the evaluation metrics to be computed
    :param seed: integer specifying the random seed of the experiment
    :return: xarray containing the mean and standard deviations of all evaluation metrics (computed over the state
        space of the MDP)
    """
    # set random seed
    np.random.seed(seed)

    # initialize empty xarray to store results
    coords = dict(nData=nData, methods=methods, metrics=metrics)
    dims = ['nData', 'methods', 'metrics']
    XR = init_Dataset(['mean', 'std'], coords, dims)

    # create environment
    if env == '10x10_TunnelWorldCorner':
        mdp = DiagonalTunnelWorld(length=10, corner=True)
        distmat = positions2distmat(mdp.statePositions)

    # uniform action exploration policy
    pi = np.full([mdp.nStates, mdp.nActions], fill_value=1/mdp.nActions)

    # create data set
    D = mdp.sampleTrajectories(nTrajs=1, nSteps=nData.max(), pi=pi)
    states, actions = D['states'].ravel(), D['actions'].ravel()

    # iterate over all inference methods
    for method in methods:

        # method specific settings
        if method == 'pg':
            distmat_kernel = normalize01(distmat + distmat.T) ** 2
            Sigma = NegExponential(distmat_kernel)
            T = PGVarStateTransitionModel(mdp.nStates, mdp.nActions, Sigma=Sigma, nonInformative=False)
        elif method == 'dir_sparse':
            alpha = 1e-3
        elif method == 'dir_uniform':
            alpha = 1

        # iterate over different dataset sizes
        for nSteps in nData:
            # extract subset of data
            S = states[0:nSteps]
            A = actions[0:nSteps]

            # create count matrix
            X = np.zeros_like(mdp.T, dtype=int)
            for s1, s2, a in zip(S[:-1], S[1:], A):
                X[s2, s1, a] += 1

            # estimate transition model
            if method == 'pg':
                T.data = X
                T.fit()
                T_hat = T.mean()
            elif method in ('dir_sparse', 'dir_uniform'):
                T_hat = (X + alpha) / (X + alpha).sum(axis=1, keepdims=True)

            # evaluate estimate
            for metric in metrics:
                mean, std = estimation_error(mdp.T, T_hat, metric, distmat)
                XR['mean'].loc[dict(methods=method, nData=nSteps, metrics=metric)] = mean
                XR['std'].loc[dict(methods=method, nData=nSteps, metrics=metric)] = std

    return XR


if __name__ == '__main__':

    # path to store result
    file = '../results/modellearning/result.h5'
    assert os.path.isdir(os.path.dirname(file))

    # number of Monte Carlo runs
    MC = 20

    # "outer" parameter sweep (performed via xyzpy)
    combos = dict(
        seed=np.r_[0:MC],
        env=['10x10_TunnelWorldCorner'],
    )

    # "inner" parameter sweep (manually implemented to share quantities across loops)
    constants = dict(
        nData=np.linspace(0, 1e4, 20, dtype=int),
        metrics=[
            # 'emd',
            'hellinger'
        ],
        methods=[
            'dir_sparse',
            'dir_uniform',
            'pg',
        ],
    )

    # run experiments
    runner = xyz.Runner(experiment, var_names=None)
    harvester = xyz.Harvester(runner, file)
    harvester.harvest_combos(combos, constants=constants, parallel=True)
