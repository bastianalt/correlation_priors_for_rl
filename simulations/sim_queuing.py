import sys

sys.path.append('..')

import numpy as np
from packages.mdp.standard_domains import QueuingNetwork2DBatch
from packages.mdp.belief_transition_model import DirichletTransitionModel, PGVarStateTransitionModel
from evaluation.eval_queuing import evaluate
from packages.policy.exploration import epsilonGreedy
from packages.policy.learning import QLearning
from packages.policy.simple_policy import ModelFreePolicy
import copy
from joblib import Parallel, delayed
import multiprocessing
from packages.pgmult.pg_cov import NegExponential
from packages.utils.utils import normalize01

def distmat2dqueuing(B1, B2):
    """
    Creates distance matrix for queuing
    Parameters
    ----------
    B1 - Buffer size of first queue
    B2 - Buffer size of second queue

    Returns
    -------
    distance matrix
    """
    nStates = (B1 + 1) * (B2 + 1)
    d = np.ones((nStates, nStates))
    vec_tuple = (B1 + 1, B2 + 1)
    for s in range(nStates):
        b = np.array(np.unravel_index(s, vec_tuple))
        for s_prime in range(nStates):
            b_prime = np.array(np.unravel_index(s_prime, vec_tuple))
            d[s, s_prime] = np.sqrt((b - b_prime).T @ (b - b_prime))
    return d


def experiment(path, file_name, conf_env, conf_mdp, conf_transition_model, nMC):

    np.random.seed(nMC)

    final_greedy_policy = []
    values = []
    env = generateEnvironment(conf_env)
    policies = generatePolicies(env, conf_transition_model, conf_env)

    # For policy
    for policy, Tmodel_name, T_model, model_params in policies:
        pi = None
        last_state = 0
        # Generate trajectories
        for episode in range(conf_mdp['nEpisodes']):
            traj = env.sampleTrajectories(nTrajs=1, nSteps=conf_mdp['nSteps'], pi=policy.get_greedy_policy(),
                                          initialStates=last_state, computeRewards=False)
            states = traj['states'][0, :]
            last_state = states[-1]
            actions = traj['actions'][0, :]

            # learn transition model
            nextStates = states[1:]
            curStates = states[:-1]
            T_model.update(nextStates, curStates, actions)

            if model_params['sampling']:
                Q = np.zeros((env.nStates, env.nActions))
                env_estimated = copy.deepcopy(env)
                for sam in range(conf_mdp['nModelSamples']):
                    T_est = T_model.draw()
                    env_estimated.T = T_est
                    _, Q_sam, _ = env_estimated.valueIteration()
                    Q += Q_sam / conf_mdp['nModelSamples']  # Average
            else:
                T_est = T_model.mean()
                env_estimated = copy.deepcopy(env)
                env_estimated.T = T_est
                _, Q, _ = env_estimated.valueIteration()

            # update polciy
            policy.Qtable = Q

            # Greedy policy
            pi = policy.get_greedy_policy()
            # Evalulate for all initial states
            traj_eval = env.sampleTrajectories(nTrajs=env.nStates, nSteps=conf_mdp['nEvaluationSteps'], pi=pi,
                                               initialStates=np.arange(0, env.nStates), computeRewards=True)
            value_eval = np.mean(traj_eval['rewards'], axis=1)

            # save covariance hyperparams

            # evaluate policy
            values.append(dict(value=value_eval,
                               Tmodel_name=Tmodel_name,
                               episode=episode,
                               nMC=nMC))

            # print output
            print(f"episode: {episode + 1}/{conf_mdp['nEpisodes']}" + '     ' +
                  Tmodel_name + '   ' + f"Monte Carlo: {nMC + 1}/{conf_mdp['nMC']}")

        final_greedy_policy.append((pi, Tmodel_name, nMC))

        # np.savez(path + file_name + '_MC' + str(nMC),
        #          values=values,
        #          final_greedy_policy=final_greedy_policy,
        #          conf_env=conf_env,
        #          conf_mdp=conf_mdp,
        #          conf_transition_model=conf_transition_model,
        #          nMC=nMC
        #          )

    return [values, final_greedy_policy]


def generateEnvironment(conf_env):
    _, constructor, params = conf_env
    return constructor(**params)


def generatePolicies(env, conf_transition_model, conf_env):
    policies = []
    for t, t_model in enumerate(conf_transition_model):
        # Qlearning with epsilonGreeedy
        decisionStrategy = lambda Q, states: epsilonGreedy(Q, states, epsilon=.5)
        learning_module = QLearning(nStates=env.nStates, nActions=env.nActions)
        policy_eps = ModelFreePolicy(nStates=env.nStates, nActions=env.nActions, decisionStrategy=decisionStrategy,
                                     learning_module=learning_module)

        Tmodel_name, constructor, params, model_params = t_model
        T_model = constructor(nStates=env.nStates, nActions=env.nActions, **params)

        policies.append((policy_eps, Tmodel_name, T_model, model_params))
    return tuple(policies)


def generateTransitionModels(env, conf_transition_model):
    return [constructor(nStates=env.nStates, nActions=env.nActions) for _, constructor in conf_transition_model]


if __name__ == '__main__':
    path = '../results/queuing/'
    file_name = 'result'
    file_extension = '.npz'

    conf_mdp = dict(nEpisodes=100,
                    nSteps=20,
                    discount=0.9,
                    nEvaluationSteps=1000,
                    nModelSamples=10,
                    nMC=100)

    conf_env = ('2Dim Queuing 10x10', QueuingNetwork2DBatch, dict(B1=10, B2=10, a1=1, d1=3, d2=2, initialStates=0))

    # Covariance matrix
    distmat = normalize01(distmat2dqueuing(conf_env[2]['B1'], conf_env[2]['B2']))
    Sigma = NegExponential(distmat ** 2 / .5)

    conf_transition_model = (
        ('PG Mean', PGVarStateTransitionModel, dict(Sigma=Sigma, nonInformative=False), dict(sampling=False)),
        ('Dir Dense Mean', DirichletTransitionModel, dict(alpha=1), dict(sampling=False)),
        ('Dir Sparse Mean', DirichletTransitionModel, dict(alpha=.001), dict(sampling=False)),
        ('PG Sampling', PGVarStateTransitionModel, dict(Sigma=Sigma, nonInformative=False), dict(sampling=True)),
        ('Dir DenseSampling', DirichletTransitionModel, dict(alpha=1), dict(sampling=True)),
        ('Dir Sparse Sampling', DirichletTransitionModel, dict(alpha=.001), dict(sampling=True))
    )

    num_cores = multiprocessing.cpu_count()
    # Check if experiment already there
    try:
        np.load(path + file_name + file_extension)
    except FileNotFoundError:
        results = Parallel(n_jobs=num_cores)(
            delayed(experiment)(path, file_name, conf_env, conf_mdp, conf_transition_model, nMC) for nMC in
            range(conf_mdp['nMC']))

        np.savez(path + file_name,
                 results=results,
                 conf_env=conf_env,
                 conf_mdp=conf_mdp,
                 conf_transition_model=conf_transition_model,
                 )

    evaluate(path, file_name,file_extension)
