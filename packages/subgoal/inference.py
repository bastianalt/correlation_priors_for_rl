import numpy as np
import packages.mdp.mdp as mdp
from packages.pgmult.pg_mult_model import PgMultNormal, NegExponential
from packages.utils.utils import softmax, sampleDiscrete
from packages.utils.utils import positions2distmat, normalize01


def estimatePolicy(data, env: mdp.MDP, prior, beta, nSamples=100, nBurnin=50):

    if prior == 'pg':
        # create PG model as prior for the subgoal assignments
        distmat = positions2distmat(env.statePositions)
        distmat_kernel = normalize01(distmat + distmat.T) ** 2
        Sigma = NegExponential(distmat_kernel)
        PG = PgMultNormal(M=env.nStates, K_prime=env.nStates-1, mu=None, Sigma=Sigma, nonInformative=False)

        def fitGoalDists(goals):
            # represent goals as "histograms" (i.e. data for the PG model)
            goal_hists = mdp.det2stoch(goals, env.nStates)
            PG.fit(goal_hists)
            sg_mean = PG.mean_variational_posterior(proj_2_prob=True)['Pi']
            # sg_sample = PG.sample_var_posterior(1, proj_2_prob=True)['Pi']
            return sg_mean
            # return sg_sample

    elif prior == 'dir':
        def fitGoalDists(goals):
            alpha = 1e-3
            goal_hists = mdp.det2stoch(goals, env.nStates) + alpha
            return goal_hists / goal_hists.sum(axis=1)

    # compute subgoal values and softmax policies
    _, Q, _ = env.goalPlanning()
    # Q = mdp.normalizeQ(Q)
    goalPolicies = softmax(beta * Q, axis=1)

    # evaluate the action likelihoods under all subgoal policies
    L = mdp.actionLikelihoods(data, goalPolicies, logOut=True)

    # create container for Gibbs samples and initialize first sample randomly
    G = np.zeros((nSamples, env.nStates), dtype=int)
    G[0, :] = np.random.randint(0, env.nStates, env.nStates)

    def gibbs_step(goals):
        # get Gibbs sample from the PG conditional distribution
        goal_distributions = fitGoalDists(goals)

        # combine with "prior" (i.e. the action likelihoods) in the log domain and convert back to linear domain
        weights = softmax(np.log(goal_distributions) + L, axis=1)

        # return a sample from the resulting conditional distribution
        return sampleDiscrete(weights, axis=1)

    # run the Gibbs sampler
    for i in range(nSamples - 1):
        print(i)
        G[i+1, :] = gibbs_step(G[i, :])

    # discard the burnin samples
    G = G[nBurnin + 1:]

    # construct policy estimate by averaging the subgoal policies of all samples
    pi = np.zeros((env.nStates, env.nActions))
    for goals in G:
        pi += np.array([goalPolicies[s, :, g] for s, g in enumerate(goals)])
    pi /= len(G)

    return pi



