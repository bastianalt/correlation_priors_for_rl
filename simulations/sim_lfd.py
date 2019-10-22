import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

from packages.mdp.standard_domains import Gridworld, GoalGridworld
from packages.mdp.mdp import MDP, normalizeQ, policyDivergence, det2stoch
from packages.policy.exploration import softmaxPolicy, softmax
from packages.pgmult.pg_mult_model import PGinfer, PG_infer_var
from packages.pgmult.pg_cov import NegExponential
from packages.utils.utils import distmat2covmat, randargmax, positions2distmat, hellinger

np.random.seed(7)


def generate_environment(env):
    _, constructor, params = env
    return constructor(**params)


def generate_lfd_task(mdp: MDP, beta, nTrajs, nSteps):
    # compute optimal policy, Q-values and corresponding advantage values
    _, Q, pi_opt = mdp.policyIteration()
    # Q = normalizeQ(Q)

    # compute softmax policy
    pi_dem = softmaxPolicy(Q, beta)

    # generate demonstration set
    trajs = mdp.sampleTrajectories(nTrajs, nSteps, pi_dem, dropLastState=True)
    S, A = trajs['states'], trajs['actions']

    return pi_dem, pi_opt, S, A


def dir_mean(mdp: MDP, S, A, alpha):
    D = mdp.trajs2dems(S, A)
    return (D + alpha) / (D + alpha).sum(axis=1, keepdims=True)


def pg_mean(mdp: Gridworld, S, A):
    D = mdp.trajs2dems(S, A)
    distmat = positions2distmat(mdp.statePositions)
    Sigma = NegExponential(distmat)
    pi_hat = PG_infer_var(D, Sigma)
    return pi_hat


def dist2angle(dist):
    dirs = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
    ds = dist @ dirs
    mag = np.linalg.norm(ds)
    angle = np.rad2deg(np.arctan2(-ds[1], ds[0]))
    return angle, mag


def dist2color(dist):
    angle, mag = dist2angle(dist)
    cmap = cm.get_cmap('hsv')
    color = list(cmap((angle+180) / 360))
    color[3] = mag
    return color


localpolicy2arrow = lambda dist: dist2angle(dist)
localpolicy2value = None


def visualize(envs, agent, methods):

    longest_name = np.array([len(m[0]) for m in methods]).max()
    hell_dist = []
    emd = []
    value_loss = []
    cmap = matplotlib.colors.ListedColormap([[1]*3, [0.8]*3, sns.color_palette()[2]])

    for env in envs:

        # construct MDP and generate demonstration data
        mdp = generate_environment(env)
        rewardStates = np.flatnonzero(mdp.R)
        pi_dem, pi_opt, S, A = generate_lfd_task(mdp, **agent)
        values = np.array([2 if s in rewardStates else 0 for s in range(mdp.nStates)])
        mdp.plotPolicy(pi_dem, localpolicy2arrow, values=values, cmap=cmap)
        plt.savefig('../evaluation/plots/lfd/expert.pdf',
                    bbox_inches='tight')
        plt.show()

        values = np.array([2 if s in rewardStates else 1 if s in S else 0 for s in range(mdp.nStates)])

        for method_name, method, method_params in methods:
            pi_hat = method(mdp, S, A, **method_params)
            rec_err = policyDivergence(pi_dem, pi_hat, mdp=mdp, method='EMD').mean()
            vl = policyDivergence(pi_dem, pi_hat, mdp=mdp, method='VL')
            value_loss.append(vl)
            hd = hellinger(pi_dem, pi_hat)
            hell_dist.append(hd)
            print(f"{method_name.ljust(longest_name)} -- " +
                f"emd: {rec_err:.3f} -- " +
                f"value loss: {vl:.3f} -- " +
                f"hellinger: {hd:.3f}")
            mdp.plotPolicy(pi_hat, localpolicy2arrow, values=values, cmap=cmap)
            plt.savefig('../evaluation/plots/lfd/' + method_name + '.pdf',
                        bbox_inches='tight')
            plt.show()

    return hell_dist, value_loss

if __name__ == '__main__':

    plt.style.use('default')
    pg_samples = 200
    pg_burnin = 100

    envs = (
        ('10x10_Gridworld', GoalGridworld, dict(nRewards=3, shape=(10, 10), dist=np.ones)),
    )

    agent = dict(
        nTrajs=50,
        nSteps=1,
        beta=20,
    )

    methods = (
        ('Dirichlet', dir_mean, dict(alpha=1e-10)),
        ('PG', pg_mean, dict()),
    )

    hell_dist, value_loss = visualize(envs, agent, methods)
    names = [m[0] for m in methods]

    plt.style.use('seaborn')
    df = pd.DataFrame({'Hellinger':hell_dist, 'value loss':value_loss}, index=names)
    df_norm = df / df.max()
    df_norm.T.plot.barh(color=['tab:blue', 'tab:orange'])
    # plt.gca().get_xaxis().set_visible(False)
    plt.xlim([0, 1])
    plt.gcf().set_size_inches(0.7*np.array((5, 3)))
    plt.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, borderaxespad=0., mode='expand')
    plt.savefig('../evaluation/plots/lfd/metric.pdf', bbox_inches='tight')
    plt.show()

