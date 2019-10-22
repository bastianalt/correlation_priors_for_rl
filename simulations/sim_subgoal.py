import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from packages.subgoal.inference import estimatePolicy
from packages.mdp.standard_domains import GoalGridworld
from packages.mdp.mdp import normalizeQ, policyDivergence
from packages.policy.exploration import softmax
from packages.pgmult.pg_mult_model import NegExponential, PG_infer_var
from packages.utils.utils import positions2distmat, hellinger

np.random.seed(0)

def dist2angle(dist):
    dirs = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
    ds = dist @ dirs
    mag = np.linalg.norm(ds)
    angle = np.rad2deg(np.arctan2(-ds[1], ds[0]))
    return angle, mag

localpolicy2arrow = lambda dist: dist2angle(dist)
cmap = mpl.colors.ListedColormap([[1] * 3, [0.8] * 3, sns.color_palette()[2]])
cmap.set_bad(color='k')

# generate environment / lfd task
walls = np.full((11, 11), fill_value=False)
walls[np.r_[0:7, 8:11], 3] = 1
walls[np.r_[0:3, 4:11], 7] = 1
mdp = GoalGridworld(0, walls=walls)
R = np.zeros(mdp.nStates)
R[[10, 90]] = 1
mdp.R = R
data = np.zeros((0, 2))

data = np.r_[1, 19, 28, 38, 47, 56, 65:70, 59, 41, 31:36, 44, 53, 62, 72, 81, 99]
data = np.c_[data, [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 0]]

beta = 5
V, Q, pi = mdp.policyIteration()
Q_norm = normalizeQ(Q)
pi_soft = softmax(beta * Q_norm, axis=1)


pi_data = np.full([mdp.nStates, mdp.nActions], fill_value=1/mdp.nActions)
for s, a in data:
    pi_data[s] = 0
    pi_data[s, a] = 1
mdp.plotPolicy(pi_data, localpolicy2arrow, values=mdp.R, cmap=cmap)
plt.savefig(f"../evaluation/plots/subgoal/data.pdf", bbox_inches='tight')
plt.show()

D = mdp.trajs2dems(*data.T)
distmat = positions2distmat(mdp.statePositions)
Sigma = NegExponential(distmat)
pi_hat_pg_action = PG_infer_var(D, Sigma)

pi_hat_pg_sg = estimatePolicy(data, mdp, 'pg', beta=5, nSamples=200, nBurnin=100)
pi_hat_dir_sg = estimatePolicy(data, mdp, 'dir', beta=5, nSamples=200, nBurnin=100)
pi_hat_dir_action = (D+0.001)/np.sum(D+0.001, axis=1, keepdims=True)

mdp.plotPolicy(pi_soft, localpolicy2arrow, values=mdp.R, cmap=cmap)
plt.savefig(f"../evaluation/plots/subgoal/expert.pdf", bbox_inches='tight')
plt.show()

cmap = mpl.cm.get_cmap('Reds')
cmap.set_bad(color='k')
error = []
loss = []
for pi in [pi_hat_pg_sg, pi_hat_pg_action, pi_hat_dir_sg, pi_hat_dir_action]:
    V = mdp.policyEvaluation(pi)
    loss.append(policyDivergence(pi_soft, pi, 'VL', mdp))
    error.append([hellinger(x, y) for x, y in zip(pi, pi_soft)])
    print(f"V: {np.mean(V)},  E: {np.mean(error[-1])},  L: {loss[-1]}")

for i, (pi, label) in enumerate(zip([pi_hat_pg_sg, pi_hat_pg_action, pi_hat_dir_sg, pi_hat_dir_action],
                                    ['pg_sg', 'pg_action', 'dir_sg', 'dir_action'])):
    mdp.plotPolicy(pi, localpolicy2arrow, values=error[i]/np.max(error), cmap=cmap)
    plt.title(f"mean Hellinger distance: {np.mean(error[i]):.2f}", fontsize=16)
    plt.savefig(f"../evaluation/plots/subgoal/{label}.pdf", bbox_inches='tight')
    plt.show()