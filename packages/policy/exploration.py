import numpy as np

from packages.mdp.mdp import selectActions
from packages.utils.utils import softmax


def epsilonGreedy(Q, states, epsilon):
    """
    Epsilon-greedy action selection for a given Q-table. For each state, the action probability mass of (1-epsilon)
    is distributed over the optimal actions (those which achieve the maximum Q-value for the given state) while the
    remaining probability mass of epsilon is distributed over the remaining actions.

    :param Q: [S x A] array containing Q-values
    :param states: array of length N containing the query states for which the actions should be selected
    :param epsilon: parameter in [0,1]  (0 means greedy action selection)
    :return:
    """
    # initialize decision table
    decisionTable = np.zeros_like(Q, dtype=float)

    # distribute action probability
    for s, q in enumerate(Q):
        # if all actions are equally good --> uniform
        if np.all(q[0] == q):
            decisionTable[s, :] = 1 / len(q)
            continue

        # row of decision table
        t = np.zeros_like(q, dtype=float)

        # find maximizing actions for the current row and distribute (1-epsilon) probability mass
        maxInds = q == q.max()
        nMaxInds = maxInds.sum()
        t[maxInds] = (1 - epsilon) / nMaxInds

        # distribute remaining probability mass and store row
        t[~maxInds] = epsilon / (len(q) - nMaxInds)
        decisionTable[s, :] = t

    # select and return actions
    return selectActions(decisionTable, states)


def softmaxPolicy(Q, beta):
    """
    Converts an array of Q-values into a softmax policy.

    :param Q: [S x A] array of Q-values
    :param beta: non-negative number specifying the inverse temperature parameter
    :return: [S x A] array representing the corresponding softmax policy
    """
    return softmax(beta * Q, axis=1)