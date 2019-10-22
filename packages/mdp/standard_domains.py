import numpy as np
from scipy.stats import multivariate_normal

from packages.mdp.gridworld import Gridworld
from packages.mdp.mdp import MDP
from packages.mdp.simulator import QueuingNetworkSimulator2D, QueuingNetworkSimulator4D, QueuingNetworkSimulator2DBatch, \
    QueuingNetworkSimulator4DBatch


class GoalGridworld(Gridworld):
    """
    Simple 4-directional Gridworld with deterministic transitions and a specified number of uniformly placed rewards.
    """

    def __init__(self, nRewards, dist=np.random.rand, discount=0.9, **kwargs):
        motionPatterns = [np.rot90([[0, 0.7, 0], [0.1, 0, 0.1], [0, 0.1, 0]], r) for r in range(4)]
        Gridworld.__init__(self, discount=discount, motionPatterns=motionPatterns, **kwargs)
        rewardStates = np.random.choice(range(self.nStates), nRewards, replace=False)
        rewards = dist(nRewards)
        R = np.zeros(self.nStates)
        R[rewardStates] = rewards
        self.R = R


class DenseCardinalGridworld(Gridworld):
    """
    Gridworld with the four cardinal motion directions.

    The transition model is generally dense but the connectivity of states depends on the specified accuracy of the
    agent's moves, i.e., for high accuracy the agent will transitions to the desired state with high probability and
    reach neighboring states of the target state only occasionally, while for low accuracy the spread to neighboring
    states is high.
    """

    def __init__(self, shape=(10, 10), step_size=1, step_spread=1, **kwargs):
        """
        :param shape: dimensions of the gridworld
        :param step_size: mean step size (shift of the Gaussian from the current state)
        :param step_spread: spread of the step (std of the Gaussian)
        """
        # convert shape parameter to numpy array
        shape = np.array(shape)

        # round size of motion pattern to the next odd integer so that current state can be placed at center
        patternShape = shape // 2 * 2 + 1

        # construct the basic motion pattern for one direction
        dist = multivariate_normal(patternShape // 2 - np.array([0, step_size]), cov=step_spread ** 2)
        y, x = np.mgrid[0:patternShape[0], 0:patternShape[1]]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        basicPattern = dist.pdf(pos)
        basicPattern /= basicPattern.sum()

        # create all four cardinal motions
        motionPatterns = [np.rot90(basicPattern, r) for r in range(4)]

        # call constructor of super class with the respective parameters
        Gridworld.__init__(self, motionPatterns=motionPatterns, shape=shape, **kwargs)


class TunnelWorld(DenseCardinalGridworld):
    def __init__(self, shape=(5, 5), tunnels=((6, 18),), R=None, discount=0.9, **kwargs):
        DenseCardinalGridworld.__init__(self, shape, R=R, discount=discount, **kwargs)
        for tunnel in tunnels:
            start, end = tunnel
            self.T = self._tunnelState(start, end)


class DiagonalTunnelWorld(TunnelWorld):
    def __init__(self, length=10, corner=False, **kwargs):
        assert length >= 2
        x = int((length + 1) / 4)
        if corner:
            startPos = [0, 0]
            goalPos = [length - 1] * 2
        else:
            startPos = [x, x]
            goalPos = [length - x - 1] * 2
        startState = np.ravel_multi_index(startPos, (length, length))
        goalState = np.ravel_multi_index(goalPos, (length, length))
        R = np.zeros(length ** 2)
        R[goalState] = 1

        TunnelWorld.__init__(self, shape=(length, length), tunnels=((goalState, startState),), R=R, **kwargs)


class QueuingNetwork2DBatch(MDP):
    """2Dimensional Queuing Network with batch arrivals and processing"""

    def __init__(self, B1=100, B2=100, a1=.08, d1=.12, d2=.12, initialStates=0, **kwargs):
        T = QueuingNetworkSimulator2DBatch(B1=B1, B2=B2, a1=a1, d1=d1, d2=d2, initialStates=initialStates)
        vec_tuple = (B1 + 1, B2 + 1)
        R = -np.sum(np.array(np.unravel_index(np.arange(0, T.nStates), vec_tuple)), axis=0)
        super().__init__(T, R, discount=.9, **kwargs)
