import numpy as np
from abc import ABC, abstractmethod
from packages.utils.utils import sampleDiscrete
from packages.utils.utils import randargmax, configurator


class Policy(ABC, configurator):
    def __init__(self, nStates=None, nActions=None, isStochastic=None):
        self.nStates = nStates
        self.nActions = nActions
        self.isStochastic = isStochastic

    @property
    def isStochastic(self):
        return self.__isStochastic

    @isStochastic.setter
    def isStochastic(self, x):
        if x is None:
            self.__isStochastic = x
            return

        x = bool(x)
        self.__isStochastic = x
        return

    @abstractmethod
    def selectActions(self, states):
        pass

    @abstractmethod
    def update(self, curState, action, curRew, nextState):
        pass


class TabularPolicy(Policy):
    """Policy class for discrete state MDPs"""

    def __init__(self, decisionTable=None, nStates=None, nActions=None):
        """

        Parameters
        ----------
        decisionTable: np.array - [S] dimensional table for deterministic policy
                                - [SxA] dimensional conditional probability table for P(a | s)
        nStates: int - number of states
        nActions: int - number of actions
        """

        super().__init__(nStates=nStates, nActions=nActions)
        self.decisionTable = decisionTable

    @property
    def decisionTable(self):
        return self._decisionTable

    @decisionTable.setter
    def decisionTable(self, x):
        if x is None:
            self._decisionTable = x
            return

        x = np.array(x)
        try:
            assert x.ndim == 1 or x.ndim == 2  # Check for either deterministic or stochastic policy
        except AssertionError:
            raise ValueError('Invalid decision table')

        if x.ndim == 1:
            try:
                # If deterministic policy check for number of actions to be defined
                assert self.nActions is not None

                # Set properties for consistency
                self.nStates = np.shape(x)[0]
                self.isStochastic = False
            except AssertionError:
                raise ValueError('Unknown number of actions')

        if x.ndim == 2:
            try:
                # If stochastic policy check for well defined probability matrix
                assert np.allclose(x.sum(axis=1), 1)
                assert np.all(x >= 0)

                # Set properties for consistency
                (self.nStates, self.nActions) = np.shape(x)
                self.isStochastic = True
            except AssertionError:
                raise ValueError('Invalid probabilities in decision table')

        self._decisionTable = x
        return

    def selectActions(self, states):
        """
        Selects actions at the specified states.
        :param states: array of arbitrary length specifying the query states
        :return: array of the same size as the query array containing the selected actions
        """
        states = np.array(states)

        if self.isStochastic:
            # stochastic policy
            if states.ndim == 0:
                return sampleDiscrete(self.decisionTable[states, :], axis=0)
            return sampleDiscrete(self.decisionTable[states, :], axis=1)

        else:
            # deterministic policy
            return self.decisionTable[states]

    @abstractmethod
    def update(self, curState, action, curRew, nextState):
        pass

    @abstractmethod
    def get_greedy_policy(self):
        pass


class PlanningPolicy(TabularPolicy):
    """
    Computes the behavior using an internal transition model that gets updated with incoming experience.
    """

    def __init__(self, nStates, nActions, decisionStrategy, planner, beliefTransitionModel, nSamples=1, updateFreq=1):
        """
        :param nStates: number of MDP states
        :param nActions: number of MDP actions
        :param decisionStrategy: function(Q, states) that produces actions for all provided states based on a
            given Q-table
        :param planner: function(T) that returns a Q-table for a given transition model
        :param beliefTransitionModel: Object handling the internal transtion model. Must have a function draw() to
            sample transition models and a function update(s, a, s') to incorporate new experience.
        :param nSamples: Number of samples generated from the internal model for planning.
            If 'None', the posterior mean of the model is used.
        :param updateFreq: integer indicating after how many update calls the behavior is replanned
        """
        # store parameters
        TabularPolicy.__init__(self, nStates=nStates, nActions=nActions)
        self.decisionStrategy = decisionStrategy
        self.planner = planner
        self.modelGenerator = beliefTransitionModel
        self.nModels = nSamples
        self.updateFreq = updateFreq

        # initialization
        self.nUpdateCalls = 0
        self.meanQtable = self.computeQtable()

    def computeQtable(self):
        if self.nModels is None:
            model = self.modelGenerator.mean()
            Qtable = self.planner(model)
        else:
            models = [self.modelGenerator.draw() for _ in range(self.nModels)]
            Qtable = np.array([self.planner(m) for m in models]).mean(axis=0)
        return Qtable

    def selectActions(self, states):
        return self.decisionStrategy(self.meanQtable, states)

    def update(self, currState, currAction, currReward, nextState):
        self.nUpdateCalls += 1
        self.modelGenerator.update(nextState, currState, currAction)
        if self.nUpdateCalls == self.updateFreq:
            self.meanQtable = self.computeQtable()
            self.nUpdateCalls = 0

    def get_greedy_policy(self):
        return np.argmax(self.meanQtable, axis=1)


class ModelFreePolicy(TabularPolicy):
    """
    Class for a model free policy
    """

    def __init__(self, nStates, nActions, decisionStrategy, learning_module):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        decisionStrategy - callback function with inputs Qtable, states - returns action
        learning_module - learning module with method update(curState, action, curRew, nextState) and property Qtable
        """

        # store parameters
        super().__init__(nStates=nStates, nActions=nActions)
        self.decisionStrategy = decisionStrategy
        self.learning_module = learning_module
        self.Qtable = self.learning_module.Qtable

    def update(self, curState, action, curRew, nextState):
        """
               Qtable update with state, action, reward, next state

              Parameters
              ----------
              curState: int - start state
              action: int - action chosen
              curRew: float - reward obtained
              nextState: int - next state
              """
        self.learning_module.update(curState, action, curRew, nextState)
        self.Qtable = self.learning_module.Qtable

    def selectActions(self, states):
        return self.decisionStrategy(self.Qtable, states)

    def get_greedy_policy(self):
        return randargmax(self.Qtable, axis=1)


