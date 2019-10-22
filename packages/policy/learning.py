import numpy as np


class QLearning(object):
    """ Q-learning class to be used for RL policies"""

    def __init__(self, nStates=None, nActions=None, Qtable=None, learning_rate=.1, discount=.9):
        """

        Parameters
        ----------
        nStates: int - number of states
        nActions: int - number of actions
        Qtable: np.array - [SxA] Q value table for states and actions
        learning_rate: float 0<= learning_rate <=1 - learning rate parameter for Q-learning
        """
        # Set input values
        self.nStates = nStates
        self.nActions = nActions
        self.learning_rate = learning_rate
        self.discount = discount
        if Qtable is None:  # Initialization
            self.Qtable = np.zeros((self.nStates, self.nActions))
        else:
            self.Qtable = Qtable

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, x):
        try:
            assert 0 <= x <= 1
        except AssertionError:
            raise ValueError('Invalid learning rate')

        self.__learning_rate = x
        return

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, x):
        # in case of empty discount factor
        if x is None:
            self._discount = None
            return

        # assert that the discount is in [0,1)
        if not 0 <= x < 1:
            raise ValueError('discount must be in [0,1)')
        self._discount = x

    @property
    def Qtable(self):
        return self.__Qtable

    @Qtable.setter
    def Qtable(self, x):
        if x is None:
            self.__Qtable = None
            return

        # represent action function as numpy array
        x = np.array(x)

        # check if input is valid state action function
        try:
            assert x.ndim == 2
        except AssertionError:
            raise ValueError('invalid state action function')

        # check consistency with remaining properties
        try:
            if self.nStates is not None and self.nActions is not None:
                assert x.shape == (self.nStates, self.nActions)
            else:
                (self.nStates, self.nActions) = np.shape(x)  # Set number of states if not already set
        except AssertionError:
            raise ValueError('inconsistent number of states and/or actions')

        self.__Qtable = x
        return

    def update(self, curState, action, curRew, nextState):
        """
        Q-learning update with state, action, reward, next state

        Parameters
        ----------
        curState: int - start state
        action: int - action chosen
        curRew: float - reward obtained
        nextState: int - next state
        """

        # Do update of private variable for speed
        self.__Qtable[curState, action] += self.learning_rate * (
                curRew + self.discount * np.max(self.Qtable[nextState, :]) - self.Qtable[curState, action])


