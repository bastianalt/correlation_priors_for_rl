import numpy as np
from abc import ABC, abstractmethod


class Simulator(ABC):
    """
    Base class for a simulator of a transition model
    """

    def __init__(self, nStates, nActions, initialState=None):
        """

        Parameters
        ----------
        nStates - int: number of states for the simulator
        nActions - int: number of actions for the simulator
        initialState - int: initial state of the simulator
        """
        # Save inputs
        self.nStates = nStates
        self.nActions = nActions
        if initialState is None:
            self.lastStates = 0
        else:
            self.lastStates = initialState

    @abstractmethod
    def simulate(self, curStates=None, curActions=None):
        """

        Parameters
        ----------
        curStates - int: current state of the MDP
        curActions - int: action taken in the MDP

        Returns
        -------
        nextState - int: next state of the MDP

        """
        nextStates = None
        return nextStates


class QueuingNetworkSimulator4D(Simulator):
    """ Four Dimensional queuing network (Bernoulli arrivals and servicing)"""

    def __init__(self, B1=19, B2=12, B3=12, B4=19, a1=.08, a3=.08, d1=.12, d2=.12, d3=.28, d4=.28, initialStates=None):
        """

        Parameters
        ----------
        B1 - int: bufferlength of the first queue
        B2 - int: bufferlength of the second queue
        B3 - int: bufferlength of the third queue
        B4 - int: bufferlength of the fourth queue

        a1 - float 0<=x<=1: probability of arrival at queue 1
        a3 - float 0<=x<=1: probability of arrival at queue 3

        d1 - float 0<=x<=1: probability of departure at queue 1
        d2 - float 0<=x<=1: probability of departure at queue 2
        d3 - float 0<=x<=1: probability of departure at queue 3
        d4 - float 0<=x<=1: probability of departure at queue 4
        initialStates - int: starting state of the simulator
        """

        # Initalize simulator
        super().__init__(nStates=(B1 + 1) * (B2 + 1) * (B3 + 1) * (B4 + 1), nActions=4, initialState=initialStates)

        # save input
        self.bufferLengths = dict(B1=B1, B2=B2, B3=B3, B4=B4)
        self.params = dict(a1=a1, a3=a3, d1=d1, d2=d2, d3=d3, d4=d4)

    def simulate_single(self, curState, curAction):
        """
        Simulates one step of the queuing network starting in curState under curAction
        Parameters
        ----------
        curState - starting State
        curAction - action taken

        Returns
        -------
        nextState - new State after single simulation
        """

        # Coding action=0 ==> queue 1 and 2 are served
        #       action=1 ==> queue 1 and 3 are served
        #       action=2 ==> queue 2 and 4 are served
        #       action=3 ==> queue 3 and 4 are served

        # convert state number into four dimensional vector
        vec_tuple = (self.bufferLengths['B1'] + 1, self.bufferLengths['B2'] + 1, self.bufferLengths['B3'] + 1,
                     self.bufferLengths['B4'] + 1)
        x = np.array(np.unravel_index(curState, vec_tuple)).squeeze()

        # Compute dynamics
        # Arrivals at the queues
        A1 = int(np.random.rand() < self.params['a1'])
        A3 = int(np.random.rand() < self.params['a3'])

        # Departures of the queues using curActions
        D1 = D2 = D3 = D4 = 0
        if curAction == 0:
            D1 = int(np.random.rand() < self.params['d1'])
            D2 = int(np.random.rand() < self.params['d2'])
        if curAction == 1:
            D1 = int(np.random.rand() < self.params['d1'])
            D3 = int(np.random.rand() < self.params['d3'])
        if curAction == 2:
            D2 = int(np.random.rand() < self.params['d2'])
            D4 = int(np.random.rand() < self.params['d4'])
        if curAction == 3:
            D3 = int(np.random.rand() < self.params['d3'])
            D4 = int(np.random.rand() < self.params['d4'])

        # Compute dynamics of queuing network
        x_next = x + np.array([A1 - D1, D1 - D2, A3 - D3, D3 - D4])

        # Truncate if new state over floats/ under floats buffer (0<=x<=B)
        x_next[x_next < 0] = 0
        if x_next[0] > self.bufferLengths['B1']: x_next[0] = self.bufferLengths['B1']
        if x_next[1] > self.bufferLengths['B2']: x_next[1] = self.bufferLengths['B2']
        if x_next[2] > self.bufferLengths['B3']: x_next[2] = self.bufferLengths['B3']
        if x_next[3] > self.bufferLengths['B4']: x_next[3] = self.bufferLengths['B4']

        # convert state number into scalar integer state number
        nextState = np.ravel_multi_index(tuple(x_next), vec_tuple)
        return nextState

    def simulate(self, curStates=None, curActions=None):
        """
        Simulate the 4D queuing network

        Parameters
        ----------
        curStates - int: current state of the MDP
        curActions - int: action taken in the MDP

        Returns
        -------
        nextState - int: next state of the MDP

        """

        # Store last state of the simulator
        if curStates is None:
            curStates = self.lastStates

        nTraj = len(curStates)

        if nTraj == 1:
            nextStates = self.simulate_single(curStates, curActions)
        else:
            nextStates = np.zeros(nTraj)
            for n in range(nTraj):
                nextStates[n] = self.simulate_single(curStates[n], curActions[n])
        # Save state internally
        self.lastStates = nextStates
        return nextStates


class QueuingNetworkSimulator4DBatch(Simulator):
    """ Four Dimensional queuing network (Poisson arrivals and servicing)"""

    def __init__(self, B1=19, B2=12, B3=12, B4=19, a1=.08, a3=.08, d1=.12, d2=.12, d3=.28, d4=.28, initialStates=None):
        """

        Parameters
        ----------
        B1 - int: bufferlength of the first queue
        B2 - int: bufferlength of the second queue
        B3 - int: bufferlength of the third queue
        B4 - int: bufferlength of the fourth queue

        a1 - float 0<=x<=1: probability of arrival at queue 1
        a3 - float 0<=x<=1: probability of arrival at queue 3

        d1 - float 0<=x<=1: probability of departure at queue 1
        d2 - float 0<=x<=1: probability of departure at queue 2
        d3 - float 0<=x<=1: probability of departure at queue 3
        d4 - float 0<=x<=1: probability of departure at queue 4
        initialStates - int: starting state of the simulator
        """

        # Initalize simulator
        super().__init__(nStates=(B1 + 1) * (B2 + 1) * (B3 + 1) * (B4 + 1), nActions=4, initialState=initialStates)

        # save input
        self.bufferLengths = dict(B1=B1, B2=B2, B3=B3, B4=B4)
        self.params = dict(a1=a1, a3=a3, d1=d1, d2=d2, d3=d3, d4=d4)

    def simulate_single(self, curState, curAction):
        """
              Simulates one step of the queuing network starting in curState under curAction
              Parameters
              ----------
              curState - starting State
              curAction - action taken

              Returns
              -------
              nextState - new State after single simulation
              """

        # Coding action=0 ==> queue 1 and 2 are served
        #       action=1 ==> queue 1 and 3 are served
        #       action=2 ==> queue 2 and 4 are served
        #       action=3 ==> queue 3 and 4 are served

        # convert state number into four dimensional vector
        vec_tuple = (self.bufferLengths['B1'] + 1, self.bufferLengths['B2'] + 1, self.bufferLengths['B3'] + 1,
                     self.bufferLengths['B4'] + 1)
        x = np.array(np.unravel_index(curState, vec_tuple)).squeeze()

        # Compute dynamics
        # Arrivals at the queues
        A1 = int(np.random.poisson(self.params['a1']))
        A3 = int(np.random.poisson(self.params['a3']))

        # Departures of the queues using curActions
        D1 = D2 = D3 = D4 = 0
        if curAction == 0:
            D1 = int(np.random.poisson(self.params['d1']))
            D2 = int(np.random.poisson(self.params['d2']))
        if curAction == 1:
            D1 = int(np.random.poisson(self.params['d1']))
            D3 = int(np.random.poisson(self.params['d3']))
        if curAction == 2:
            D2 = int(np.random.poisson(self.params['d2']))
            D4 = int(np.random.poisson(self.params['d4']))
        if curAction == 3:
            D3 = int(np.random.poisson(self.params['d3']))
            D4 = int(np.random.poisson(self.params['d4']))

        # Compute dynamics of queuing network
        x_next = x + np.array([A1 - D1, D1 - D2, A3 - D3, D3 - D4])

        # Truncate if new state over floats/ under floats buffer (0<=x<=B)
        x_next[x_next < 0] = 0
        if x_next[0] > self.bufferLengths['B1']: x_next[0] = self.bufferLengths['B1']
        if x_next[1] > self.bufferLengths['B2']: x_next[1] = self.bufferLengths['B2']
        if x_next[2] > self.bufferLengths['B3']: x_next[2] = self.bufferLengths['B3']
        if x_next[3] > self.bufferLengths['B4']: x_next[3] = self.bufferLengths['B4']

        # convert state number into scalar integer state number
        nextState = np.ravel_multi_index(tuple(x_next), vec_tuple)
        return nextState

    def simulate(self, curStates=None, curActions=None):
        """
        Simulate the 4D queuing network

        Parameters
        ----------
        curStates - int: current state of the MDP
        curActions - int: action taken in the MDP

        Returns
        -------
        nextState - int: next state of the MDP

        """

        # Store last state of the simulator
        if curStates is None:
            curStates = self.lastStates

        nTraj = len(curStates)

        if nTraj == 1:
            nextStates = self.simulate_single(curStates, curActions)
        else:
            nextStates = np.zeros(nTraj)
            for n in range(nTraj):
                nextStates[n] = self.simulate_single(curStates[n], curActions[n])
        # Save state internally
        self.lastStates = nextStates
        return nextStates


class QueuingNetworkSimulator2D(Simulator):
    """ Two Dimensional queuing network (Bernoulli arrivals and servicing)"""

    def __init__(self, B1=19, B2=12, a1=.08, d1=.12, d2=.12, initialStates=None):
        """

        Parameters
        ----------
        B1 - int: bufferlength of the first queue
        B2 - int: bufferlength of the second queue

        a1 - float 0<=x<=1: probability of arrival at queue 1

        d1 - float 0<=x<=1: probability of departure at queue 1
        d2 - float 0<=x<=1: probability of departure at queue 2

        initialStates - int: starting state of the simulator
        """

        # Initalize simulator
        super().__init__(nStates=(B1 + 1) * (B2 + 1), nActions=2, initialState=initialStates)

        # save input
        self.bufferLengths = dict(B1=B1, B2=B2)
        self.params = dict(a1=a1, d1=d1, d2=d2)

    def simulate_single(self, curState, curAction):
        """
              Simulates one step of the queuing network starting in curState under curAction
              Parameters
              ----------
              curState - starting State
              curAction - action taken

              Returns
              -------
              nextState - new State after single simulation
              """

        # Coding action=0 ==> queue 1 is served
        #       action=1 ==> queue 2 is served

        # convert state number into four dimensional vector
        vec_tuple = (self.bufferLengths['B1'] + 1, self.bufferLengths['B2'] + 1)
        x = np.array(np.unravel_index(curState, vec_tuple)).squeeze()

        # Compute dynamics
        # Arrivals at the queues
        A1 = int(np.random.rand() < self.params['a1'])

        # Departures of the queues using curActions
        D1 = D2 = 0
        if curAction == 0:
            D1 = int(np.random.rand() < self.params['d1'])
        if curAction == 1:
            D2 = int(np.random.rand() < self.params['d2'])

        # Compute dynamics of queuing network
        x_next = x + np.array([A1 - D1, D1 - D2])

        # Truncate if new state over floats/ under floats buffer (0<=x<=B)
        x_next[x_next < 0] = 0
        if x_next[0] > self.bufferLengths['B1']: x_next[0] = self.bufferLengths['B1']
        if x_next[1] > self.bufferLengths['B2']: x_next[1] = self.bufferLengths['B2']

        # convert state number into scalar integer state number
        nextState = np.ravel_multi_index(tuple(x_next), vec_tuple)
        return nextState

    def simulate(self, curStates=None, curActions=None):
        """
        Simulate the 2D queuing network

        Parameters
        ----------
        curStates - int: current states of the MDP
        curActions - int: actions taken in the MDP

        Returns
        -------
        nextState - int: next state of the MDP

        """

        # Store last state of the simulator
        if curStates is None:
            curStates = self.lastStates

        nTraj = len(curStates)

        if nTraj == 1:
            nextStates = self.simulate_single(curStates, curActions)
        else:
            nextStates = np.zeros(nTraj)
            for n in range(nTraj):
                nextStates[n] = self.simulate_single(curStates[n], curActions[n])

        # Save state internally
        self.lastStates = nextStates
        return nextStates


class QueuingNetworkSimulator2DBatch(Simulator):
    """ Two Dimensional queuing network (Poisson arrivals and servicing)"""

    def __init__(self, B1=19, B2=12, a1=.08, d1=.12, d2=.12, initialStates=None):
        """

        Parameters
        ----------
        B1 - int: bufferlength of the first queue
        B2 - int: bufferlength of the second queue

        a1 - float 0<=x<=1: probability of arrival at queue 1

        d1 - float 0<=x<=1: probability of departure at queue 1
        d2 - float 0<=x<=1: probability of departure at queue 2

        initialStates - int: starting state of the simulator
        """

        # Initalize simulator
        super().__init__(nStates=(B1 + 1) * (B2 + 1), nActions=2, initialState=initialStates)

        # save input
        self.bufferLengths = dict(B1=B1, B2=B2)
        self.params = dict(a1=a1, d1=d1, d2=d2)

    def simulate_single(self, curState, curAction):
        """
              Simulates one step of the queuing network starting in curState under curAction
              Parameters
              ----------
              curState - starting State
              curAction - action taken

              Returns
              -------
              nextState - new State after single simulation
              """

        # Coding action=0 ==> queue 1 is served
        #       action=1 ==> queue 2 is served

        # convert state number into four dimensional vector
        vec_tuple = (self.bufferLengths['B1'] + 1, self.bufferLengths['B2'] + 1)
        x = np.array(np.unravel_index(curState, vec_tuple)).squeeze()

        # Compute dynamics
        # Arrivals at the queues

        A1 = int(np.random.poisson(self.params['a1']))

        # Departures of the queues using curActions
        D1 = D2 = 0
        if curAction == 0:
            D1 = int(np.random.poisson(self.params['d1']))
        if curAction == 1:
            D2 = int(np.random.poisson(self.params['d2']))

        # Compute dynamics of queuing network
        x_next = x + np.array([A1 - D1, D1 - D2])

        # Truncate if new state over floats/ under floats buffer (0<=x<=B)
        x_next[x_next < 0] = 0
        if x_next[0] > self.bufferLengths['B1']: x_next[0] = self.bufferLengths['B1']
        if x_next[1] > self.bufferLengths['B2']: x_next[1] = self.bufferLengths['B2']

        # convert state number into scalar integer state number
        nextState = np.ravel_multi_index(tuple(x_next), vec_tuple)
        return nextState

    def simulate(self, curStates=None, curActions=None):
        """
        Simulate the 2D queuing network

        Parameters
        ----------
        curStates - int: current states of the MDP
        curActions - int: actions taken in the MDP

        Returns
        -------
        nextState - int: next state of the MDP

        """

        # Store last state of the simulator
        if curStates is None:
            curStates = self.lastStates

        nTraj = len(curStates)

        if nTraj == 1:
            nextStates = self.simulate_single(curStates, curActions)
        else:
            nextStates = np.zeros(nTraj)
            for n in range(nTraj):
                nextStates[n] = self.simulate_single(curStates[n], curActions[n])

        # Save state internally
        self.lastStates = nextStates
        return nextStates
