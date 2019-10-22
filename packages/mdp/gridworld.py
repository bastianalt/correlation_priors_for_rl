import numpy as np
import matplotlib.pyplot as plt
from packages.mdp.mdp import MDP
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from itertools import groupby, cycle
from collections.abc import Iterable


class Gridworld(MDP):

    def __init__(self, R=None, discount=None, motionPatterns=None, motionMethod='stay', walls=None, shape=None,
                 circular=False):

        # initialize reward and discount through base class constructor
        MDP.__init__(self, R=R, discount=discount)

        # store motion specifications
        if motionMethod not in ('stay', 'renormalize'):
            raise ValueError(f"motion method '{motionMethod}' not available")
        self.motionMethod = motionMethod
        self.motionPatterns = motionPatterns
        self.circular = circular

        # construct map
        self.walls_2d = self.constructMap(walls, shape)

    @MDP.T.getter
    def T(self):
        T = MDP.T.fget(self)
        if T is None:
            T, self._Pmaps = self.constructTransitionModel(self.motionPatterns, self.motionMethod, self.circular)
            self._T = T
        return T

    @property
    def Pmaps(self):
        if self._Pmaps is None:
            MDP._T, self._Pmaps = self.constructTransitionModel(self.motionPatterns, self.motionMethod, self.circular)
            return self._Pmaps
        else:
            return None

    @property
    def shape(self):
        return self.walls_2d.shape if self.walls_2d is not None else None

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def nWalls(self):
        return self.walls_2d.sum() if self.walls_2d is not None else None

    @property
    def nStates(self):
        try:
            return self.size - self.nWalls
        except AttributeError:
            return None

    @property
    def nActions(self):
        return self.motionPatterns.shape[0] if self.motionPatterns is not None else None

    @property
    def statePositions(self):
        if self.walls_2d is None:
            return None
        else:
            # get 2d positions of the valid states
            gridIndices = np.c_[np.unravel_index(range(self.size), self.shape)]
            return gridIndices[~self.walls_2d.ravel(), :]

    @property
    def map(self):
        if self.walls_2d is None:
            return None
        else:
            # create map from wall specifications
            m = np.zeros_like(self.walls_2d, dtype=float)
            m[self.walls_2d] = np.nan
            m[~self.walls_2d] = range(self.nStates)

    @property
    def walls(self):
        # set walls as property without setter to forbid manual changes without calling the constructor
        return self.walls

    @property
    def motionPatterns(self):
        return self._motionPatterns

    @motionPatterns.setter
    def motionPatterns(self, x):
        if x is None:
            self._motionPatterns = None
        else:
            # convert to numpy array
            x = np.array(x)

            # assert that the pattern shapes are odd (current state assumed in center)
            try:
                assert np.allclose(x.sum(axis=(1, 2)), 1)
                assert np.all(x >= 0)
                assert np.all(np.mod(x.shape[1:3], 2))
            except AssertionError:
                raise ValueError('invalid motion patterns')

            # store patterns
            self._motionPatterns = x

    def constructMap(self, walls, shape):

        # assert that size information is provided
        if (walls is None) and (shape is None):
            raise ValueError('either shape or walls must be specified')

        # assert that size information is consistent
        if (walls is not None) and (shape is not None) and (np.asarray(walls).shape != shape):
            raise ValueError('inconsistent shape and wall specifications')

        # always convert wall specifications to 2d boolean map
        if walls is None:
            walls_2d = np.full(shape, False)
        else:
            walls = np.array(walls)
            if walls.ndim == 1:
                walls_2d = np.full(self.shape, False)
                walls_2d[np.unravel_index(walls, self.shape)] = True
            else:
                walls_2d = walls

        return walls_2d

    def constructTransitionModel(self, motionPatterns, invalidMovesMethod, circularFlag):

        # convert to numpy array
        motionPatterns = np.array(motionPatterns)

        # containers for the transition model / probability maps
        Pmaps = np.zeros((*self.shape, self.nStates, self.nActions))
        T = np.zeros((self.nStates, self.nStates, self.nActions))

        # boundary conditions for convolution of motion kernel
        boundary = 'wrap' if circularFlag else 'fill'

        # iterate over actions
        for a, pattern in enumerate(motionPatterns):

            # iterate over all possible current states
            for s in range(self.nStates):
                # create a map indicating the current state
                Pmap_before = np.zeros(self.shape)
                Pmap_before[tuple(self.statePositions[s, :])] = 1

                # apply motion pattern and add walls
                Pmap = convolve2d(Pmap_before, pattern, mode='same', boundary=boundary)
                Pmap[self.walls_2d] = np.nan

                # compute leftover probability mass
                validMass = np.nansum(Pmap)
                if invalidMovesMethod == 'renormalize':
                    if validMass == 0:
                        # if no motion possible for the given pattern, stay at current state
                        Pmap_after = Pmap_before
                        Pmap_after[self.walls_2d] = np.nan
                    else:
                        # otherwise, renormalize the leftover probability mass
                        Pmap_after = Pmap / validMass
                elif invalidMovesMethod == 'stay':
                    # shift the invalid probability mass to the current state
                    Pmap_after = Pmap
                    Pmap_after[tuple(self.statePositions[s, :])] += 1 - validMass

                # store current map and extract corresponding transition vector
                Pmaps[:, :, s, a] = Pmap_after
                T[:, s, a] = Pmap_after[~self.walls_2d]

        return T, Pmaps

    def plot(self, values=None, cmap=None):
        fig, ax = plt.subplots()

        if values is None:
            if cmap is None:
                cmap = plt.get_cmap('gray').reversed()
            plt.imshow(self.walls_2d, cmap=cmap)
        elif np.ndim(values) == 2:
            im = np.reshape(values, (*self.shape, -1))
            plt.imshow(im, cmap=cmap)
        else:
            if cmap is None:
                cmap = plt.get_cmap('viridis')
                cmap.set_bad(color='k')
            map = np.full(self.shape, np.nan)
            map[tuple(self.statePositions.T)] = values
            plt.imshow(map, cmap=cmap)
        ax.set_xticks(np.arange(self.shape[1])-0.5, minor=True)
        ax.set_xticks(np.arange(self.shape[1]), minor=False)
        ax.xaxis.grid(True, which='minor', linewidth=2)
        ax.xaxis.grid(False, which='major')
        ax.set_yticks(np.arange(self.shape[0])-0.5, minor=True)
        ax.set_yticks(np.arange(self.shape[0]), minor=False)
        ax.yaxis.grid(True, which='minor', linewidth=2)
        ax.yaxis.grid(False, which='major')

    def plotLabels(self):
        self.plot()
        for s in range(self.nStates):
            pos = self.statePositions[s, :]
            plt.text(pos[1], pos[0], s, horizontalalignment='center', verticalalignment='center')

    def plotTrajectories(self, trajectories, **kwargs):

        # always represent trajectories as nested lists
        if isinstance(trajectories, np.ndarray):
            trajectories = trajectories.tolist()
        if not isinstance(trajectories, Iterable):
            trajectories = [[trajectories]]
        elif not isinstance(trajectories[0], Iterable):
            trajectories = [trajectories]

        # plot the grid world
        self.plot()

        # iterate over all trajectories
        for trajectory in trajectories:
            # remove duplicate successive states
            trajectory = [s for s,_ in groupby(trajectory)]

            # get 2d positions of trajectory states
            positions = self.statePositions[trajectory, :]

            # depending on the trajectory length ...
            if len(trajectory) == 1:
                # plot a single point
                plt.plot(positions[:, 1], positions[:, 0], marker='o', **kwargs)
            elif len(trajectory) == 2:
                # plot a straight line
                plt.plot(positions[:, 1], positions[:, 0], **kwargs)
            else:
                # interpolate along the state positions

                # linear length along the points
                distance = np.cumsum(np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1)))
                distance = np.insert(distance, 0, 0) / distance[-1]

                # interpolator object
                interpolator = interp1d(distance, positions, kind='quadratic', axis=0)

                # interpolation grid and interpolated values
                grid = np.linspace(0, 1, 10 * len(trajectory)-1)
                interpolation = interpolator(grid)

                # plot the interpolated line
                plt.plot(interpolation[:, 1], interpolation[:, 0], **kwargs)

    def plotArrows(self, angles, mags=None, length=0.5, head_width=0.2, head_length=0.2, facecolor='k', **kwargs):

        # if no magnitudes specified, use constant magnitude of 1
        if mags is None:
            mags = cycle([1])

        # convert degrees to radians
        angles_rad = np.deg2rad(angles)

        # plot all arrows
        for pos, ang, mag in zip(self.statePositions, angles_rad, mags):
            if mag == 0:
                continue
            dx = -length * mag * np.sin(ang)
            dy = -length * mag * np.cos(ang)
            plt.arrow(pos[1] - dx/2, pos[0] - dy/2, dx, dy,
                      head_width=mag*head_width, head_length=mag*head_length,
                      length_includes_head=True, facecolor=facecolor, **kwargs)

    def plotPolicy(self, pi, localpolicy2arrow, localpolicy2value=None, relMag=True, values=None, **kwargs):

        # convert local action distributions to arrows / values
        angles, mags = zip(*[localpolicy2arrow(a) for a in pi])
        if localpolicy2value:
            values = [localpolicy2value(a) for a in pi]

        # plot the grid world
        self.plot(values=values, cmap=kwargs.pop('cmap', None))

        # normalize magnitudes
        if relMag:
            mags = np.array(mags, dtype=float)
            mags /= mags.max()

        # plot arrows
        self.plotArrows(angles, mags, **kwargs)
