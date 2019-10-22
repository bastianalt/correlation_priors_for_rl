import packages.mdp.gridworld as gw
import numpy as np


def test_mapcreation_booleanWalls():
    walls = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])
    walls = walls.astype('bool')
    world = gw.Gridworld(walls)
    desired = np.array([[0, np.nan, 1], [2, 3, 4], [5, 6, np.nan]])
    assert np.allclose(world.map, desired, equal_nan=True)

def test_mapcreation_integerWalls():
    walls = [1, 8]
    dims = [3, 3]
    world = gw.Gridworld(walls, dims)
    desired = np.array([[0, np.nan, 1], [2, 3, 4], [5, 6, np.nan]])
    assert np.allclose(world.map, desired, equal_nan=True)

def test_mapcreation_noWalls():
    dims = [2, 3]
    world = gw.Gridworld(shape=dims)
    desired = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.array_equal(world.map, desired)
