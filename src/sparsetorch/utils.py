"""The methods in this sub-module generate coordinates compatible with this package."""

import torch


def get_equidist_coord(start, end, steps):
    """Returns all coordinates in dimensionwise equidistant grid.

    Parameters
    ----------
    start : list or torch.Tensor
        lower boundaries
    end : list or torch.Tensor
        upper boundaries
    steps : torch.Tensor
        number of steps for coordinate cumputation

    Returns
    -------
    torch.Tensor
        coordinates, shape: `(d, steps)`, where `d` is the dimension,
        i.e., number of elements in `start` and `end`
    """
    assert len(start) == len(end)
    assert len(start) == len(steps)

    d = len(steps)
    n = int(torch.prod(steps).item())

    result = torch.zeros(d, n)

    # compute grids
    coords = []

    for a, b, s in zip(start, end, steps):
        coords.append(torch.linspace(a, b, int(s)))
    all_grids = torch.meshgrid(coords)

    # reshape and store grids
    for i, grid in enumerate(all_grids):
        #result[i] = torch.reshape(grid, (torch.prod(torch.tensor(grid.shape)),))
        result[i] = torch.reshape(grid, (n, ))
    return result


def get_rand_coord(start, end, steps):
    """Returns random coordinates.

    Parameters
    ----------
    start : torch.Tensor
        lower boundaries
    end : torch.Tensor
        upper boundaries
    steps : torch.Tensor
        number of steps for coordinate cumputation

    Returns
    -------
    torch.Tensor
        coordinates, shape: `(d, steps)`, where `d` is the dimension,
        i.e., number of elements in `start` and `end`
    """
    assert len(start) == len(end)
    assert len(start) == len(steps)

    d = len(steps)
    n = int(torch.prod(steps).item())

    result = torch.rand(n, d)
    result = start + (end - start) * result

    return result.T