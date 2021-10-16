import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from sparsetorch.dD_basis_functions import Tensorprod, Elemprod, Sparse
from sparsetorch.oneD_basis_functions import Hat, Gauss, Fourier, Chebyshev, Legendre
from sparsetorch.plotter import plot_3D_all
from sparsetorch.utils import get_equidist_coord, get_rand_coord
from sparsetorch.solver import Model, Solver


def f_dD(x):
    """Simple example function defined on interval `[0, 1]`

    Parameters
    ----------
    x : torch.Tensor
        coordinates for evaluation

    Returns
    -------
    torch.Tensor
        function evaluations
    """
    result = 4 * x[0] * (x[0] - 1)
    for x_i in x[1:]:
        result *= 4 * x_i * (x_i - 1)
    result *= torch.exp(2 * torch.prod(x, dim=0))
    return result


def g_dD(x):
    """Complicated example function defined on interval `[0, 6]`

    Parameters
    ----------
    x : torch.Tensor
        coordinates for evaluation

    Returns
    -------
    torch.Tensor
        function evaluations
    """
    result = x[0] * (x[0] - 6) / 9
    for x_i in x[1:]:
        result *= x_i * (x_i - 6) / 9
    result *= torch.exp(torch.sin(torch.prod(x, dim=0)))
    return result


def step_dD(x):
    """Another example function defined on interval `[0, 1]`, discontinuous

    Parameters
    ----------
    x : torch.Tensor
        coordinates for evaluation

    Returns
    -------
    torch.Tensor
        function evaluations
    """
    result = 1.0
    for x_i in x:
        result *= torch.round(2 * x_i)
    return result


def example_1():
    """Example with same equidistant basis functions in 2D and tensorprod combination"""
    #############
    # settings: #
    #############
    # basis function settings
    basis = Gauss  # Hat or Gauss
    bf_num = 30  # number of basis functions in one dimension
    BF_dD = Tensorprod  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 100  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = f_dD(input)
    #############

    # create 1D basis with equidistant basis functions
    bf_1D = basis.equidist(bf_num)
    bfs_1D = [bf_1D] * 2

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, f_dD, "Example 1")


def example_2():
    """Example with different equidistant basis functions in 2D,
    tensorprod combination and different number of basis functions 
    in different dimensions"""
    #############
    # settings: #
    #############
    # basis function settings
    basis_x = Hat  # Hat or Gauss
    basis_y = Gauss  # Hat or Gauss
    bf_num_x = 7  # number of basis functions in x direction
    bf_num_y = 3  # number of basis functions in y direction
    BF_dD = Tensorprod  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num_x = 50  # number of function evaluations in x direction
    eval_num_y = 60  # number of function evaluations in y direction
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.tensor([eval_num_x, eval_num_y]))
    # function evaluations
    target = f_dD(input)
    #############

    # create 1D basis with equidistant basis functions
    bf_1D_x = basis_x.equidist(bf_num_x)
    bf_1D_y = basis_y.equidist(bf_num_y)
    bfs_1D = [bf_1D_x, bf_1D_y]

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, f_dD, "Example 2")


def example_3():
    """Example with custom basis functions and elemprod combination"""
    #############
    # settings: #
    #############
    # basis function settings
    basis_x = Hat  # Hat or Gauss
    basis_y = Gauss  # Hat or Gauss
    bf_num = 50  # number of basis functions
    torch.manual_seed(332)
    # position and width parameters of basis functions
    mu_x = torch.rand(bf_num)
    h_x = torch.rand(bf_num)
    mu_y = torch.rand(bf_num)
    h_y = torch.rand(bf_num)
    BF_dD = Elemprod  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 60  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = f_dD(input)
    #############

    # create 1D basis with equidistant basis functions
    bf_1D_x = basis_x(mu_x, h_x)
    bf_1D_y = basis_y(mu_y, h_y)
    bfs_1D = [bf_1D_x, bf_1D_y]

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, f_dD, "Example 3")


def example_4():
    """Example with same hierarchical basis functions in 2D, sparse combination
    and approximated function nonzero on boundary
    """
    #
    #############
    # settings: #
    #############
    # basis function settings
    basis = Hat  # Hat or Gauss
    level = 5  # highest level of basis functions in one dimension
    BF_dD = Sparse  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 100  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = step_dD(input)
    #############

    # create 1D basis with hierarchical basis functions
    bf_1D = basis.hierarchical(level, boundary=True)
    bfs_1D = [bf_1D] * 2

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, step_dD, "Example 4")


def example_5():
    """Example with hierarchical basis functions in 2D, sparse combination
    and approximated function nonzero on boundary
    """
    #############
    # settings: #
    #############
    # basis function settings
    basis = Hat  # Hat or Gauss
    level_x = 4  # highest level of basis functions in x direction
    level_y = 5  # highest level of basis functions in y direction
    BF_dD = Sparse  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 100  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = step_dD(input)
    #############

    # create 1D basis with hierarchical basis functions
    bf_1D_x = basis.hierarchical(level_x, boundary=True)
    bf_1D_y = basis.hierarchical(level_y, boundary=True)
    bfs_1D = [bf_1D_x, bf_1D_y]

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, step_dD, "Example 5")


def example_6():
    """Example with orthogonal basis functionsin 2D, sparse combination
    and approximated function nonzero on boundary
    """
    #
    #############
    # settings: #
    #############
    # basis function settings
    basis = Chebyshev  # Fourier, Chebyshev, or Legendre
    n_max = 40  # maximum level of basis functions
    BF_dD = Sparse  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 100  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = step_dD(input)
    #############

    # create 1D basis with orthogonal basis functions
    bfs_1D = [basis(n_max)] * 2

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot_3D_all(model, step_dD, "Example 6")


def example_7():
    """Example with challenging function, orthogonal basis functions,
    sparse combination and approximated function nonzero on boundary
    """
    #############
    # settings: #
    #############
    # basis function settings
    basis = Fourier  # Fourier, Chebyshev, or Legendre
    n_max = 16  # maximum level of basis functions
    BF_dD = Sparse  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 100  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), 6 * torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = g_dD(input)
    #############

    # create 1D basis with orthogonal basis functions
    bfs_1D = [basis(n_max, a=0.0, b=6.0)] * 2

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares with regularization
    solver.le()

    # plot
    plot_3D_all(
        model,
        g_dD,
        "Example 7",
        x_min=0,
        x_max=6,
        y_min=0,
        y_max=6,
        steps=2 * eval_num,
    )


def example_8():
    """Example with challenging function, hierarchical basis functions,
    sparse combination and approximated function nonzero on boundary
    """
    #############
    # settings: #
    #############
    # basis function settings
    basis = Hat  # Hat or Gauss
    level = 8  # highest level of basis functions in one dimension
    BF_dD = Sparse  # Tensorprod, Elemprod, or Sparse
    # evaluation coordinates
    eval_num = 150  # number of function evaluations in one dimension
    input = get_equidist_coord(torch.zeros(2), 6 * torch.ones(2),
                               torch.ones(2) * eval_num)
    # function evaluations
    target = g_dD(input)

    # create 1D basis with hierarchical basis functions
    bf_1D = basis.hierarchical(level, boundary=False, a=0, b=6)
    bfs_1D = [bf_1D] * 2

    # create dD basis with above declared 1D basis functions
    bf_dD = BF_dD(bfs_1D)

    # create model
    model = Model(bf_dD, bf_dD.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares with regularization
    solver.le()

    # plot
    plot_3D_all(
        model,
        g_dD,
        "Example 8",
        x_min=0,
        x_max=6,
        y_min=0,
        y_max=6,
        steps=2 * eval_num,
    )


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_5()
    example_6()
    example_7()
    example_8()