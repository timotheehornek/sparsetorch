import torch

from sparsetorch.oneD_basis_functions import Hat, Gauss, Fourier, Chebyshev, Legendre
from sparsetorch.solver import Model, Solver
from sparsetorch.plotter import plot
import math


def f(x):
    """Example function 1

    Parameters
    ----------
    x : float
        function argument

    Returns
    -------
    float
        function value in `x`
    """
    return -8 * x * (x - 1)


def g(x):
    """Example function 2

    Parameters
    ----------
    x : float
        function argument

    Returns
    -------
    float
        function value in `x`
    """
    x = abs(x - .5)
    if x < .1:
        return 2 + x
    elif x < .3:
        return 2.1 + (x - .1)**2
    else:
        return -(x - .3) + 2.14


def h(x):
    """Example function 3

    Parameters
    ----------
    x : float
        function argument

    Returns
    -------
    float
        function value in `x`
    """
    return math.exp(math.sin(x * x))


def example_1():
    """Example with equidistant basis functions"""
    #############
    # settings: #
    #############
    basis = Gauss  # Hat or, Gauss
    bf_num = 5  # number of basis functions
    eval_num = 60  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 1, eval_num)
    # function evaluations
    target = torch.linspace(0, 1, eval_num).apply_(f)
    #############

    # create 1D basis with equidistant basis functions
    bf = basis.equidist(bf_num)

    # create model
    model = Model(bf, bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot(model, f, name='Example 1')


def example_2():
    """Example with hierarchical basis functions"""
    #############
    # settings: #
    #############
    basis = Hat  # Hat or, Gauss
    level = 1  # highest level of basis functions
    eval_num = 60  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 1, eval_num)
    # function evaluations
    target = torch.linspace(0, 1, eval_num).apply_(f)
    #############

    # compute number of basis functions
    bf_num = 2**(level + 1) - 1

    # create 1D basis with hierarchical basis functions
    bf = basis.hierarchical(level)

    # create model
    model = Model(bf, bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot(model, f, name='Example 2')


def example_3():
    """Example with custom basis functions"""
    #############
    # settings: #
    #############
    basis = Gauss  # Hat or, Gauss
    bf_num = 5  # number of basis functions
    torch.manual_seed(123)
    mu = torch.rand(bf_num)  # centers of basis functions
    h = torch.rand(bf_num)  # width parameters of basis functions
    eval_num = 50  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 1, eval_num)
    # function evaluations
    target = torch.linspace(0, 1, eval_num).apply_(f)
    #############

    # create 1D basis with custom basis functions
    bf = basis(mu, h)

    # create model
    model = Model(bf, bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot(model, f, name='Example 3')


def example_4():
    """Example with equidistant basis functions and SGD optimizer"""
    #############
    # settings: #
    #############
    basis = Gauss  # Hat or, Gauss
    bf_num = 5  # number of basis functions
    eval_num = 50  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 1, eval_num)
    # function evaluations
    target = torch.linspace(0, 1, eval_num).apply_(f)
    # optimizer settings
    criterion = torch.nn.MSELoss()
    lr = 0.01  # learning rate
    momentum = 0.9  # momentum factor
    eps = 10e-4  # threshold for loss
    #############

    # create 1D basis with equidistant basis functions
    bf = basis.equidist(bf_num)

    # create model
    model = Model(bf, bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve with sgd optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    solver.general(criterion, optimizer, eps)

    # plot
    plot(model, f, name='Example 4')


def example_5():
    """Example with orthogonal basis functions"""
    #############
    # settings: #
    #############
    basis = Fourier  # Fourier, Chebyshev, or Legendre
    n_max = 10  # maximum level of basis functions
    eval_num = 60  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 1, eval_num)
    # function evaluations
    target = torch.linspace(0, 1, eval_num).apply_(g)
    #############

    # create 1D basis with orthogonal functions
    bf = basis(n_max)

    # create model
    model = Model(bf, bf.bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot(model, g, name='Example 5', steps=400)


def example_6():
    """Another example with equidistant basis functions and challenging example function"""
    #############
    # settings: #
    #############
    basis = Gauss  # Hat or, Gauss
    bf_num = 30  # number of basis functions
    eval_num = 300  # number of function evaluations
    # evaluation coordinates
    input = torch.linspace(0, 5, eval_num)
    # function evaluations
    target = torch.linspace(0, 5, eval_num).apply_(h)
    #############

    # create 1D basis with equidistant basis functions
    bf = basis.equidist(bf_num, a=0., b=5.)

    # create model
    model = Model(bf, bf_num)

    # create solver
    solver = Solver(model, input, target)

    # solve linear equation / least squares
    solver.le()

    # plot
    plot(model, h, name='Example 6', a=0., b=5., steps=3 * eval_num)


if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()
    example_5()
    example_6()