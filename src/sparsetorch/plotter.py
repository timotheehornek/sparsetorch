"""This sub-module contains basic plotting capability for 1D and 2D functions and is mainly intended for demonstration purposes.
The module is used in the examples provided with the package."""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from sparsetorch.utils import get_equidist_coord


def plot(model, f_target_, name='plot', a=0, b=1, steps=100):
    """Plots target function along with approximation.

    Parameters
    ----------
    model : Model
        function approximation model
    f_target_ : function
        target function, takes scalar
    name : str, optional
        name of plot, by default 'plot'
    a : int, optional
        left border of evaluation boundary, by default 0
    b : int, optional
        right border of evaluation boundary, by default 1
    steps : int, optional
        steps for discretisation of visualisation, by default 100
    """
    f_target = np.vectorize(f_target_)
    x = torch.linspace(a, b, steps)

    model_approx = torch.flatten(model(x))

    fig = plt.figure()
    plt.plot(x.numpy(), f_target(x), '-')
    plt.plot(x.numpy(), model_approx.detach().numpy(), 'r--')
    plt.title(name)
    # plt.savefig(name+'.png')
    plt.show()
    plt.close()


def plot_3D(f,
            name='plot',
            x_min=0,
            x_max=1,
            y_min=0,
            y_max=1,
            steps=100,
            beautify=False,
            save=False):
    """Plots function.

    Parameters
    ----------
    f : function
        function to plot
    name : str, optional
        name of plot, by default 'plot'
    x_min : int, optional
        left border of evaluation boundary in x direction, by default 0
    x_max : int, optional
        right border of evaluation boundary in x direction, by default 1
    y_min : int, optional
        left border of evaluation boundary in y direction, by default 0
    y_max : int, optional
        right border of evaluation boundary in y direction, by default 1
    steps : int, optional
        steps in each dimension for discretisation of evaluation, by default 100
    beautify : bool, optional
        more beautiful plot if True, takes longer to render, by default False
    save : bool, optional
        save plot if True, by default False
    """
    # create evaluation grid
    x = torch.linspace(x_min, x_max, steps)
    y = torch.linspace(y_min, y_max, steps)
    X, Y = torch.meshgrid(x, y)
    grid_x = torch.reshape(X, (steps**2, ))
    grid_y = torch.reshape(Y, (steps**2, ))

    # initialize input for function and model
    input = torch.empty(2, steps**2)
    input[0] = grid_x
    input[1] = grid_y

    # evaluate function
    Z = f(input)
    Z = torch.reshape(Z, (steps, steps))

    # detach tensors for plotting
    Z = torch.reshape(Z, (steps, steps)).detach()

    # plot real function evaluation
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if not beautify:
        # standard plot
        ax.plot_wireframe(X, Y, Z, color='blue', rcount=10, ccount=10)
    else:
        # alternative more beautiful plot (takes longer to render)
        ax.view_init(41, -26)
        ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap=cm.coolwarm,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)
    if save:
        # save plot
        plt.savefig(name+'.png', dpi=300, pad_inches=0.0, bbox_inches='tight')

    ax.set_title(name)
    plt.show()
    plt.close()


def plot_3D_all(model,
                f_dD,
                name='plot',
                x_min=0,
                x_max=1,
                y_min=0,
                y_max=1,
                steps=100):
    """Plots results of model with 2D basis functions.

    Parameters
    ----------
    model : Model
        function approximation model
    f_dD : function
        target function, takes scalar
    name : str, optional
        name of plot, by default 'plot'
    x_min : int, optional
        left border of evaluation boundary in x direction, by default 0
    x_max : int, optional
        right border of evaluation boundary in x direction, by default 1
    y_min : int, optional
        left border of evaluation boundary in y direction, by default 0
    y_max : int, optional
        right border of evaluation boundary in y direction, by default 1
    steps : int, optional
        steps in each dimension for discretisation of visualisation, by default 100
    """

    title = name+': Real Function Evaluation'
    plot_3D(f_dD, title, x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max, steps=steps)

    title = name+': Model Evaluation'
    plot_3D(model, title, x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max, steps=steps)

    title = name+': Absolute Error'
    plot_3D(lambda x: torch.abs(f_dD(x)-model(x)), title, x_min=x_min,
            x_max=x_max, y_min=y_min, y_max=y_max, steps=steps)

    # compute derivatives
    # generate points
    points = get_equidist_coord(torch.tensor([x_min, y_min]),
                                torch.tensor([x_max, y_max]),
                                steps*torch.ones(2))
    # enable gradient computation
    points.requires_grad = True
    # evaluate model
    approx = model(points)
    # compute derivatives
    approx.backward(torch.ones(steps**2))
    # detach tensor for further use like plotting
    derivative = points.grad.detach()

    title = name+': Model x-Derivative'
    plot_3D(lambda _: derivative[0], title, x_min=x_min,
            x_max=x_max, y_min=y_min, y_max=y_max, steps=steps)
    # note that the lambda function is a workaround to plot the derivative
    # step number has to match step number used for derivatives

    title = name+': Model y-Derivative'
    plot_3D(lambda _: derivative[1], title, x_min=x_min,
            x_max=x_max, y_min=y_min, y_max=y_max, steps=steps)
    # note that the lambda function is a workaround to plot the derivative
    # step number has to match step number used for derivatives