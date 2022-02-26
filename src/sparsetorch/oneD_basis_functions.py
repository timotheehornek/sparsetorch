"""Find the implementation of single dimensional basis functions in this module. All basis functions inherit from `BF_1D`.
Basis functions defined on mesh and orthogonal basis functions are split into different sub-classes: `BF_1D_mesh` and `BF_1D_orthofun` respectively."""

import math
import torch
import torch.nn.functional as F


class BF_1D(torch.nn.Module):
    """Parent class for implementation of 1D-basis function evaluations as Pytorch layer.

    Attributes
    ----------
    levels : list of int
        contains number of basis function at each level,
        levels are represented by index
    bf_num : int
        total number of basis functions
    """

    def __init__(self, levels):
        """
        Parameters
        ----------
        levels : list of int
            contains number of basis function at each level,
            levels are represented by index
        """
        super().__init__()
        self.levels = levels
        self.bf_num = sum(levels)

    def forward(self, x):
        """Interface method that should be implemented in child class.
        Applies layer to input `x` and returns interpolation matrix.

        Parameters
        ----------
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in all data points, i.e., interpolation matrix
        """
        pass


class BF_1D_mesh(BF_1D):
    """Base class for 1D-basis functions defined on mesh, i.e.,
    defined by position and weight parameters.

    Attributes
    ----------
    mu : torch.nn.parameter.Parameter or torch.Tensor
        position information of basis functions
    h : torch.nn.parameter.Parameter or torch.Tensor
        width information of basis functions
    """

    def __init__(self, mu, h, levels=None):
        """
        Parameters
        ----------
        mu : torch.Tensor
            position information of basis functions
        h : torch.Tensor
            width information of basis functions
        levels : list of int, optional
            number of basis functions at each level,
            entry at index `i` is number of basis functions at level `i`,
            by default None
        """
        assert mu.shape == h.shape
        self.mu = mu
        self.h = h

        if levels is None:
            levels = [len(self.mu)]
        else:
            assert sum(levels) == len(self.mu)
        super().__init__(levels)

    @classmethod
    def hierarchical(cls, level, boundary=False, a=0.0, b=1.0):
        """Alternative constructor that constructs `mu` and `h`
        for hierarchical basis functions.

        Parameters
        ----------
        basis : str
            type of 1D-basis functions to use
        level : int
            maximum level of hierarchical basis
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        a : float, optional
            left boundary of domain, by default 0.0
        b : float, optional
            right boundary of domain, by default 1.0

        Returns
        -------
        BF_1D_mesh
            1D-basis function object
        """
        # index shift caused by boundaries
        shift = int(boundary) * 2

        mu = torch.zeros(2 ** (level + 1) - 1 + shift)
        h = torch.zeros(2 ** (level + 1) - 1 + shift)
        levels = [0] * (level + 1)

        # boundary basis functions at level 0
        if boundary:
            # left
            mu[0] = 0
            h[0] = 1
            # right
            mu[1] = 1
            h[1] = 1
            # update levels
            levels[0] += 2

        for l in range(level + 1):
            levels[l] += 2 ** l
            h_ = 2 ** -l
            for i in range(2 ** l):
                mu[2 ** l + i - 1 + shift] = (i + 0.5) * h_
                h[2 ** l + i - 1 + shift] = h_ / 2

        # scale entries
        mu = a + mu * (b - a)
        h = h * (b - a)
        return cls(mu, h, levels)

    @classmethod
    def equidist(cls, num, a=0.0, b=1.0):
        """Alternative constructor that constructs `mu` and `h`
        for equidistant basis functions.

        Parameters
        ----------
        basis : str
            type of 1D-basis functions to use
        num : int
            number of basis functions
        a : float, optional
            left boundary of domain, by default 0.0
        b : float, optional
            right boundary of domain, by default 1.0

        Returns
        -------
        BF_1D_mesh
            1D-basis function object
        """
        mu = torch.linspace(a, b, num)
        h = torch.ones(num) * ((b - a) / (num - 1))
        return cls(mu, h)


class Gauss(BF_1D_mesh):
    """Implementation of 1-D Gauss basis."""

    def forward(self, x):
        """Overrides interface method and returns tensor with gaussian basis function evaluations
        $e^{-((x-\\mu)/h)^2}$.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """
        x = torch.unsqueeze(x, -1)
        return torch.exp(-torch.square((x - self.mu) / self.h))
    
    def dx(self):
        """Construct first order derivative object.

        Returns
        -------
        _Gauss_dx
            first order derivative object
        """
        return _Gauss_dx(self.mu, self.h, self.levels)
    
    def dxx(self):
        """Construct second order derivative object.

        Returns
        -------
        _Gauss_dxx
            second order derivative object
        """
        return _Gauss_dxx(self.mu, self.h, self.levels)


class _Gauss_dx(Gauss):
    """Implementation of 1-D Gauss basis first order derivative."""

    def forward(self, x):
        """Overrides interface method and returns tensor
        with first order derivatives of gaussian basis function evaluations
        $e^{-((x-\\mu)/h)^2}$.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """
        x = torch.unsqueeze(x, -1)
        return (
            -2
            * ((x - self.mu) / self.h ** 2)
            * torch.exp(-torch.square((x - self.mu) / self.h))
        )


class _Gauss_dxx(Gauss):
    """Implementation of 1-D Gauss basis second order derivative."""

    def forward(self, x):
        """Overrides interface method and returns tensor
        with second order derivatives of gaussian basis function evaluations
        $e^{-((x-\\mu)/h)^2}$.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """
        x = torch.unsqueeze(x, -1)
        return (
            -2
            * (-2 * (x - self.mu) ** 2 + self.h ** 2)
            / self.h ** 4
            * torch.exp(-torch.square((x - self.mu) / self.h))
        )


class Hat(BF_1D_mesh):
    """Implementation of 1-D Hat basis."""

    def forward(self, x):
        """Overrides interface method and returns tensor with hat basis function evaluations
        $max((1-|x-\\mu|/h),0) = ReLU((1-|x-\\mu|/h))$.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """
        x = torch.unsqueeze(x, -1)
        return F.relu(1 - torch.abs(x - self.mu) / self.h)


class BF_1D_orthofun(BF_1D):
    """Base class for 1D-basis functions defined as orthogonal functions.

    Attributes
    ----------
    n_max : int
        maximum degree of basis functions
    data_a : float
        left boundary of domain
    data_w : float
        width of domain
    bf_a : float
        left boundary of basis function domain
    bf_w : float
        width of basis function domain
    """

    def __init__(self, levels, n_max, a, b, bf_a, bf_b):
        """
        Parameters
        ----------
        levels : list of int
            contains number of basis function at each level,
            levels are represented by index
        n_max : int
            maximum degree of basis functions
        a : float
            left boundary of domain
        b : float
            right boundary of domain
        bf_a : float
            left boundary of basis function domain
        bf_b : float
            right boundary of basis function domain
        """
        super().__init__(levels)
        assert n_max >= 0
        self.n_max = n_max
        self.data_a = a
        self.data_b = b
        self.data_w = b - a
        self.bf_a = bf_a
        self.bf_w = bf_b - bf_a

    def scale(self, x):
        """Scales input from input domain to basis function domain.

        Parameters
        ----------
        x : torch.Tensor
            input data to scale

        Returns
        -------
        torch.Tensor
            data scaled to basis function domain
        """
        return (x - self.data_a) / self.data_w * self.bf_w + self.bf_a


class Fourier(BF_1D_orthofun):
    """Implementation of Fourier series as 1-D basis.
    Attributes
    ----------
    trapezoid : bool
        add two basis functions at level 1 for trapezoid (level added)
    """

    def __init__(self, n_max, a=0.0, b=1.0):
        """
        Parameters
        ----------
        n_max : int
            maximum level in series
        a : float, optional
            left boundary of domain, by default 0.0
        b : float, optional
            right boundary of domain, by default 1.0
        """
        levels = [1]
        levels.extend([2] * n_max)
        super().__init__(levels, n_max, a, b, bf_a=-math.pi, bf_b=math.pi)
    
    def dx(self):
        """Construct first order derivative object.

        Returns
        -------
        _Fourier_dx
            first order derivative object
        """
        return _Fourier_dx(self.n_max, self.data_a, self.data_b)
    
    def dxx(self):
        """Construct second order derivative object.

        Returns
        -------
        _Fourier_dxx
            second order derivative object
        """
        return _Fourier_dxx(self.n_max, self.data_a, self.data_b)

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of Fourier series elements.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """

        x = self.scale(x)
        eval = torch.empty(self.bf_num, len(x))
        eval[0] = 0.5

        for n in range(1, self.bf_num // 2 + 1):
            eval[2 * n - 1] = torch.cos(n * x)
            eval[2 * n] = torch.sin(n * x)
        return eval.T


class _Fourier_dx(Fourier):
    """Implementation of first order derivative of Fourier series as 1-D basis."""

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of first order derivatives of Fourier series elements.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """

        x_scaled = self.scale(x)
        eval = torch.empty(self.bf_num, len(x))
        eval[0] = 0.0

        for n in range(1, self.bf_num // 2 + 1):
            eval[2 * n - 1] = -n * self.bf_w / self.data_w * torch.sin(n * x_scaled)
            eval[2 * n] = n * self.bf_w / self.data_w * torch.cos(n * x_scaled)
        return eval.T


class _Fourier_dxx(Fourier):
    """Implementation of second order derivative of Fourier series as 1-D basis."""

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of second order derivatives of Fourier series elements.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """

        x_scaled = self.scale(x)
        eval = torch.empty(self.bf_num, len(x))
        eval[0] = 0.0

        for n in range(1, self.bf_num // 2 + 1):
            # apply product rule
            duv = -((n * self.bf_w / self.data_w) ** 2) * torch.cos(n * x_scaled)
            udv = -n * self.bf_w / self.data_w * torch.sin(n * x_scaled)
            eval[2 * n - 1] = duv #+ udv  # u'v+uv'

            # apply product rule
            duv = -((n * self.bf_w / self.data_w) ** 2) * torch.sin(n * x_scaled)
            udv = n * self.bf_w / self.data_w * torch.cos(n * x_scaled)
            eval[2 * n] = duv #+ udv  # u'v+uv'
        return eval.T


class Chebyshev(BF_1D_orthofun):
    """Implementation of Chebyshev orthogonal polynomials as 1-D basis functions."""

    def __init__(self, n_max, a=0.0, b=1.0):
        """
        Parameters
        ----------
        n_max : int
            maximum degree of polynomials
        a : float, optional
            left boundary of domain, by default 0.0
        b : float, optional
            right boundary of domain, by default 1.0
        """
        levels = [1] * (n_max + 1)
        super().__init__(levels, n_max, a, b, bf_a=-1.0, bf_b=1.0)

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of Chebyshev polynomials.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """

        x = self.scale(x)
        eval = torch.empty(self.bf_num, len(x))
        eval[0] = 1
        if self.n_max > 0:
            eval[1] = x
            for n in range(2, self.bf_num):
                # recursive definition not allowed by autograd
                eval[n] = 2 * x * eval[n - 1].clone() - eval[n - 2].clone()
                # eval[n] = torch.cos(n * torch.acos(x))
        return eval.T


class Legendre(BF_1D_orthofun):
    """Implementation of Legendre orthogonal polynomials as 1-D basis functions."""

    def __init__(self, n_max, a=0.0, b=1.0):
        """
        Parameters
        ----------
        n_max : int
            maximum degree of polynomials
        a : float, optional
            left boundary of domain, by default 0.0
        b : float, optional
            right boundary of domain, by default 1.0
        """
        levels = [1] * (n_max + 1)
        super().__init__(levels, n_max, a, b, bf_a=-1, bf_b=1)

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of Legendre polynomials.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in `x`
        """

        x = self.scale(x)
        eval = torch.empty(self.bf_num, len(x))
        eval[0] = 1
        if self.n_max > 0:
            eval[1] = x
            for n in range(2, self.bf_num):
                a = (2 * n - 1) / n
                b = (n - 1) / n
                # cloning required for autograd to work
                eval[n] = a * x * eval[n - 1].clone() - b * eval[n - 2].clone()
        return eval.T
