"""Find the implementation of different B-splines in this module. All basis functions inherit from `BF_1D`."""

import functools
import math
import torch
from sparsetorch.oneD_basis_functions import BF_1D


class Splines(BF_1D):
    """Parent class for implementation of 1D B-spline evaluations as Pytorch layer.

    Attributes
    ----------
    data_a : float
        left boundary of domain
    data_w : float
        width of domain
    """

    def __init__(self, levels, a=0.0, b=1.0):
        """
        Parameters
        ----------
        levels : list of int
            contains number of basis function at each level,
            levels are represented by index
        a : float
            left boundary of domain
        b : float
            right boundary of domain
        """
        super().__init__(levels)
        self.data_a = a
        self.data_b = b
        self.data_w = b - a

    def _scale(self, xi):
        """Scales knot sequence from unit interval to input interval.

        Parameters
        ----------
        xi : torch.Tensor
            knot sequence

        Returns
        -------
        torch.Tensor
            knots scaled to data interval
        """
        return xi * self.data_w + self.data_a

    @functools.lru_cache(maxsize=128, typed=False)
    def _eval_b_spline(self, n, k, xi, x):
        """Evaluate standard uniform B-splines following Cox-de-Boor recursion.

        Parameters
        ----------
        n : int
            degree
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of standard uniform B-splines
        """
        if n == 0:
            condition = torch.logical_and(xi[k] <= x, x < xi[k + 1])
            return torch.where(
                condition,
                torch.ones_like(x),
                torch.zeros_like(x),
            )

        a = (x - xi[k]) / (xi[k + n] - xi[k])
        b = (xi[k + n + 1] - x) / (xi[k + n + 1] - xi[k + 1])
        result = a * self._eval_b_spline(n - 1, k, xi, x)
        result += b * self._eval_b_spline(n - 1, k + 1, xi, x)
        return result

    def _eval_b_spline_dx(self, n, k, xi, x):
        """Evaluate derivative of standard uniform B-splines.

        Parameters
        ----------
        n : int
            degree
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of derivative of standard uniform B-splines
        """
        result = n / (xi[k + n] - xi[k]) * self._eval_b_spline(n - 1, k, xi, x)
        result -= (
            n / (xi[k + n + 1] - xi[k + 1]) * self._eval_b_spline(n - 1, k + 1, xi, x)
        )
        return result

    def _eval_b_spline_dxx(self, n, k, xi, x):
        """Evaluate second order derivative of standard uniform B-splines.

        Parameters
        ----------
        n : int
            degree
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of second order derivative of standard uniform B-splines
        """
        result = n / (xi[k + n] - xi[k]) * self._eval_b_spline_dx(n - 1, k, xi, x)
        result -= (
            n
            / (xi[k + n + 1] - xi[k + 1])
            * self._eval_b_spline_dx(n - 1, k + 1, xi, x)
        )
        return result

    def _eval_lagrange(self, k, xi, x):
        """Evaluate Lagrange polynomials.

        Parameters
        ----------
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of Lagrange polynomial
        """
        result = torch.ones_like(x)
        for m in range(len(xi)):
            if m != k:
                result *= (x - xi[m]) / (xi[k] - xi[m])
        return result

    def _eval_lagrange_dx(self, k, xi, x):
        """Evaluate derivative of Lagrange polynomials.

        Parameters
        ----------
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of derivative of Lagrange polynomial
        """
        result = torch.zeros_like(x)
        for m in range(len(xi)):
            if m != k:
                temp = torch.ones_like(x)
                for l in range(len(xi)):
                    if l != m and l != k:
                        temp *= (x - xi[l]) / (xi[k] - xi[l])
                result += 1 / (xi[k] - xi[m]) * temp
        return result

    def _eval_lagrange_dxx(self, k, xi, x):
        """Evaluate second order derivative of Lagrange polynomials.

        Parameters
        ----------
        k : int
            index
        xi : torch.Tensor
            knot sequence
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            evaluations of second order derivative of Lagrange polynomial
        """
        result = torch.zeros_like(x)
        for m in range(len(xi)):
            if m != k:
                temp_m = torch.zeros_like(x)
                for l in range(len(xi)):
                    if l != m and l != k:
                        temp_l = torch.ones_like(x)
                        for n in range(len(xi)):
                            if n != l and n != m and n != k:
                                temp_l *= (x - xi[n]) / (xi[k] - xi[n])
                        temp_m += 1 / (xi[k] - xi[l]) * temp_l
                result += 1 / (xi[k] - xi[m]) * temp_m
        return result

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
            evaluations of all basis functions in all data points,
            i.e., interpolation matrix
        """
        pass


class Hier_B_splines(Splines):
    """Implementation of hierarchical B-splines.

    Attributes
    ----------
    l_max : int
            maximum level
    boundary : bool
            if `True`, basis functions at left and right
            boundary are added at level `0`
    n : int
            degree
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0.0, b=1.0, n=3, boundary=True):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3

        Raises
        ------
        ValueError
            Degree violation detected.
        """
        self.l_max = l_max
        levels = [0 for _ in range(self.l_max + 1)]
        levels[0] = boundary * 2
        for l in range(1, self.l_max + 1):
            levels[l] = 2 ** (l - 1)
        super().__init__(levels, a=a, b=b)
        self.boundary = boundary
        if n % 2 != 1:
            raise ValueError("Only odd degrees allowed.")
        self.n = n

        # attribute containing function to call for evaluation
        # Note: might be changed to alter behavior of `forward` method
        self.spline_func = self._eval_b_spline

    def dx(self):
        """Construct first order derivative object.

        Returns
        -------
        Hier_B_splines
            first order derivative object
        """
        spline_obj = Hier_B_splines(
            self.l_max, self.data_a, self.data_b, self.n, self.boundary
        )
        # replace spline evaluation by derivative
        spline_obj.spline_func = spline_obj._eval_b_spline_dx
        return spline_obj

    def dxx(self):
        """Construct second order derivative object.

        Returns
        -------
        Hier_B_splines
            second order derivative object
        """
        spline_obj = Hier_B_splines(
            self.l_max, self.data_a, self.data_b, self.n, self.boundary
        )
        # replace spline evaluation by derivative
        spline_obj.spline_func = spline_obj._eval_b_spline_dxx
        return spline_obj

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of hierarchical B-splines.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in all data points,
            i.e., interpolation matrix
        """
        eval = torch.empty(self.bf_num, len(x))

        write_idx = 0

        for l in range(0, self.l_max + 1):
            h_l = 2 ** -l
            xi = torch.linspace(
                -self.n * h_l, (2 ** l + self.n) * h_l, 2 ** l + 2 * self.n + 1
            )
            xi = self._scale(xi)
            if l == 0:
                if self.boundary:
                    for k in range(2):
                        k_hier = int(k + (self.n - 1) / 2)
                        eval[write_idx] = self.spline_func(self.n, k_hier, xi, x)
                        write_idx += 1
            else:
                for k in range(1, 2 ** l + 1, 2):
                    k_hier = int(k + (self.n - 1) / 2)
                    eval[write_idx] = self.spline_func(self.n, k_hier, xi, x)
                    write_idx += 1

        assert write_idx == self.bf_num

        return eval.T


'''class Hier_B_splines_dx(Hier_B_splines):
    """Implementation of derivative of hierarchical B-splines.

    Attributes
    ----------
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0.0, b=1.0, boundary=False, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, boundary, n)
        # set spline evaluation to derivative
        self.spline_func = self._eval_b_spline_dx


class Hier_B_splines_dxx(Hier_B_splines):
    """Implementation of second order derivative of hierarchical B-splines.

    Attributes
    ----------
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0.0, b=1.0, boundary=False, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, boundary, n)
        # set spline evaluation to second order derivative
        self.spline_func = self._eval_b_spline_dxx'''


class Nak_B_splines(Splines):
    """Implementation of not-a-knot B-splines.

    Attributes
    ----------
    l_max : int
            maximum level
    boundary : bool
            if `True`, basis functions at left and right
            boundary are added at level `0`
    n : int
            degree
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, n=3, boundary=True):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3

        Raises
        ------
        ValueError
            Degree violation detected.
        """
        self.l_max = l_max
        levels = [0 for _ in range(self.l_max + 1)]
        levels[0] = boundary * 2
        for l in range(1, self.l_max + 1):
            levels[l] = 2 ** (l - 1)
        super().__init__(levels, a=a, b=b)
        self.boundary = boundary
        if n % 2 != 1:
            raise ValueError("Only odd degrees allowed.")
        self.n = n

        # attributes containing functions to call for evaluation
        # Note: might be changed to alter behavior of `forward` method
        self.lagrange_func = self._eval_lagrange
        self.spline_func = self._eval_b_spline

    def dx(self):
        """Construct first order derivative object.

        Returns
        -------
        Nak_B_splines
            first order derivative object
        """
        spline_obj = Nak_B_splines(
            self.l_max, self.data_a, self.data_b, self.n, self.boundary
        )
        # replace spline and Lagrange evaluation by derivative
        spline_obj.lagrange_func = spline_obj._eval_lagrange_dx
        spline_obj.spline_func = spline_obj._eval_b_spline_dx
        return spline_obj

    def dxx(self):
        """Construct second order derivative object.

        Returns
        -------
        Nak_B_splines
            second order derivative object
        """
        spline_obj = Nak_B_splines(
            self.l_max, self.data_a, self.data_b, self.n, self.boundary
        )
        # replace spline and Lagrange evaluation by derivative
        spline_obj.lagrange_func = spline_obj._eval_lagrange_dxx
        spline_obj.spline_func = spline_obj._eval_b_spline_dxx
        return spline_obj

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of not-a-knot B-splines.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in all data points,
            i.e., interpolation matrix
        """
        eval = torch.empty(self.bf_num, len(x))

        write_idx = 0

        for l in range(0, self.l_max + 1):
            h_l = 2 ** -l
            if l < math.ceil(math.log2(self.n)):
                # Lagrange polynomials
                xi = torch.linspace(0, 1, 2 ** l + 1)
                xi = self._scale(xi)

                if l == 0:
                    if self.boundary:
                        for k in range(2):
                            eval[write_idx] = self.lagrange_func(k, xi, x)
                            write_idx += 1
                else:
                    for k in range(1, 2 ** l + 1, 2):
                        eval[write_idx] = self.lagrange_func(k, xi, x)
                        write_idx += 1
            else:
                # B-splines
                xi = torch.zeros(2 ** l + self.n + 2)
                for k in range(self.n + 1):
                    xi[k] = (k - self.n) * h_l
                for k in range(self.n + 1, 2 ** l + 1):
                    k_local = k + (self.n - 1) / 2
                    xi[k] = (k_local - self.n) * h_l
                for k in range(2 ** l + 1, 2 ** l + self.n + 2):
                    k_local = k + self.n - 1
                    xi[k] = (k_local - self.n) * h_l
                xi = self._scale(xi)

                if l == 0:
                    if self.boundary:
                        for k in range(2):
                            eval[write_idx] = self.spline_func(self.n, k, xi, x)
                            write_idx += 1
                else:
                    for k in range(1, 2 ** l + 1, 2):
                        eval[write_idx] = self.spline_func(self.n, k, xi, x)
                        write_idx += 1

        assert write_idx == self.bf_num

        return eval.T


'''class Nak_B_splines_dx(Nak_B_splines):
    """Implementation of derivative of not-a-knot B-splines.
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, boundary=False, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, boundary, n)
        # set evaluations to derivatives
        self.lagrange_func = self._eval_lagrange_dx
        self.spline_func = self._eval_b_spline_dx


class Nak_B_splines_dxx(Nak_B_splines):
    """Implementation of second order derivative of not-a-knot B-splines.
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, boundary=False, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        boundary : bool, optional
            if `True`, basis functions at left and right
            boundary are added at level `0`, by default False
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, boundary, n)
        # set evaluations to second order derivatives
        self.lagrange_func = self._eval_lagrange_dxx
        self.spline_func = self._eval_b_spline_dxx'''


class Boundary_B_splines(Splines):
    """Implementation of boundaryless not-a-knot B-splines.

    Attributes
    ----------
    l_max : int
            maximum level
    n : int
            degree
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        n : int, optional
            degree, by default 3

        Raises
        ------
        ValueError
            Degree violation detected.
        """
        self.l_max = l_max
        levels = [0 for _ in range(self.l_max + 1)]
        for l in range(1, self.l_max + 1):
            levels[l] = 2 ** (l - 1)
        super().__init__(levels, a=a, b=b)
        if n % 2 != 1:
            raise ValueError("Only odd degrees allowed.")
        self.n = n

        # attributes containing functions to call for evaluation
        # Note: might be changed in child class,
        # altering behavior of `forward` method
        self.lagrange_func = self._eval_lagrange
        self.spline_func = self._eval_b_spline

    def dx(self):
        """Construct first order derivative object.

        Returns
        -------
        Boundary_B_splines
            first order derivative object
        """
        spline_obj = Boundary_B_splines(self.l_max, self.data_a, self.data_b, self.n)
        # replace spline and Lagrange evaluation by derivative
        spline_obj.lagrange_func = spline_obj._eval_lagrange_dx
        spline_obj.spline_func = spline_obj._eval_b_spline_dx
        return spline_obj

    def dxx(self):
        """Construct second order derivative object.

        Returns
        -------
        Boundary_B_splines
            second order derivative object
        """
        spline_obj = Boundary_B_splines(self.l_max, self.data_a, self.data_b, self.n)
        # replace spline and Lagrange evaluation by derivative
        spline_obj.lagrange_func = spline_obj._eval_lagrange_dxx
        spline_obj.spline_func = spline_obj._eval_b_spline_dxx
        return spline_obj

    def forward(self, x):
        """Overrides interface method and returns tensor
        with evaluations of boundaryless not-a-knot B-splines.

        Returns
        -------
        torch.Tensor
            evaluations of all basis functions in all data points,
            i.e., interpolation matrix
        """
        eval = torch.empty(self.bf_num, len(x))

        write_idx = 0

        for l in range(1, self.l_max + 1):
            h_l = 2 ** -l
            if l < math.ceil(math.log2(self.n + 2)):
                # Lagrange polynomials
                xi = torch.linspace(h_l, 1 - h_l, 2 ** l - 1)
                xi = self._scale(xi)

                for k in range(0, 2 ** l, 2):
                    eval[write_idx] = self.lagrange_func(k, xi, x)
                    write_idx += 1
            else:
                # B-splines
                #xi = torch.zeros(2 ** l + self.n + 1)
                xi = torch.zeros(2 ** l + self.n)
                for k in range(self.n + 1):
                    xi[k] = (k - self.n) * h_l
                for k in range(self.n + 1, 2 ** l - 1):
                    k_local = k + (self.n + 1) / 2
                    xi[k] = (k_local - self.n) * h_l
                for k in range(2 ** l - 1, 2 ** l + self.n):
                    k_local = k + self.n + 1
                    xi[k] = (k_local - self.n) * h_l

                xi = self._scale(xi)

                for k in range(0, 2 ** l, 2):
                    eval[write_idx] = self.spline_func(self.n, k, xi, x)
                    write_idx += 1

        assert write_idx == self.bf_num

        return eval.T


'''class Boundary_B_splines_dx(Boundary_B_splines):
    """Implementation of derivative of boundaryless not-a-knot B-splines.

    Attributes
    ----------
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, n)
        # set evaluations to derivatives
        self.lagrange_func = self._eval_lagrange_dx
        self.spline_func = self._eval_b_spline_dx


class Boundary_B_splines_dxx(Boundary_B_splines):
    """Implementation of second order derivative of boundaryless not-a-knot B-splines.

    Attributes
    ----------
    lagrange_func : function
            function for Lagrange evaluation
    spline_func : function
            function for spline evaluation
    """

    def __init__(self, l_max, a=0, b=1, n=3):
        """
        Parameters
        ----------
        l_max : int
            maximum level
        a : float, optional
            left boundary of domain, by default 0.
        b : float, optional
            left boundary of domain, by default 1.
        n : int, optional
            degree, by default 3
        """
        super().__init__(l_max, a, b, n)
        # set evaluations to second order derivatives
        self.lagrange_func = self._eval_lagrange_dxx
        self.spline_func = self._eval_b_spline_dxx'''


def rescale(parent_spline, rescaler, *args):
    """Rescale knot distribution of spline object.

    Parameters
    ----------
    parent_spline : type
        spline class type
    rescaler : function
        function with positive derivative from unit interval to unit interval

    Returns
    -------
    Type[Splines]
        spline object with rescaled knots
    """

    class Helper(parent_spline):
        """Helper class to inherit from custom class.

        Parameters
        ----------
        parent_spline : Type[Splines]
            spline object
        """

        def __init__(self):
            """Construct new spline object and initialize parent spline."""
            super().__init__(*args)

        def _scale(self, xi):
            """Overrides original scaling method for knots.

            Parameters
            ----------
            xi : torch.Tensor
                knot sequence

            Returns
            -------
            torch.Tensor
                knots scaled to data interval with rescaled distribution
            """
            return super()._scale(rescaler(xi))

    return Helper()
