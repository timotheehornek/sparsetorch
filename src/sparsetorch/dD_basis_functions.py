"""Basis function objects (children from `BF_1D`) can be combined to create higher dimensional basis functions.
Different possibilities are implemented as children of `BF_dD`, which can be understood as an interface for combination methods."""

from collections import deque
import torch


class BF_dD(torch.nn.Module):
    """Implementation of dD basis function evaluations as Pytorch layer.

    Attributes
    ----------
    bfs_1D : list of BF_1D
        1D basis functions
    bf_num : int
        number of basis functions after combination
    """

    def __init__(self, bfs_1D):
        """
        Parameters
        ----------
        bfs_1D : list of BF_1D
            1D basis functions
        """

        super().__init__()
        self.bfs_1D = bfs_1D
        self.bf_num = self.get_bf_num()

    def get_bf_num(self):
        """Interface method that should be implemented in child class.

        Returns
        -------
        int
            number of basis functions after combination
        """
        pass

    def comb_meth(self, evals_a, evals_b, levels_a, levels_b):
        """Interface method that should be implemented in child class.
        Contains combination method of two arbitrary interpolation matrices.

        Parameters
        ----------
        evals_a : torch.Tensor
            first interpolation matrix
        evals_b : torch.Tensor
            second interpolation matrix
        levels_a : list of int
            number of basis functions at each level in 'evals_a'
        levels_b : list of int
            number of basis functions at each level in 'evals_b'

        Returns
        -------
        torch.Tensor
            new interpolation matrix
        list of int
            number of basis functions at each level in new interpolation matrix
            (can be returned as 'None' if not required for implementation of this method
            or undefined)
        """
        pass

    def forward(self, x):
        """Apply pairwise combination method implemented in respective child class
        and return interpolation matrix.

        Parameters
        ----------
        x : list of torch.Tensor or torch.Tensor
            coordinates of all evaluation points, 'len(x)' should return number of evaluation points

        Returns
        -------
        torch.Tensor
            interpolation matrix
        """
        
        # create deques (usage like queue)
        eval_deque = deque(bf(x) for bf,x in zip(self.bfs_1D, x))
        level_deque = deque(bf.levels for bf in self.bfs_1D)
        
        # combine functions, FIFO order
        # (apend to the right and pop from left)
        while len(eval_deque) > 1:
            evals, levels = self.comb_meth(eval_deque.popleft(), eval_deque.popleft(), level_deque.popleft(), level_deque.popleft())
            eval_deque.append(evals)
            level_deque.append(levels)
        return eval_deque.pop()

        '''
        # combine functions, LIFO order
        # (append and pop from right side)
        while len(eval_deque) > 1:
            evals, levels = self.comb_meth(eval_deque.pop(), eval_deque.pop(), level_deque.pop(), level_deque.pop())
            eval_deque.append(evals)
            level_deque.append(levels)
        return eval_deque.pop()
        '''

class Tensorprod(BF_dD):
    """Implementation of dD basis function evaluations as Pytorch layer.
    Basis functions are combined as tensor product.
    """
    def get_bf_num(self):
        """Overrides interface method and returns number of basis functions
        after tensorprod combination of basis functions.

        Returns
        -------
        int
            number of basis functions after combination
        """
        bf_num = 1
        for bf in self.bfs_1D:
            bf_num *= sum(bf.levels)
        return bf_num

    def comb_meth(self, evals_a, evals_b, levels_a, levels_b):
        """Overrides interface method and implements tensorproduct combination
        of interpolation matrices.

        Parameters
        ----------
        evals_a : torch.Tensor
            first interpolation matrix
        evals_b : torch.Tensor
            second interpolation matrix
        levels_a : list of int
            number of basis functions at each level in 'evals_a', unused
        levels_b : list of int
            number of basis functions at each level in 'evals_b', unused

        Returns
        -------
        torch.Tensor
            new interpolation matrix
        list of int
            number of basis functions at each level unused in implementation, therefore always returned as 'None'
        """
        result = evals_a.T.unsqueeze(-2) * evals_b.T
        size = torch.tensor(result.shape)
        return torch.reshape(result.T, (size[-1], torch.prod(size[:-1]))), None


class Elemprod(BF_dD):
    """Implementation of dD basis function evaluations as Pytorch layer.
    Basis functions are combined as elementwise product.
    """
    def get_bf_num(self):
        """Overrides interface method and returns number of basis functions
        after elemprod combination of basis functions.

        Returns
        -------
        int
            number of basis functions after combination
        """
        return sum(self.bfs_1D[0].levels)

    def comb_meth(self, evals_a, evals_b, levels_a, levels_b):
        """Overrides interface method and implements elementwise combination
        of interpolation matrices.

        Parameters
        ----------
        evals_a : torch.Tensor
            first interpolation matrix
        evals_b : torch.Tensor
            second interpolation matrix
        levels_a : list of int
            number of basis functions at each level in 'evals_a', unused
        levels_b : list of int
            number of basis functions at each level in 'evals_b', unused

        Returns
        -------
        torch.Tensor
            new interpolation matrix
        list of int
            number of basis functions at each level unused in implementation, therefore always returned as 'None'
        """
        return evals_a * evals_b, None


class Sparse(BF_dD):
    """Implementation of dD basis function evaluations as Pytorch layer.
    Basis functions are combined as sparse grid.
    """
    def __get_levcombs(self, l_max):
        """Generator for all possible sparse level
        combination pairs for given maximum level.
        Pairs generated in ascending order of corresponding
        level (sum of returned level pair).

        Parameters
        ----------
        l_max : int
            maximum level

        Yields
        -------
        tuple
            maximum levels, two entries
        """
        for max in range(l_max + 1):
            for lev_b in range(max + 1):
                lev_a = max - lev_b
                yield (lev_a, lev_b)

    def get_bf_num(self):
        """Overrides interface method and returns number of basis functions
        after sparse combination of basis functions.

        Returns
        -------
        int
            number of basis functions after combination
        """
        # initialize
        levels_dD = self.bfs_1D[0].levels

        # go through all dimensions
        for bf in self.bfs_1D[1:]:
            # pad list
            pad = max(len(bf.levels) - len(levels_dD), 0)
            levels_dD.extend([0] * pad)

            # initialize list for new entries
            levels_new = [0] * len(levels_dD)

            # levelwise traversation
            for lev_1D, lev_dD in self.__get_levcombs(len(levels_dD) - 1):
                try:
                    levels_new[lev_1D + lev_dD] += levels_dD[lev_dD] * bf.levels[lev_1D]
                except IndexError:
                    # handle case that basis functions do not have the same number of levels
                    continue
            levels_dD = levels_new
        return sum(levels_dD)

    def comb_meth(self, evals_a, evals_b, levels_a, levels_b):
        """Overrides interface method and implements sparse combination
        of interpolation matrices.

        Parameters
        ----------
        evals_a : torch.Tensor
            first interpolation matrix
        evals_b : torch.Tensor
            second interpolation matrix
        levels_a : list of int
            number of basis functions at each level in 'evals_a'
        levels_b : list of int
            number of basis functions at each level in 'evals_b'

        Returns
        -------
        torch.Tensor
            new interpolation matrix
        list of int
            number of basis functions at each level in new interpolation matrix
        """

        # compute highest level for computation
        l_max = max(len(levels_a), len(levels_b)) - 1

        # new list with number of evaluations at each level
        levels_new = [0] * (l_max + 1)

        # number of evaluation points
        n = evals_a.shape[0]

        # compute number of evaluations
        num_evals_new = 0
        for i, j in self.__get_levcombs(l_max):
            num_evals_new += levels_a[i] * levels_b[j]

        # initialize new evaluation tensor
        evals_new = torch.empty((n, num_evals_new))

        idx = 0  # writing index
        # levelwise traversation w.r.t. new evaluation tensor
        for lev_a, lev_b in self.__get_levcombs(l_max):
            try:
                # compute index ranges for respective levels
                start_a = sum(levels_a[:lev_a])
                end_a = start_a + levels_a[lev_a]
                start_b = sum(levels_b[:lev_b])
                end_b = start_b + levels_b[lev_b]
            except IndexError:
                # handle case that one basis provides more levels than the other
                continue

            # update level list
            levels_new[lev_a + lev_b] += levels_a[lev_a] * levels_b[lev_b]

            # extract and duplicate relevant columns
            idx_end = idx + (end_a - start_a) * (end_b - start_b)
            cols_a = torch.repeat_interleave(
                evals_a[:, start_a:end_a], end_b - start_b, dim=1
            )
            cols_b = evals_b[:, start_b:end_b].repeat(1, end_a - start_a)

            # elementwise multiplication of columns
            evals_new[:, idx:idx_end] = cols_a * cols_b

            idx = idx_end

        return evals_new, levels_new