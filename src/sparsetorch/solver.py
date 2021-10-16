"""The solver module contains the `Model` class, which finalizes the definition of the `torch` model.
The `Solver` class provides methods to fit the model to training data."""

import torch


class Model(torch.nn.Module):
    """Function approximation model consisting
    of basis function layer computing interpolation matrix
    and linear layer computing linear combination of basis functions.

    Attributes
    ----------
    bf : BF_1D or BF_dD
        basis function layer
    lin : torch.nn.Linear
        linear layer
    """

    def __init__(self, bf, bf_num):
        """
        Parameters
        ----------
        bf : BF_1D or BF_dD
            basis function layer
        bf_num : int
            number of basis functions implemented in `bf`
        """
        super().__init__()
        self.bf = bf
        self.lin = torch.nn.Linear(bf_num, 1, bias=False)

    def get_interpolmat(self, x):
        """Returns interpolation matrix by only
        applying linear layer to input `x`.

        Parameters
        ----------
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            interpolation matrix
        """
        x = self.bf(x)
        return x

    def forward(self, x):
        """Apply model to input `x`, i.e.,
        first apply basis function layer and linear layer afterwards.

        Parameters
        ----------
        x : torch.Tensor
            evaluation points

        Returns
        -------
        torch.Tensor
            approximation of evaluations
        """
        x = self.bf(x)
        x = self.lin(x)
        return torch.squeeze(x)


class Solver:
    """Solver to solve function approximation problem.

    Attributes
    ----------
    model : Model
        model for function approximation
    input : torch.Tensor
        evaluation points for optimization
    target : torch.Tensor
        function evaluations in 'input' points used for loss computation
    """

    def __init__(self, model, input, target):
        """
        Parameters
        ----------
        model : Model
            model for function approximation
        input : torch.Tensor
            evaluation points for optimization
        target : torch.Tensor
            function evaluations in 'input' points used for loss computation
        """
        self.model = model
        self.input = input
        self.target = target

    def le(self, lam_reg=None, device=torch.device("cpu")):
        """Solve approximation problem using linear equation.
        Parameters of `model` are updated afterwards.

        Parameters
        ----------
        lam : float, optional
            parameter for regularization, by default None
        device : torch.device, optional
            device to run solver on, remember to send input to device,
            by default torch.device('cpu')
        """

        # get interpolation matrix
        matrix = self.model.get_interpolmat(self.input).to(device)

        # solve
        if lam_reg is not None:  # solve with regularization
            lam_n = lam_reg * matrix.shape[0]
            b = matrix.T @ self.target
            A = matrix.T @ matrix
            R = lam_n * torch.eye(matrix.shape[1])
            # lam, _ = torch.solve(b, A + R) # torch==1.8.1
            lam = torch.linalg.solve(A + R, b) # torch==1.9.0
        else:  # solve without regularization
            if matrix.shape[0] == matrix.shape[1]:  # square matrix
                # solve linear equation
                # lam, _ = torch.solve(self.target, matrix) # torch==1.8.1
                lam = torch.linalg.solve(matrix, self.target) # torch==1.9.0
            else:  # nonsquare matrix
                # solve least squares problem
                # S, _ = torch.lstsq(self.target, matrix) # torch==1.8.1
                S = torch.linalg.lstsq(matrix, self.target).solution # torch==1.9.0
                lam = S[: len(self.model.lin.weight[0])]

        # write computed weights into model
        self.model.lin.weight = torch.nn.Parameter(lam.T)

    def general(self, criterion, optimizer, eps, max_it=10000):
        """Implements functionality for general solver of function
        approximation problem.

        Parameters
        ----------
        criterion : torch.nn.*
            loss function from `torch.nn.*`
        optimizer : torch.optim.*
            optimizer from `torch.optim.*`
        eps : float
            threshold for loss
        max_it : int, optional
            maximum number of iterations of optimizer applied, by default 10000
        """
        output = self.model(self.input)
        loss = criterion(output, self.target)
        cur_it = 0
        while loss > eps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = self.model(self.input)
            loss_old = loss
            loss = criterion(output, self.target)
            cur_it += 1
            if cur_it % 10 == 0:
                print("Current iteration:", cur_it)
                print("Current loss:", "{:.4e}".format(loss))
            if cur_it == max_it:
                print("Stop optimizer, maximum number of iterations reached.")
                break
            if loss_old == loss:
                print("Stop optimizer, loss did not change.")
                break
        print("Final loss:", "{:.4e}".format(loss))
