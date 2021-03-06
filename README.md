# Sparsetorch Package
This package implements higher dimensional function approximation by setting up interpolation matrices using different combination methods, including the sparse grid method. After the setup, the approximation problem can be solved by either solving the least squares problem or by using any gradient descent method. The package builds on top of the PyTorch package.

## Installation
The package can be downloaded from Github:
```
git clone https://github.com/timotheehornek/sparsetorch.git
```
Afterwards, it can be installed using `pip`. Simply run the following after cloning:
```
pip install ./path_to_folder/sparsetorch/
```
Note that `torch==1.9.0+cpu` (for the package) and `matplotlib==3.4.3` (for the examples) are also required.
PyTorch can be downloaded [here](https://pytorch.org/get-started/locally/).

## Documentation
Documentation can be found in the `docs` folder and additionally hosted [here](https://timotheehornek.github.io/sparsetorch/).

## Examples
A tutorial is provided as Jupyter notebook. Moreover, several examples are provided for demonstrating the capabilities in the same folder.