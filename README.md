# CombinatorialApproachestoehzcapacityofsimplex
Code for gradient descent on SL(2n,R), affine symplectomorphism tests, and EHZ capacity computations for simplices.
# Symplectic simplex computations

This repository contains Python code for numerical experiments related to simplices,
symplectic geometry, and optimization on the special linear group.

The code covers four main tasks:

- exact computation of the Ekeland--Hofer--Zehnder capacity of a simplex,
- exact computation of the associated systolic ratio,
- gradient-descent experiments on `SL(2n, R)`,
- affine symplectomorphism tests for simplices.

## Included code files

This repository uses the following Python files.

- `ilp.py`  
  Gurobi-based integer linear programming code for extracting optimal and near-active
  tournament matrices associated with the trace reformulation.

- `gradientdescent.py`  
  Gradient-descent code on `SL(2n, R)` using the ILP output to evaluate the objective
  and compute active-branch subgradients.

- `ehzofsimplex.py`  
  Script for computing the EHZ capacity and systolic ratio of a simplex from its vertices.

- `Symplectomorphismtest.py`  
  Standalone console/programmatic script for testing affine symplectomorphism of simplices
  using NetworkX.

## What the code computes

### 1. Exact EHZ capacity and systolic ratio

The script `ehzofsimplex.py` takes simplex vertices, centers the simplex at its barycenter,
computes the polar simplex, solves the associated integer linear program, and then computes
the EHZ capacity and the corresponding systolic ratio.

### 2. Gradient descent on `SL(2n, R)`

The script `gradientdescent.py` performs a multi-phase descent on `SL(2n, R)`.
At each iterate it computes active branch matrices from the ILP, forms projected branch gradients,
takes the minimum-Frobenius-norm element of their convex hull, and uses this as the descent direction.

### 3. Affine symplectomorphism of simplices

The script `Symplectomorphismtest.py` centers simplices, forms their skew-symmetric pairing matrices
`W = X^T J X`, and uses a NetworkX graph-isomorphism step on edge-labeled complete graphs to detect
permutation-conjugacy. If a matching permutation is found, it reconstructs the corresponding symplectic map.

## Related dissertation

This repository accompanies my dissertation on symplectic geometry and simplex computations.

Dissertation link: 
## Tested environment

This repository was tested with:

- Python 3.9.23
- NumPy 1.26.4
- SciPy 1.13.1
- CVXPY 1.6.7
- OSQP 1.1.1
- NetworkX 3.2.1
- gurobipy 12.0.3

A working Gurobi installation and valid Gurobi license are required for the ILP-based parts.

## Installation

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate symplectic-simplex
```

## How to run

### Run the simplex EHZ / systolic-ratio script

```bash
python ehzofsimplex.py
```

### Run the gradient-descent script

```bash
python gradientdescent.py
```

### Run the affine symplectomorphism test

```bash
python Symplectomorphismtest.py
```


## Notes

- The ambient dimension must be even.
- The ILP computations are the main computational bottleneck.
- The gradient-descent script uses many iterations per phase in order to increase the chance
  of ending at a reasonable point.
- Higher-dimensional computations may require substantial computational resources.

## License

This repository is released under the MIT License.

## Citation

## Author

Karla Leipold  
University of Cologne
