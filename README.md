# PyEuler1D
Description

This project implements three numerical schemes for solving hyperbolic PDEs in computational fluid dynamics (CFD):

    Beam-Warming Scheme: Second-order implicit method.
    MacCormack Scheme: Explicit predictor-corrector method.
    Lax-Wendroff Scheme: Second-order explicit method.

Currently, the solver focuses on the Sod shock tube problem.
Features:

    Modular linear solvers: Jacobi iteration and direct matrix inversion.
    Support for boundary conditions like supersonic inlet/outlet.

Installation

Install required packages:

bash

pip install -r requirements.txt

Running the Solver

Select your numerical scheme and linear solver, then call iterate to solve the Sod shock tube problem.
