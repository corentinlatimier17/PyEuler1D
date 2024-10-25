# PyEuler1D

This project implements numerical schemes for solving the Euler equations in computational fluid dynamics (CFD):

    Beam-Warming Scheme: Second-order implicit method.
    MacCormack Scheme: Explicit predictor-corrector method.
    Lax-Wendroff Scheme: Second-order explicit method.

The solver supports various linear solvers:

    Jacobi Method
    Classical Matrix Inversion

Currently, the solver is applied to the Sod shock tube and the Quasi-1D nozzle problems.
