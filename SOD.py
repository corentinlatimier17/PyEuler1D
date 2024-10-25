import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from Conservatives import *
from Fluxes import *
from SourceTerm import *
from BoundaryConditions import *
from Numerics import *
from Thermodynamics import *
from Solver import *
from LinearAlgebra import *
from PostProcessing import *

################################### MESH #########################################
Ncells = 500
xmin = 0 
xmax = 1000

def area_SOD(x):
    return np.ones_like(x)  

def dAdx_SOD(x):
    return np.zeros_like(x) 

MESH = mesh(Ncells, xmin, xmax)
MESH.init_mesh(area_SOD, dAdx_SOD)

############################# Q - Conserved variables ###############################
Q = ConservativeVariables(MESH.num_TotCells)
Q.init_Q_SOD(MESH)

################################# F - Fluxes #########################################
E = Fluxes(MESH.num_TotCells)

################################ S - SourceTerm ######################################
S = SourceTerm(MESH.num_TotCells)

################################ BoundaryConditions ####################################
BC_LEFT = BoundaryCondition(type="O_order_extrapolation")
BC_RIGHT = BoundaryCondition(type="O_oder_extrapolation")
BCS = BoundaryConditions(BC_LEFT, BC_RIGHT)

################################ Numerical Scheme #####################################
SCHEME = LaxWendroff()
# SCHEME = LaxWendroff(0.05, 2.5*0.05)

################################# Linear Solver #######################################
LINEAR_SOLVER = DirectSolver()

################################ Solver (transient) ###################################
CFL = 0.4
maxTime = 250
files = ['output/SOD/rhoA.txt', 'output/SOD/u.txt', 'output/SOD/rhoEA.txt', 'output/SOD/pressure.txt', 'output/SOD/mach.txt']
solver = TransientSolver(maxTime, CFL, MESH, Q, E, S, BCS, SCHEME, files, LINEAR_SOLVER)

################################ Resolution #############################################
solver.solve()

################################ Post processing #########################################
plot_all_in_one(files, xmax, "Lax Wendroff")
plot_mach(files, xmax, "Lax-Wendroff")
plt.show()





