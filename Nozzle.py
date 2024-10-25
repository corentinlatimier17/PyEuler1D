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

################################### MESH #########################################
Ncells = 50
xmin = 0 
xmax = 10

def area_NOZZLE(x):
    return 1.398 +0.347*np.tanh(0.8*x-4) 

def dAdx_NOZZLE(x):
    return 0.347*0.8*(1-np.tanh(0.8*x-4.0)**2)

MESH = mesh(Ncells, xmin, xmax)
MESH.init_mesh(area_NOZZLE, dAdx_NOZZLE)

############################# Q - Conserved variables ###############################
Q = ConservativeVariables(MESH.num_TotCells)

# Initialization with intlet
Minf, pinf, rho_inf = 1.25, 1.0, 1.0
u_inf = Minf*np.sqrt(GAMMA*pinf/rho_inf)
Q.init_Q_Nozzle(MESH, rho_inf, u_inf, pinf)

################################# F - Fluxes #########################################
E = Fluxes(MESH.num_TotCells)

################################ S - SourceTerm ######################################
S = SourceTerm(MESH.num_TotCells)

################################ BoundaryConditions ####################################
BC_LEFT = BoundaryCondition(type="SupersonicInlet")
BC_RIGHT =  BoundaryCondition(type="SupersonicOutlet")
# BC_RIGHT = BoundaryCondition(type="SubsonicOutlet", back_pressure=1.9*pinf, mesh=MESH)
BCS = BoundaryConditions(BC_LEFT, BC_RIGHT)

################################ Numerical Scheme #####################################
SCHEME = BeamWarming(0.125, 5*0.125)

################################# Linear Solver #######################################
LINEAR_SOLVER = DirectSolver()

################################ Solver ################################################
CFL = 0.4
maxTime = 30
files = ['output/SOD/rhoA.txt', 'output/SOD/u.txt', 'output/SOD/rhoEA.txt', 'output/SOD/pressure.txt']
solver = TransientSolver(maxTime, CFL, MESH, Q, E, S, BCS, SCHEME, files, LINEAR_SOLVER)

################################ Resolution #############################################
solver.solve()




