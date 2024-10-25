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
Ncells = 300
xmin = 0 
xmax = 1000

def area_SOD(x):
    return np.ones_like(x)  

def dAdx_SOD(x):
    return np.zeros_like(x) 

MESH = mesh(Ncells, xmin, xmax)
MESH.init_mesh(area_SOD, dAdx_SOD)

############################# Q - Conserved variables ###############################
Q_1 = ConservativeVariables(MESH.num_TotCells)
Q_2 = ConservativeVariables(MESH.num_TotCells)
Q_3 = ConservativeVariables(MESH.num_TotCells)

Q_1.init_Q_SOD(MESH)
Q_2.init_Q_SOD(MESH)
Q_3.init_Q_SOD(MESH)

################################# F - Fluxes #########################################
E_1 = Fluxes(MESH.num_TotCells)
E_2 = Fluxes(MESH.num_TotCells)
E_3 = Fluxes(MESH.num_TotCells)

################################ S - SourceTerm ######################################
S_1 = SourceTerm(MESH.num_TotCells)
S_2 = SourceTerm(MESH.num_TotCells)
S_3 = SourceTerm(MESH.num_TotCells)

################################ BoundaryConditions ####################################
BC_LEFT = BoundaryCondition(type="O_order_extrapolation")
BC_RIGHT = BoundaryCondition(type="O_oder_extrapolation")
BCS = BoundaryConditions(BC_LEFT, BC_RIGHT)

################################ Numerical Schemes #####################################
SCHEME_1 = MacCormack()
SCHEME_2 = LaxWendroff()
SCHEME_3 = BeamWarming(0.05, 2.5*0.05)

################################# Linear Solver #######################################
LINEAR_SOLVER = DirectSolver()

################################ Solvers (transient) ###################################
CFL = 0.5
maxTime = 250
files = ['output/SOD/rhoA.txt', 'output/SOD/u.txt', 'output/SOD/rhoEA.txt', 'output/SOD/pressure.txt', 'output/SOD/mach.txt']

solver_1 = TransientSolver(maxTime, CFL, MESH, Q_1, E_1, S_1, BCS, SCHEME_2, files, LINEAR_SOLVER)
solver_2 = TransientSolver(maxTime, CFL, MESH, Q_2, E_2, S_2, BCS, SCHEME_2, files, LINEAR_SOLVER)
solver_3 = TransientSolver(maxTime, CFL, MESH, Q_3, E_3, S_3, BCS, SCHEME_3, files, LINEAR_SOLVER)

################################ Resolution & Post processing #############################################
solver_1.solve()
plot_all_in_one(files, xmax, "Mac-Cormack")
plot_density(files, xmax, "Mac-Cormack")
plot_velocity(files, xmax, "Mac-Cormack")
plot_energy(files, xmax, "Mac-Cormack")
plot_pressure(files, xmax, "Mac-Cormack")
plot_mach(files, xmax, "Mac-Cormack")


solver_2.solve()
plot_all_in_one(files, xmax, "Lax-Wendroff")
plot_density(files, xmax, "Lax-Wendroff")
plot_velocity(files, xmax, "Lax-Wendroff")
plot_energy(files, xmax, "Lax-Wendroff")
plot_pressure(files, xmax, "Lax-Wendroff")
plot_mach(files, xmax, "Lax-Wendroff")

solver_3.solve()
plot_all_in_one(files, xmax, "Beam Warming")
plot_density(files, xmax,  "Beam Warming")
plot_velocity(files, xmax,  "Beam Warming")
plot_energy(files, xmax,  "Beam Warming")
plot_pressure(files, xmax,  "Beam Warming")
plot_mach(files, xmax,  "Beam Warming")

plt.show()






