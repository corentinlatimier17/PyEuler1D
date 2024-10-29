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
# Initialization with intlet
Minf, pinf, rho_inf = 1.25, 1.0, 1.0
u_inf = Minf*np.sqrt(GAMMA*pinf/rho_inf)

Q_1 = ConservativeVariables(MESH.num_TotCells)
Q_2 = ConservativeVariables(MESH.num_TotCells)
Q_3 = ConservativeVariables(MESH.num_TotCells)

Q_1.init_Q_Nozzle(MESH, rho_inf, u_inf, pinf, Minf)
Q_2.init_Q_Nozzle(MESH, rho_inf, u_inf, pinf, Minf)
Q_3.init_Q_Nozzle(MESH, rho_inf, u_inf, pinf, Minf)

################################# F - Fluxes #########################################
E_1 = Fluxes(MESH.num_TotCells)
E_2 = Fluxes(MESH.num_TotCells)
E_3 = Fluxes(MESH.num_TotCells)

################################ S - SourceTerm ######################################
S_1 = SourceTerm(MESH.num_TotCells)
S_2 = SourceTerm(MESH.num_TotCells)
S_3 = SourceTerm(MESH.num_TotCells)

################################ BoundaryConditions ####################################
BC_LEFT = BoundaryCondition(type="SupersonicInlet")
# BC_RIGHT =  BoundaryCondition(type="SupersonicOutlet")
BC_RIGHT = BoundaryCondition(type="SubsonicOutlet", back_pressure=1.9*pinf, mesh=MESH)
BCS = BoundaryConditions(BC_LEFT, BC_RIGHT)

################################ Numerical Scheme #####################################
SCHEME_1 = MacCormack()
SCHEME_2 = LaxWendroff()
SCHEME_3 = BeamWarming(0.125, 2*0.125)
scheme1 = "Mac-Cormack - CFL = 0.8, N = 200"
scheme2 = "Lax-Wendroff - CFL = 0.8, N = 200"
scheme3 = r'Beam-Warming - CFL = 0.8, N = 200, $\epsilon_e = 0.125$, $\epsilon_i = 2*\epsilon_e$'

################################# Linear Solver #######################################
LINEAR_SOLVER = DirectSolver() # only used for Beam Warming scheme

################################ Solver (steady) ################################################
CFL = 1
eps_res = 10**(-6)
files = ['output/Nozzle/rhoA.txt', 'output/Nozzle/u.txt', 'output/Nozzle/rhoEA.txt', 'output/Nozzle/pressure.txt', 'output/Nozzle/mach.txt']
file_residual = 'output/Nozzle/residuals.txt'

solver_1 = SteadySolver(eps_res, CFL, MESH, Q_1, E_1, S_1, BCS, SCHEME_1, files,file_residual,LINEAR_SOLVER, itermax=50000)
solver_2 = SteadySolver(eps_res, CFL, MESH, Q_2, E_2, S_2, BCS, SCHEME_2, files,file_residual,LINEAR_SOLVER, itermax=50000)
solver_3 = SteadySolver(eps_res, CFL, MESH, Q_3, E_3, S_3, BCS, SCHEME_3, files,file_residual,LINEAR_SOLVER, itermax=50000)

scheme = scheme3 # to change the label, change scheme3 in the scheme you are working on

################################ Resolution & Post-processing #############################################
solver_3.solve()
plot_residuals_from_file(file_residual, scheme) # plot residuals across iterations
plot_mach_colormap(files[4], scheme) # Mach distribution

plot_all_in_one(files, xmax, scheme)
plot_mach(files, xmax, scheme)
plot_density(files, xmax, scheme)
plot_energy(files, xmax, scheme)
plot_pressure(files, xmax, scheme)

plt.show()





