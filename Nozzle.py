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
Ncells = 200
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

Q.init_Q_Nozzle(MESH, rho_inf, u_inf, pinf, Minf)

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
SCHEME = LaxWendroff()
scheme = 'Lax-Wendroff'

################################# Linear Solver #######################################
LINEAR_SOLVER = DirectSolver() # only used for Beam Warming scheme

################################ Solver (steady) ################################################
CFL = 0.5
eps_res = 10**(-5)
files = ['output/Nozzle/rhoA.txt', 'output/Nozzle/u.txt', 'output/Nozzle/rhoEA.txt', 'output/Nozzle/pressure.txt', 'output/Nozzle/mach.txt']
file_residual = 'output/Nozzle/residuals.txt'
solver = SteadySolver(eps_res, CFL, MESH, Q, E, S, BCS, SCHEME, files,file_residual,LINEAR_SOLVER, itermax=50000)

################################ Resolution #############################################
solver.solve()

################################ Post-processing ########################################
plot_residuals_from_file(file_residual) # plot residuals across iterations
plot_mach_colormap(files[4]) # Mach distribution
plot_all_in_one(files, xmax, scheme)
plot_mach(files, xmax, scheme)
plot_density(files, xmax, scheme)
plot_energy(files, xmax, scheme)
plot_pressure(files, xmax, scheme)
plt.show()



