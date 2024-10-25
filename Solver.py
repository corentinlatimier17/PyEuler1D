import numpy as np
import matplotlib.pyplot as plt
from Thermodynamics import *

class TransientSolver:
    def __init__(self, tmax, CFL, mesh, Q, E, S, BCS, scheme, files, LINEAR_SOLVER):
        self.tmax = tmax
        self.CFL = CFL
        self.mesh = mesh
        self.Q = Q
        self.E = E
        self.S = S
        self.BCS = BCS
        self.scheme = scheme
        self.CurrentTime = 0.0
        self.files = files
        self.linear_solver = LINEAR_SOLVER
        self.num_iter = 0
    
    def solve(self):
        while self.CurrentTime<=self.tmax:
            deltaT = self.computeDeltaT()
            self.scheme.iterate(self.Q, self.E, self.S, self.mesh, deltaT, self.BCS, self.linear_solver, self.num_iter)
            self.write_solution()
            self.CurrentTime += deltaT
            self.num_iter +=1
            print(f"Iteration n° {self.num_iter}, t = {self.CurrentTime}")
        plt.plot(self.mesh.coordinates[self.mesh.innerCellsIndexes] , self.Q.rhouA[self.mesh.innerCellsIndexes])
        plt.show()
    
    def computeDeltaT(self):
        return  np.min(self.mesh.dx*self.CFL/(compute_sound_velocity(self.Q, self.mesh)+np.abs(self.Q.rhouA[self.mesh.innerCellsIndexes]/self.Q.rhoA[self.mesh.innerCellsIndexes])))
    
    def write_solution(self):
        # If this is the first time step, write the x-coordinates as the first line
        if self.CurrentTime==0:
            with open(self.files[0], 'w') as f_rhoA, open(self.files[1], 'w') as f_rhouA, open(self.files[2], 'w') as f_rhoEA, open(self.files[3], 'w') as f_pressure:
                # x-coordinates with -1 at the first column
                x_coords = np.concatenate(([-1], self.mesh.get_coordinates()[self.mesh.innerCellsIndexes]))
                f_rhoA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhouA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhoEA.write(" ".join(map(str, x_coords)) + "\n")
                f_pressure.write(" ".join(map(str, x_coords)) + "\n")
        else:
            # Append mode for subsequent time steps
            with open(self.files[0], 'a') as f_rhoA, open(self.files[1], 'a') as f_rhouA, open(self.files[2], 'a') as f_rhoEA, open(self.files[3], 'a') as f_pressure:
                # Prepare the current row of data for each quantity
                # First column is the current time, followed by the solution in inner cells
                row_rhoA = [self.CurrentTime] + list(self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhouA = [self.CurrentTime] + list(self.Q.rhouA[self.mesh.innerCellsIndexes]/self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhoEA = [self.CurrentTime] + list(self.Q.rhoEA[self.mesh.innerCellsIndexes])
                row_pressure = [self.CurrentTime] + list(self.Q.pressure[self.mesh.innerCellsIndexes])

                # Write the current time step data to each file
                f_rhoA.write(" ".join(map(str, row_rhoA)) + "\n")
                f_rhouA.write(" ".join(map(str, row_rhouA)) + "\n")
                f_rhoEA.write(" ".join(map(str, row_rhoEA)) + "\n")
                f_pressure.write(" ".join(map(str, row_pressure)) + "\n")