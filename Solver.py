import numpy as np
import matplotlib.pyplot as plt
from Thermodynamics import *

class TransientSolver:
    def __init__(self, tmax, CFL, mesh, Q, E, S, BCS, scheme, files, LINEAR_SOLVER):
        self.tmax = tmax
        self.isSteady = False
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
            self.scheme.iterate(self.Q, self.E, self.S, self.mesh, deltaT, self.BCS, self.linear_solver, self.isSteady, self.num_iter)
            self.write_solution()
            self.CurrentTime += deltaT
            self.num_iter +=1
            print(f"Iteration n° {self.num_iter}, t = {self.CurrentTime}")
    
    def computeDeltaT(self):
        return  np.min(self.mesh.dx*self.CFL/(compute_sound_velocity(self.Q, self.mesh)[self.mesh.innerCellsIndexes]+np.abs(self.Q.rhouA[self.mesh.innerCellsIndexes]/self.Q.rhoA[self.mesh.innerCellsIndexes])))
    
    def write_solution(self):
        # If this is the first time step, write the x-coordinates as the first line
        if self.CurrentTime==0:
            with open(self.files[0], 'w') as f_rhoA, open(self.files[1], 'w') as f_rhouA, open(self.files[2], 'w') as f_rhoEA, open(self.files[3], 'w') as f_pressure, open(self.files[4], 'w') as f_mach:
                # x-coordinates with -1 at the first column
                x_coords = np.concatenate(([-1], self.mesh.get_coordinates()[self.mesh.innerCellsIndexes]))
                f_rhoA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhouA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhoEA.write(" ".join(map(str, x_coords)) + "\n")
                f_pressure.write(" ".join(map(str, x_coords)) + "\n")
                f_mach.write(" ".join(map(str, x_coords)) + "\n")
        else:
            # Append mode for subsequent time steps
            with open(self.files[0], 'a') as f_rhoA, open(self.files[1], 'a') as f_rhouA, open(self.files[2], 'a') as f_rhoEA, open(self.files[3], 'a') as f_pressure,  open(self.files[4], 'a') as f_mach:
                # Prepare the current row of data for each quantity
                # First column is the current time, followed by the solution in inner cells
                row_rhoA = [self.CurrentTime] + list(self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhouA = [self.CurrentTime] + list(self.Q.rhouA[self.mesh.innerCellsIndexes]/self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhoEA = [self.CurrentTime] + list(self.Q.rhoEA[self.mesh.innerCellsIndexes])
                row_pressure = [self.CurrentTime] + list(self.Q.pressure[self.mesh.innerCellsIndexes])
                row_mach = [self.CurrentTime] + list(self.Q.rhouA[self.mesh.innerCellsIndexes]/(self.Q.rhoA[self.mesh.innerCellsIndexes]*compute_sound_velocity(self.Q, self.mesh)[self.mesh.innerCellsIndexes]))

                # Write the current time step data to each file
                f_rhoA.write(" ".join(map(str, row_rhoA)) + "\n")
                f_rhouA.write(" ".join(map(str, row_rhouA)) + "\n")
                f_rhoEA.write(" ".join(map(str, row_rhoEA)) + "\n")
                f_pressure.write(" ".join(map(str, row_pressure)) + "\n")
                f_mach.write(" ".join(map(str, row_mach)) + "\n")

class SteadySolver:
    def __init__(self, eps_res, CFL, mesh, Q, E, S, BCS, scheme, files, file_residual, LINEAR_SOLVER, itermax = 1000):
        self.itermax = itermax
        self.isSteady = True
        self.CFL = CFL
        self.mesh = mesh
        self.Q = Q
        self.E = E
        self.S = S
        self.BCS = BCS
        self.scheme = scheme
        self.files = files
        self.file_residual = file_residual
        self.linear_solver = LINEAR_SOLVER
        self.num_iter = 0
        # Initialize residuals as a dictionary
        self.residuals = {'rhoA': 0, 'rhouA': 0, 'rhoEA': 0}
        self.eps_res = eps_res
    
    def solve(self):
        while self.num_iter<self.itermax:
            deltaT = self.computeDeltaT()
            self.residuals['rhoA'],self.residuals['rhouA'], self.residuals['rhoEA']= self.scheme.iterate(self.Q, self.E, self.S, self.mesh, deltaT, self.BCS, self.linear_solver, self.isSteady, self.num_iter)
            self.write_solution()
            self.write_residual()
            # Check if all residuals are below the threshold
            if all(res < self.eps_res for res in self.residuals.values()):
                print("Convergence achieved: All residuals below threshold.")
                break
            self.num_iter +=1
            print(f"Iteration n° {self.num_iter} | Residuals rhoA = {self.residuals["rhoA"]},  Residuals rhouA = {self.residuals["rhouA"]},  Residuals rhoEA = {self.residuals["rhoEA"]}")
    
    def computeDeltaT(self):
        return  self.mesh.dx*self.CFL/(compute_sound_velocity(self.Q, self.mesh)+np.abs(self.Q.rhouA/self.Q.rhoA))
    
    def write_solution(self):
        # If this is the first time step, write the x-coordinates as the first line
        if self.num_iter==0:
            with open(self.files[0], 'w') as f_rhoA, open(self.files[1], 'w') as f_rhouA, open(self.files[2], 'w') as f_rhoEA, open(self.files[3], 'w') as f_pressure, open(self.files[4], 'w') as f_mach:
                # x-coordinates with -1 at the first column
                x_coords = np.concatenate(([-1], self.mesh.get_coordinates()[self.mesh.innerCellsIndexes]))
                f_rhoA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhouA.write(" ".join(map(str, x_coords)) + "\n")
                f_rhoEA.write(" ".join(map(str, x_coords)) + "\n")
                f_pressure.write(" ".join(map(str, x_coords)) + "\n")
                f_mach.write(" ".join(map(str, x_coords)) + "\n")
        else:
            # Append mode for subsequent time steps
            with open(self.files[0], 'a') as f_rhoA, open(self.files[1], 'a') as f_rhouA, open(self.files[2], 'a') as f_rhoEA, open(self.files[3], 'a') as f_pressure,  open(self.files[4], 'a') as f_mach:
                # Prepare the current row of data for each quantity
                # First column is the current time, followed by the solution in inner cells
                row_rhoA = [self.num_iter] + list(self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhouA = [self.num_iter] + list(self.Q.rhouA[self.mesh.innerCellsIndexes]/self.Q.rhoA[self.mesh.innerCellsIndexes])
                row_rhoEA = [self.num_iter] + list(self.Q.rhoEA[self.mesh.innerCellsIndexes])
                row_pressure = [self.num_iter] + list(self.Q.pressure[self.mesh.innerCellsIndexes])
                row_mach = [self.num_iter] + list(self.Q.rhouA[self.mesh.innerCellsIndexes]/(self.Q.rhoA[self.mesh.innerCellsIndexes]*compute_sound_velocity(self.Q, self.mesh)[self.mesh.innerCellsIndexes]))

                # Write the current time step data to each file
                f_rhoA.write(" ".join(map(str, row_rhoA)) + "\n")
                f_rhouA.write(" ".join(map(str, row_rhouA)) + "\n")
                f_rhoEA.write(" ".join(map(str, row_rhoEA)) + "\n")
                f_pressure.write(" ".join(map(str, row_pressure)) + "\n")
                f_mach.write(" ".join(map(str, row_mach)) + "\n")
    
    def write_residual(self):
        # If this is the first time step, write the header
        if self.num_iter == 0:
            with open(self.file_residual, 'w') as f_residual:
                # Header with a placeholder -1 and residual labels
                header = ["-1", "RhoA", "RhouA", "RhoEA"]
                f_residual.write(" ".join(header) + "\n")
        else:
            # Append mode for subsequent time steps
            with open(self.file_residual, 'a') as f_residual:
                # Write the current iteration count, followed by latest residuals
                row_residuals = [
                    self.num_iter,
                    self.residuals["rhoA"],  # Latest residual for RhoA
                    self.residuals["rhouA"],  # Latest residual for RhouA
                    self.residuals["rhoEA"]   # Latest residual for RhoEA
                ]
                # Write to file, converting each entry to a string
                f_residual.write(" ".join(map(str, row_residuals)) + "\n")
