import numpy as np
from Thermodynamics import *

class ConservativeVariables:
    def __init__(self, num_TotCells):
        self.rhoA = np.zeros(num_TotCells, dtype=np.float64)
        self.rhouA = np.zeros(num_TotCells, dtype=np.float64)
        self.rhoEA = np.zeros(num_TotCells, dtype=np.float64)
        self.pressure = np.zeros(num_TotCells, dtype=np.float64)
        self.num_TotCells = num_TotCells

    def get_rhoA(self):
        return self.rhoA
    
    def get_rhouA(self):
        return self.rhouA
    
    def get_rhoEA(self):
        return self.rhoEA
    
    def get_pressure(self):
        return self.pressure
    
    def get_QCell(self, i):
        Q = np.zeros((3, 1), dtype=np.float64)
        Q[0] = self.rhoA[i]
        Q[1] = self.rhouA[i]
        Q[2] = self.rhoEA[i]
        return Q

    def init_Q_SOD(self, mesh):
        for i in range(0, self.num_TotCells):
            if mesh.coordinates[i]< 500:
                self.rhoA[i] = 4
                self.pressure[i] = 4
            else:
                self.rhoA[i] = 1
                self.pressure[i] = 1
            self.rhoEA[i] = self.pressure[i]/(GAMMA-1)*mesh.area[i]

    def update_pressure(self, mesh):
        self.pressure = compute_pressure(self, mesh)




    