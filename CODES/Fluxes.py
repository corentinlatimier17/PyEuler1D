import numpy as np

class Fluxes:
    def __init__(self, num_TotCells):
        self.F1 = np.zeros(num_TotCells, dtype=np.float64)
        self.F2 = np.zeros(num_TotCells, dtype=np.float64)
        self.F3 = np.zeros(num_TotCells, dtype=np.float64)
        self.num_TotCells = num_TotCells
    
    def get_F1(self):
        return self.F1
    
    def get_F2(self):
        return self.F2
    
    def get_F3(self):
        return self.F3
    
    def get_FluxesCell(self, i):
        E = np.zeros((3,1), dtype=np.float64)
        E[0] = self.F1[i]
        E[1] = self.F2[i]
        E[2]= self.F3[i]
        return E
    
    def update_Fluxes(self, Q, mesh):
        for i in range(0, self.num_TotCells):
            self.F1[i] = Q.rhouA[i]
            self.F2[i] = Q.rhouA[i]**2/Q.rhoA[i] + Q.pressure[i]*mesh.area[i]
            self.F3[i] = Q.rhouA[i]*Q.rhoEA[i]/Q.rhoA[i] + Q.pressure[i]*Q.rhouA[i]/Q.rhoA[i]*mesh.area[i]
    