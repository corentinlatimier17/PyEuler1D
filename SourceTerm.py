import numpy as np

class SourceTerm:
    def __init__(self, num_TotCells):
        self.S1 = np.zeros(num_TotCells, dtype=np.float64)
        self.S2 = np.zeros(num_TotCells, dtype=np.float64)
        self.S3 = np.zeros(num_TotCells, dtype=np.float64)
        self.num_TotCells = num_TotCells
        self.S1_prev = None
        self.S2_prev = None
        self.S3_prev = None

    def update_SourceTerm(self, Q, mesh):
        self.S1_prev = self.S1
        self.S2_prev = self.S2
        self.S3_prev = self.S3

        self.S2 = Q.pressure*mesh.dAdx

    def get_SourceTermCell(self, i):
        Scell = np.zeros((3,1), dtype=np.float64)
        Scell[0] = self.S1[i]
        Scell[1] = self.S2[i]
        Scell[2] = self.S3[i]
        return Scell
    
    def get_prev_SourceTermCell(self, i):
        Scell = np.zeros((3,1), dtype=np.float64)
        Scell[0] = self.S1_prev[i]
        Scell[1] = self.S2_prev[i]
        Scell[2] = self.S3_prev[i]
        return Scell