import numpy as np   

class mesh:
    def __init__(self, N, xmin, xmax):
        self.dx = (xmax - xmin) / N  # Cell size

        self.coordinates = np.linspace(xmin + self.dx/2, xmax - self.dx/2, N, dtype=np.float64)  # Offset by dx/2

        self.area = None
        self.dAdx = None
        self.num_InnerCells = self.coordinates.shape[0]

        # Add ghost cells
        ghost_left = self.coordinates[0] - self.dx  # One ghost cell on the left
        ghost_right = self.coordinates[-1] + self.dx  # One ghost cell on the right
        self.coordinates = np.concatenate(([ghost_left], self.coordinates, [ghost_right]))

        self.innerCellsIndexes = np.arange(1, N+1)  # Only valid if 2 ghost cells
        self.num_TotCells = self.coordinates.shape[0]

    def get_dx(self):
        return self.dx
    
    def get_coordinates(self):
        return self.coordinates
    
    def get_InnerCellsIndexes(self):
        return self.innerCellsIndexes
    
    def isGhostCell(self):
        """Returns an array of True/False where True indicates a ghost cell"""
        ghost_mask = np.ones(self.num_TotCells, dtype=bool)  # Initialize all as True (ghost cells)
        
        # Mark inner cells (indices 1 to N) as False (not ghost cells)
        ghost_mask[self.innerCellsIndexes] = False  
        
        return ghost_mask

    def init_mesh_SOD(self, func_area, func_dadx):
        self.area = func_area(self.coordinates)
        self.dAdx = func_dadx(self.coordinates)








    




