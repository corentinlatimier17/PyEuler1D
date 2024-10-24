import numpy as np

GAMMA = 1.4

def compute_pressure(Q, mesh):
    return (GAMMA-1)*(Q.rhoEA/mesh.area-0.5*Q.rhouA**2/(Q.rhoA*mesh.area))

def compute_sound_velocity(Q, mesh):
    area = mesh.area[mesh.innerCellsIndexes]  # Access inner cell areas if necessary
    density = Q.rhoA[mesh.innerCellsIndexes]/area  # Access inner cell densities
    pressure = Q.pressure[mesh.innerCellsIndexes]  # Access inner cell pressures

    # Calculate sound velocity for inner cells
    c = np.sqrt(GAMMA * pressure / density)
    return c
