from Thermodynamics import *

class BoundaryConditions:
    def __init__(self, BC_LEFT, BC_RIGHT):
        self.BC_right = BC_RIGHT
        self.BC_left = BC_LEFT

    def apply_BCs(self, Q):
        self.apply_BC_left(Q)
        self.apply_BC_right(Q)

    def apply_BC_right(self, Q):
        if self.BC_right.type == "O_order_extrapolation":
            self.apply_zero_order_extrapolation_right(Q)
        if self.BC_right.type == "SupersonicOutlet":
            self.apply_supersonic_outlet_right(Q)
        if self.BC_right.type == "SubsonicOutlet":
            self.apply_subsonic_outlet_right(Q)


    def apply_BC_left(self, Q):
        if self.BC_left.type == "O_order_extrapolation":
            self.apply_zero_order_extrapolation_left(Q)
        if self.BC_left.type == "SupersonicInlet":
            self.apply_supersonic_inlet_left(Q)

    def apply_supersonic_inlet_left(self, Q):    
        pass # assumes that initialization is done with supersonic inlet conditions


    def apply_zero_order_extrapolation_right(self, Q):    
        # Ghost cell de droite (après la dernière cellule intérieure)
        Q.rhoA[-1] = Q.rhoA[-2]  # Copie de la première cellule intérieure dans la première ghost cell
        Q.rhouA[-1] = Q.rhouA[-2]
        Q.rhoEA[-1] = Q.rhoEA[-2]
    

    def apply_supersonic_outlet_right(self, Q):    
        # Ghost cell de droite (après la dernière cellule intérieure)
        Q.rhoA[-1] = Q.rhoA[-2] # Copie de la première cellule intérieure dans la première ghost cell
        Q.rhouA[-1] = Q.rhouA[-2]
        Q.rhoEA[-1] = Q.rhoEA[-2]

    def apply_subsonic_outlet_right(self, Q): # based on Riemann invariant 
        p_l = Q.pressure[-2]
        rho_l = Q.rhoA[-2]/self.BC_right.mesh.area[-2]
        s = p_l/rho_l**(GAMMA)
        c_l = compute_sound_velocity(Q, self.BC_right.mesh)[-2] # last inner cell sound velocity
        J1 = 2*c_l/(GAMMA-1)+Q.rhouA[-2]/(Q.rhoA[-2])
        c = np.sqrt(GAMMA*self.BC_right.back_pressure**((GAMMA-1)/GAMMA)*s**(1/GAMMA))
        J2 = J1-4/(GAMMA-1)*c
        Q.rhoA[-1] = ((c**2)/(GAMMA*s))**(1/(GAMMA-1))*self.BC_right.mesh.area[-1]
        Q.rhouA[-1] = 0.5*(J2+J1)*Q.rhoA[-1]
        Q.rhoEA[-1] = self.BC_right.back_pressure*self.BC_right.mesh.area[-1]/(GAMMA-1) + 0.5*Q.rhouA[-1]**2/(Q.rhoA[-1])

    def apply_zero_order_extrapolation_left(self, Q):      
     # Ghost cell de gauche (avant la première cellule intérieure)
        Q.rhoA[0] = Q.rhoA[1]  # Copie de la première cellule intérieure dans la première ghost cell
        Q.rhouA[0] = Q.rhouA[1]
        Q.rhoEA[0] = Q.rhoEA[1]

class BoundaryCondition:
    def __init__(self, type, back_pressure=None, mesh=None):
        self.type=type
        self.back_pressure = back_pressure
        self.mesh = mesh