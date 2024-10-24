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

    def apply_BC_left(self, Q):
        if self.BC_left.type == "O_order_extrapolation":
            self.apply_zero_order_extrapolation_left(Q)


    # Extrapolation d'ordre 0 aux ghost cells (côté gauche et droit)
    def apply_zero_order_extrapolation_right(self, Q):    
        # Ghost cell de droite (après la dernière cellule intérieure)
        Q.rhoA[-1] = Q.rhoA[-2]  # Copie de la première cellule intérieure dans la première ghost cell
        Q.rhouA[-1] = Q.rhouA[-2]
        Q.rhoEA[-1] = Q.rhoEA[-2]

    def apply_zero_order_extrapolation_left(self, Q):      
     # Ghost cell de gauche (avant la première cellule intérieure)
        Q.rhoA[0] = Q.rhoA[1]  # Copie de la première cellule intérieure dans la première ghost cell
        Q.rhouA[0] = Q.rhouA[1]
        Q.rhoEA[0] = Q.rhoEA[1]

class BoundaryCondition:
    def __init__(self, type):
        self.type=type