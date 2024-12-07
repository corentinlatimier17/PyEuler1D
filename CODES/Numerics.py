import numpy as np
import copy
from Thermodynamics import GAMMA

def computeJacobianMatrix(Q, i):
    A = np.zeros((3,3), dtype=np.float64)
    Qcell = Q.get_QCell(i)

    A[0][0] = 0.0 # ok
    A[0][1] = 1.0 # ok
    A[0][2] = 0.0 # ok

    A[1][0] = 0.5*(GAMMA-3.0)*(Qcell[1]/Qcell[0])**2 # ok 
    A[1][1] = Qcell[1]/Qcell[0]*(3.0-GAMMA) # ok 
    A[1][2] = GAMMA-1 # ok

    A[2][0] = (GAMMA-1)*(Qcell[1]/Qcell[0])**3-GAMMA*(Qcell[2]*Qcell[1])/Qcell[0]**2 # ok
    A[2][1] = Qcell[2]/Qcell[0]*GAMMA -3/2*(GAMMA-1)*(Qcell[1]/Qcell[0])**2 # ok 
    A[2][2] = Qcell[1]/Qcell[0]*GAMMA #ok
    return A

def computeSourceTermJacobianMatrix(Q, i, MESH): # if we use implicit source term
    A = MESH.area[i]
    dAdx = MESH.dAdx[i]
    Qcell = Q.get_QCell(i)

    B = np.zeros((3,3), dtype=np.float64)
    B[1,0] = (GAMMA-1)/2*(Qcell[1]/Qcell[0])**2
    B[1,1] = -(GAMMA-1)*(Qcell[1]/Qcell[0])
    B[1,2] = GAMMA-1
    B = 1/A*dAdx*B
    return B


class MacCormack:
    def __init__(self):
        pass

    def iterate(self, Q, E, S, MESH, DELTA_T, BCS,LINEAR_SOLVER, isSteady, num_iter=None):
        # Prediction step
        Q_pred = copy.deepcopy(Q)

        for i in MESH.innerCellsIndexes:
            if isSteady:
                deltaT = DELTA_T[i] # if steady, DELTA_T is a vector
            else:
                deltaT = DELTA_T # if transient, DELTA_T is a float
            
            QpredCell = np.zeros((3,1), dtype=np.float64)

            Qcell = Q.get_QCell(i)
            Ecell = E.get_FluxesCell(i)
            Ecell_forward = E.get_FluxesCell(i+1)
            Scell = S.get_SourceTermCell(i)

            QpredCell = Qcell -deltaT/MESH.dx*(Ecell_forward-Ecell) + deltaT*Scell

            Q_pred.rhoA[i] = QpredCell[0]
            Q_pred.rhouA[i] = QpredCell[1]
            Q_pred.rhoEA[i] = QpredCell[2]

        BCS.apply_BCs(Q_pred)
        Q_pred.update_pressure(MESH)
        E.update_Fluxes(Q_pred, MESH)
        S.update_SourceTerm(Q_pred, MESH)

        # Correction step
        Q_corr = copy.deepcopy(Q)

        for i in MESH.innerCellsIndexes:
            QcorrCell = np.zeros((3,1), dtype=np.float64)
            Qcell = Q.get_QCell(i)

            Ecell = E.get_FluxesCell(i)
            Ecell_backward = E.get_FluxesCell(i-1)
            Scell = S.get_SourceTermCell(i)

            QcorrCell = Qcell -deltaT/MESH.dx*(Ecell-Ecell_backward) + deltaT*Scell

            Q_corr.rhoA[i] = QcorrCell[0]
            Q_corr.rhouA[i] = QcorrCell[1]
            Q_corr.rhoEA[i] = QcorrCell[2]

        BCS.apply_BCs(Q_corr)

        Q_prev = copy.deepcopy(Q) # save last solution to compute residuals

        # Update of Q considering prediction and correction 
        Q.rhoA = 0.5*(Q_corr.rhoA + Q_pred.rhoA)
        Q.rhouA = 0.5*(Q_corr.rhouA + Q_pred.rhouA)
        Q.rhoEA = 0.5*(Q_corr.rhoEA + Q_pred.rhoEA)

        BCS.apply_BCs(Q)
        Q.update_pressure(MESH)
        E.update_Fluxes(Q, MESH)
        S.update_SourceTerm(Q, MESH)

        if isSteady:
            res_rhoA = np.linalg.norm(Q_prev.rhoA-Q.rhoA, 2)
            res_rhouA = np.linalg.norm(Q_prev.rhouA-Q.rhouA, 2)
            res_rhoEA = np.linalg.norm(Q_prev.rhoEA-Q.rhoEA, 2)
            return res_rhoA, res_rhouA, res_rhoEA

class LaxWendroff:
    def __init__(self):
        pass

    def iterate(self, Q, E, S, MESH, DELTA_T, BCS, LINEAR_SOLVER, isSteady, num_iter):
        if num_iter <=10:
            MAC_CORMACK = MacCormack()
            return(MAC_CORMACK.iterate(Q, E, S, MESH, DELTA_T, BCS,LINEAR_SOLVER, isSteady, num_iter))
        else:
            Qnew = np.zeros((3, MESH.num_TotCells), dtype=np.float64)
            for i in MESH.innerCellsIndexes:
                if isSteady:
                    deltaT = DELTA_T[i] # if steady, DELTA_T is a vector
                else:
                    deltaT = DELTA_T # if transient, DELTA_T is a float
    
                El = E.get_FluxesCell(i-1)
                Ecell = E.get_FluxesCell(i)
                Er = E.get_FluxesCell(i+1)

                Al = computeJacobianMatrix(Q, i-1)
                A = computeJacobianMatrix(Q, i)
                Ar = computeJacobianMatrix(Q, i+1)

                Alf = 0.5 * (Al + A)
                Arf = 0.5 * (A + Ar)

                Qcell = Q.get_QCell(i)

                Sl = S.get_SourceTermCell(i-1)
                Scell = S.get_SourceTermCell(i)
                Sr = S.get_SourceTermCell(i+1)
                S_prev = S.get_prev_SourceTermCell(i)

                Qnew_vec = np.zeros((3, 1), dtype=np.float64)
                Qnew_vec = Qcell
                Qnew_vec -= deltaT / (2 * MESH.dx) * (Er - El)
                Qnew_vec += deltaT**2 / (2 * MESH.dx**2) * (Arf @ (Er-Ecell) - Alf @ (Ecell-El))
                Qnew_vec += deltaT * Scell
                Qnew_vec += deltaT / 2 * (Scell - S_prev)
                Qnew_vec -= deltaT**2 / (4 * MESH.dx) * (Ar @ Sr - Al @ Sl)

                Qnew[:, i] = Qnew_vec.reshape((3,))

            Q_prev = copy.deepcopy(Q)

            for i in MESH.innerCellsIndexes:
                Q.rhoA[i] = Qnew[0, i]
                Q.rhouA[i] = Qnew[1, i]
                Q.rhoEA[i] = Qnew[2, i]
            
            # Apply boundary conditions and update fluxes
            BCS.apply_BCs(Q)
            Q.update_pressure(MESH)
            E.update_Fluxes(Q, MESH)
            S.update_SourceTerm(Q, MESH)

            if isSteady:
                res_rhoA = np.linalg.norm(Q_prev.rhoA-Q.rhoA, 2)
                res_rhouA = np.linalg.norm(Q_prev.rhouA-Q.rhouA, 2)
                res_rhoEA = np.linalg.norm(Q_prev.rhoEA-Q.rhoEA, 2)
                return res_rhoA, res_rhouA, res_rhoEA

class BeamWarming:
    def __init__(self, epsE=0.125, epsI=2*0.125):
        self.epsE = epsE
        self.epsI = epsI

    def iterate(self, Q, E, S, MESH, DELTA_T, BCS, LINEAR_SOLVER ,isSteady, num_iter=None):
            num_inner_cells = len(MESH.innerCellsIndexes)
            MATRIX_BM = np.zeros((3*num_inner_cells, 3*num_inner_cells), dtype=np.float64) # Reduced block tridiagonal matrix
            RHS_BM = np.zeros((3*num_inner_cells, 1), dtype=np.float64)

            for idx, i in enumerate(MESH.innerCellsIndexes):  # idx -> indices dans la liste / i -> indice dans le maillage
                if isSteady:
                    deltaT = DELTA_T[i] # if steady, DELTA_T is a vector
                else:
                    deltaT = DELTA_T # if transient, DELTA_T is a float
                
                # Get fluxes and Jacobians
                El = E.get_FluxesCell(i-1)
                Er = E.get_FluxesCell(i+1)
                Ecell = E.get_FluxesCell(i)
                Al = computeJacobianMatrix(Q, i-1)
                Ar = computeJacobianMatrix(Q, i+1)

                self.epsI = self.epsI*deltaT
                self.epsE = self.epsE*deltaT
                

                # Get Source Term
                Scell = S.get_SourceTermCell(i)
                B = computeSourceTermJacobianMatrix(Q, i, MESH)


                # Get states
                Qcell = Q.get_QCell(i)
                Qr = Q.get_QCell(i+1)
                Ql = Q.get_QCell(i-1)

                if idx!= 0 and  idx!= num_inner_cells-1:
                    # Center block (diagonal)
                    MATRIX_BM[3*idx:3*(idx+1), 3*idx:3*(idx+1)] += np.identity((3), dtype=np.float64) -deltaT*B
                    # Left block
                    MATRIX_BM[3*idx:3*(idx+1), 3*(idx-1):3*idx] += -1/(2*MESH.dx)*Al
                    # Right block
                    MATRIX_BM[3*idx:3*(idx+1), 3*(idx+1):3*(idx+2)] += 1/(2*MESH.dx)*Ar

                    # Add implicit dissipation (tridiagonal part)
                    if idx > 0:  # Add to the left block
                        MATRIX_BM[3*idx:3*(idx+1), 3*(idx-1):3*idx] -= self.epsI * np.identity(3)
                    if idx < num_inner_cells - 1:  # Add to the right block
                        MATRIX_BM[3*idx:3*(idx+1), 3*(idx+1):3*(idx+2)] -= self.epsI* np.identity(3)

                    # Add implicit dissipation to diagonal
                    MATRIX_BM[3*idx:3*(idx+1), 3*idx:3*(idx+1)] += 2*self.epsI * np.identity(3)

                    # Add explicit fourth-order dissipation to RHS
                    if idx > 1 and idx < num_inner_cells - 2:
                        Qrr = Q.get_QCell(i+2)
                        Qll = Q.get_QCell(i-2)
                        RHS_BM_cell = -deltaT/(2*MESH.dx)*(Er-El) - self.epsE * (Qll - 4*Ql + 6*Qcell - 4*Qr + Qrr) + deltaT*Scell
                    else:  # No dissipation at boundaries
                        RHS_BM_cell = -deltaT/(2*MESH.dx)*(Er-El) + deltaT*Scell
                    RHS_BM[3*idx: 3*(idx+1)] = RHS_BM_cell

                elif idx==0:
                    # Center block (diagonal)
                    MATRIX_BM[3*idx:3*(idx+1), 3*idx:3*(idx+1)] += np.identity((3), dtype=np.float64)
                    # Right block
                    MATRIX_BM[3*idx:3*(idx+1), 3*(idx+1):3*(idx+2)] += 1/(MESH.dx)*Ar
                    # Right hand side 
                    RHS_BM_cell = -deltaT/(MESH.dx)*(Er-Ecell) + deltaT*Scell 
                    RHS_BM[3*idx: 3*(idx+1)] = RHS_BM_cell

                elif idx==num_inner_cells-1:
                    # Center block (diagonal)
                    MATRIX_BM[3*idx:3*(idx+1), 3*idx:3*(idx+1)] += np.identity((3), dtype=np.float64) 
                    # Right hand side 
                    RHS_BM_cell = -deltaT/(MESH.dx)*(Ecell-El) + deltaT*Scell
                    RHS_BM[3*idx: 3*(idx+1)] = RHS_BM_cell

                self.epsI = self.epsI/deltaT
                self.epsE = self.epsE/deltaT

            # Solve for deltaQ for inner cells only
            deltaQ = LINEAR_SOLVER.solve(MATRIX_BM, RHS_BM)
            Q_prev = copy.deepcopy(Q)

            # Update Q for inner cells
            for idx, i in enumerate(MESH.innerCellsIndexes):
                Q.rhoA[i] += deltaQ[3*idx]
                Q.rhouA[i] += deltaQ[3*idx+1]
                Q.rhoEA[i] += deltaQ[3*idx+2]
                    
            # Apply boundary conditions and update fluxes
            BCS.apply_BCs(Q)
            Q.update_pressure(MESH)
            E.update_Fluxes(Q, MESH)
            S.update_SourceTerm(Q, MESH)

            if isSteady:
                res_rhoA = np.linalg.norm(Q_prev.rhoA-Q.rhoA, 2)
                res_rhouA = np.linalg.norm(Q_prev.rhouA-Q.rhouA, 2)
                res_rhoEA = np.linalg.norm(Q_prev.rhoEA-Q.rhoEA, 2)
                return res_rhoA, res_rhouA, res_rhoEA







        









        

