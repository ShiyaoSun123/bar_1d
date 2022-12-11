import numpy as np
from mesh_gen import mesh_gen
from ke_fun import ke_fun
from shape_fun import shape_fun
from pe_fun import pe_fun
from gauss_1d import gauss_1d

#u_supp is prescribed displacement at support
#supp_dof is the support dof
#load_free are the point loads at the free dofs

def solver_1Dbar(nele, nipt, L, E, A, supp_dof, ID):
    ndof = len(ID)
    nsupp_dof = len(supp_dof)
    nfree_dof = ndof - nsupp_dof

    #assemble global K , P
    K = np.zeros((ndof, ndof))
    P = np.zeros((ndof, 1))

    for i in range(nele):
        coordM, conn, LM = mesh_gen(L, nele, nipt, ID)
        xe = coordM[i]
        ke = ke_fun(xe, nipt, E, A)
        K[LM[:,i], LM[:,i]] = ke + K[LM[:,i], LM[:,i]]
        P[LM[:,i]] = pe + P[LM[:,i]]

    return K, P

nele = 3
nipt = 1
L = 10
E = 1
A = 1
supp_dof = np.array([6])
K, P = solver_1Dbar(nele, nipt, L, E, A, supp_dof, ID)
