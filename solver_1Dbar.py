import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

from mesh_gen import mesh_gen
from ke_fun import ke_fun
#from shape_fun import shape_fun
from pe_fun import pe_fun
from slice_fun import slice_fun
#from gauss_1d import gauss_1d

#u_supp is prescribed displacement at support
#supp_dof is the support dof
#load_free are the point loads at the free dofs
#ID is dof numbering
def solver_1Dbar(nele, nipt, L, E, A, supp_dof,u_supp, ID, taue):
    ndof = len(ID)
    nsupp_dof = len(supp_dof)
    nen = nipt + 1
    nfree_dof = ndof - nsupp_dof

    free_dof = np.setdiff1d(ID, supp_dof)

    #assemble global K , P
    K = np.zeros((ndof, ndof))
    P = np.zeros((ndof, 1))

    for i in range(nele):
        coordM, conn, LM = mesh_gen(L, nele, nipt, ID)
        xe = coordM[:, i].flatten()
        ke = ke_fun(xe, nipt, E, A).flatten()
        pe = pe_fun(taue, nen, xe, nipt)

        P[LM[:, i]] = pe + P[LM[:, i]]
        Pf = P[free_dof]


        K[slice_fun(LM[:, i], LM[:, i])] = ke + K[slice_fun(LM[:, i], LM[:, i])]

        Kff = K[slice_fun(free_dof, free_dof)].reshape(free_dof.shape[0], free_dof.shape[0])
        Kss = K[slice_fun(supp_dof, supp_dof)].reshape(supp_dof.shape[0], supp_dof.shape[0])
        Kfs = K[slice_fun(free_dof, supp_dof)].reshape(free_dof.shape[0], supp_dof.shape[0])
        Ksf = K[slice_fun(supp_dof, free_dof)].reshape(supp_dof.shape[0], free_dof.shape[0])

    Uf = inv(Kff) @ (Pf - Kss @ u_supp)

    return Uf

nele = 3
nipt = 2
nen = nipt + 1

#uniform load
taue = 5 * np.ones((nen, 1))
L = 10
E = 100
A = 1

supp_dof = np.array([0])
u_supp = np.array([0])
ID = np.arange(nele * nipt + 1)

Uf = solver_1Dbar(nele, nipt, L, E, A, supp_dof,u_supp, ID, taue)


end
