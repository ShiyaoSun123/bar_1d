import numpy as np
from numpy.linalg import inv

from mesh_gen import mesh_gen
from ke_fun import ke_fun
from pe_fun import pe_fun
from slice_fun import slice_fun

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
    coordM, conn, LM = mesh_gen(L, nele, nipt, ID)
    n_K_comp = nen * nen * nele
    n_ke_comp = nen * nen

    ke_vec = np.zeros((1,n_K_comp)).flatten()
    rowidx_vec = np.zeros((1, n_K_comp)).flatten()
    colidx_vec = np.zeros((1, n_K_comp)).flatten()

    for i in range(nele):
        xe = coordM[:, i].flatten()
        ke = ke_fun(xe, nipt, E, A)
        ke_hstack = np.hstack(ke)

        a = i * n_ke_comp
        b = (i + 1) * n_ke_comp
        ke_vec[a : b] += ke_hstack

        row_col_idx = LM[:, i]
        li1, li2 = slice_fun(row_col_idx, row_col_idx)
        rowidx_vec[a : b] += li1
        colidx_vec[a : b] += li2



        pe = pe_fun(taue, nen, xe, nipt)
        P[LM[:, i]] = pe + P[LM[:, i]]
        Pf = P[free_dof].view()




    Kff = K[np.ix_(free_dof, free_dof)].view()
    Kss = K[np.ix_(supp_dof, supp_dof)].view()
    Kfs = K[np.ix_(free_dof, supp_dof)].view()
    Ksf = K[np.ix_(supp_dof, free_dof)].view()




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



ID = np.flip(np.arange(nele * nipt + 1))

Uf = solver_1Dbar(nele, nipt, L, E, A, supp_dof,u_supp, ID, taue)
