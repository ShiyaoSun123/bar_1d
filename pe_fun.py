import numpy as np
from gauss_1d import guass_1d
from shape_fun import shape_fun


# taue input as column
def pe_fun(taue, nen, xe, nipt):
    wts, xk = guass_1d(nipt)
    order = nen - 1
    pe = np.zeros((nen, 1))
    for i in range(nipt):
        N, _, jac = shape_fun(xk[i], order, xe)
        txi = N @ taue
        pe = pe + (txi * jac * wts[i]) * N.T
    return pe


# xe = np.array([0, 1, 0.5])
#
# nen = 3
# nipt = 2
# taue = 5*np.ones((nen, 1))
# pe = pe_fun(taue, nen, xe, nipt)
#
#
# end
