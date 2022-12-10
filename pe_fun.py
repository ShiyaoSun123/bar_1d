import numpy as np
from gauss_1d import guass_1d
from shape_fun import shape_fun


# taue input as column
def pe_fun(taue, nen, xe, nipt):
    wts, xk = guass_1d(nipt)
    order = nen - 1
    pe = np.zeros((1, nen))
    for i in range(nipt):
        N, _, jac = shape_fun(xk[i], order, xe)
        txi = N @ taue
        pe = pe + (txi * jac * wts[i]) * N.T
    return pe


xe = np.array([0, 10, 5]).T
taue = np.array([1, 1, 1]).T
nen = 3
nipt = 2

pe = pe_fun(taue, nen, xe, nipt)

a = 1
