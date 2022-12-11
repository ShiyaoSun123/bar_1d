import numpy as np
from gauss_1d import guass_1d
from shape_fun import shape_fun

def ke_fun(xe, nipt, E, A):
    wts, xk = guass_1d(nipt)
    nen = nipt + 1
    ke = np.zeros((nen, nen))

    for i in range(nipt):
        _, dN_dx, jac = shape_fun(xk[i], nipt, xe)

        ke = ke + E * A * wts[i] * jac * dN_dx.T @ dN_dx

    return ke

E = 1
A = 1
nipt = 1
xe = np.array([0,1]).T

ke = ke_fun(xe, nipt, E, A)

end
