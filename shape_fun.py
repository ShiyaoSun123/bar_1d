import numpy as np
def shape_fun(z, order, xe):   #xe input as column vector
    if order == 1:
        N = np.array([0.5*(1-z), 0.5*(1+z)])
        jac = np.array([-0.5, 0.5]) @ xe
        dN_dx = np.array([-0.5, 0.5])/jac
        N.reshape((1, -1))
        dN_dx.reshape((1, -1))

        return N, dN_dx, jac

    if order == 2:
        N = np.array([(0.5*z**2 - 0.5*z), (0.5*z**2 + 0.5*z), (1-z**2) ])
        jac = np.array([z-0.5, z+0.5, -2*z])@xe
        dN_dx = np.array([z-0.5, z+0.5, -2*z])/jac

        N.reshape((1, -1))
        dN_dx.reshape((1, -1))

        return N, dN_dx, jac

    if order == 3:
        N = np.array([-6/19*(z+1/3)*(z-1/3)*(z-1), 6/19*(z+1/3)*(z-1/3)*(z+1), 27/16*(z+1)*(z-1/3)*(z-1), -27/16*(z+1)*(z+1/3)*(z-1)])
        jac = np.array([-2/57*(27*z**2 - 18*z -1), 2/57*(27*z**2 + 18*z -1), 9/16*(9*z**2 - 2*z - 3), -9/16*(9*z**2 + 2*z - 3)])@xe
        dN_dx = np.array([-2/57*(27*z**2 - 18*z -1), 2/57*(27*z**2 + 18*z -1), 9/16*(9*z**2 - 2*z - 3), -9/16*(9*z**2 + 2*z - 3)])/jac

        N.reshape((1, -1))
        dN_dx.reshape((1, -1))

        return N, dN_dx, jac







