import numpy as np

def mesh_gen(L, nele, order):
    # generate external nodes coordinates
    coord_ext = np.linspace(0, L, nele+1)
    dx = L/nele/order
    coord_int = np.arange(0, L+dx, dx)
    coord_int = np.delete(coord_int, np.arange(0, coord_int.size, order))

    coord = np.append(coord_ext, coord_int)


    conn = np.zeros((order+1,nele), dtype=np.int32)
    conn[0][0] = 0
    conn[1][0] = 1

    for k in range(2, order+1):
        conn[k][0] = nele + k - 1

    for i in range(order+1):
        for j in range(1, nele):
            if i <= 1:
                conn[i, j] = conn[i, j - 1] + 1

            else:
                conn[i, j] = conn[i, j-1] + order -1
    return coord, conn








#test
coord, conn = mesh_gen(10 ,4 ,2)

end