import numpy as np
# ID input as 1d array, default same as node numbering

def mesh_gen(L, nele, order, ID):
    # node numbering to dof mapping
    nen = nele * order + 1
    node_num = np.arange(nen)
    dof_map = {node_num[i]: ID[i] for i in range(nen)}

    # generate external nodes coordinates
    coord_ext = np.linspace(0, L, nele + 1)
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

    #use dof mapping to convert conn to Location Matrix
    LM = np.array([[dof_map[val] for val in sub] for sub in conn])


    #create coord to node_ele_num mapping
    coord_map = {node_num[i]: coord[i] for i in range(nen)}

    #use coord_map to map LM to coordinate matrix
    coordM = np.array([[coord_map[val] for val in sub] for sub in conn])

    return coordM, conn, LM


#test
nele = 3
order = 2
L = 10
ID = np.flip(np.arange(nele * order + 1))
coordM, conn, LM = mesh_gen(L ,nele ,order, ID)

end