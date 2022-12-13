import numpy as np


def slice_fun(row_idx, col_idx):
    if row_idx.shape[0] == 1 or col_idx.shape[0] == 1:
        li1 = row_idx
        li2 = col_idx
    else:
        li1 = np.kron(row_idx, np.ones((row_idx.shape[0],), dtype=int))
        li2 = np.kron(np.ones((col_idx.shape[0],), dtype=int), col_idx)

    return li1, li2


# row_idx = np.array([1, 2, 3])
# col_idx = np.array([0])
#
# li1, li2 = slice_fun(row_idx, col_idx)
# end