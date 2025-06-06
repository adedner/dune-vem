import numpy as np
import scipy.sparse as sps

# https://scicomp.stackexchange.com/a/25241
def fixDirichlet(matrix, dirichletBlocks, diag):
    """zero-out rows and columns associated with dirichlet dofs,
       and put 'diag' on the diagonal there."""
    # dirichlet_zero_dofs = [i for i,b in enumerate(dirichletBlocks) if b[0]==1]
    dirichlet_zero_dofs = []
    for i,block in enumerate(dirichletBlocks):
        for j,b in enumerate(block):
            if b>0:
                dirichlet_zero_dofs += [i*len(block)+j]

    M = matrix.shape[0]
    N = matrix.shape[1]

    m = len(dirichletBlocks)*len(dirichletBlocks[0])
    assert m==M or m==N

    chi_interior = np.ones(m)
    chi_interior[dirichlet_zero_dofs] = 0.0
    I_interior = sps.spdiags(chi_interior, [0], m, m).tocsr()

    if M == N:
        chi_boundary = np.zeros(n)
        chi_boundary[dirichlet_zero_dofs] = diag
        I_boundary = sps.spdiags(chi_boundary, [0], n, n).tocsr()
        matrix_modified = I_interior @ matrix @ I_interior + I_boundary
    elif matrix.shape[1] == I_interior.shape[0]:
        matrix_modified = matrix @ I_interior
    else:
        assert I_interior.shape[1] == matrix.shape[0]
        matrix_modified = I_interior @ matrix

    assert matrix_modified.shape == matrix.shape
    return matrix_modified.tocsr()

def applyDBCs(A,diag,scheme):
    try:
        dbc = scheme.dirichletBlocks
        return fixDirichlet(A, dbc, diag)
    except AttributeError:
        return A
