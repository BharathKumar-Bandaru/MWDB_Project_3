import numpy as np
from numpy import linalg


# Compute latent semantics using svd
def compute_svd(data_matrix, k = -1):
    """
    data_matrix: n*m data with n objects and m features (expected type: matrix) 
    k (optional): no. of latent semantics needed
    returns:
    U: n * min(n,m)
    S: 1d - diag elements of size min(n,m)
    Vt: min(n,m) * m
    
    """
    print(f'Computing SVD...')
    data_matrix = np.matrix(data_matrix)
    ddt_eig_values, ddt_eig_vectors = linalg.eig(data_matrix * data_matrix.T) #gives col-wise eigen vectors
    dtd_eig_values, dtd_eig_vectors = linalg.eig(data_matrix.T * data_matrix)
    
    n = data_matrix.shape[0]
    m = data_matrix.shape[1]

    ddt_eig_vectors = ddt_eig_vectors[:, :min(n,m)]
    ddt_eig_values = ddt_eig_values[:min(n,m)]

    dtd_eig_vectors = dtd_eig_vectors[:, :min(n,m)]
    dtd_eig_values = dtd_eig_values[:min(n,m)]

    ddt_sort_order = ddt_eig_values.argsort()[::-1] #getting sort order of ddt singular vectors by descending order of ddt singular values
    dtd_sort_order = dtd_eig_values.argsort()[::-1] #getting sort order of dtd singular vectors by descending order of dtd singular vaules
    
    eigen_values = ddt_eig_values[ddt_sort_order]
    #alternatively:
    #eigen_values = dtd_eig_values[dtd_sort_order]

    singular_values = eigen_values ** (1/2)
    
    U = ddt_eig_vectors[:, ddt_sort_order] #sorting ddt eigen vectors in the descending order of ddt eigen values
    V = dtd_eig_vectors[:, dtd_sort_order] #sorting dtd eigen vectors in the descending order of dtd eigen values
    
    U = np.array(np.real(U))
    S = np.array(np.real(singular_values))
    Vt = np.array(np.real(V.T))

    if k > 0:
        return U[:, :k], S[:k], Vt[:k, :]
    else:
        return U, S, Vt