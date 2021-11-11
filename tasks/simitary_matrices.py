import numpy as np
from .distance_functions import manhattan


# Compute the similarity matrix
def get_simitary_matrices(matrix):
    similarity_matrix = np.zeros((len(matrix), len(matrix)))
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            score = manhattan(matrix[i], matrix[j])
            similarity_matrix[i][j] = score
            similarity_matrix[j][i] = score
    return similarity_matrix



