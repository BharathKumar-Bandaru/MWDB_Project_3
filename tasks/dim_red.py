from .features import *
from .pca import compute_pca
from .svd import compute_svd
from .kmeans import kmeans
from .lda import LDA

# Performs the dimentionality reduction
def perform_dim_red(dim_red_technique, image_features, k):
    if dim_red_technique == 'pca':
        left_factor_matrix, right_factor_matrix = compute_pca(image_features, k)
        return left_factor_matrix, right_factor_matrix

    elif dim_red_technique == 'svd':
        left_factor_matrix, core_matrix, right_factor_matrix = compute_svd(image_features, k)
        return left_factor_matrix, right_factor_matrix

    elif dim_red_technique == 'lda':
        print("LDA calculating.")
        lda = LDA(k, image_features)
        right_factor_matrix = lda.get_image_features_latent_semantics()
        left_factor_matrix = lda.transform_data()
        print("LDA done.")
        return left_factor_matrix, right_factor_matrix


    elif dim_red_technique == 'kmeans':
        max_iterations = 500
        kmeans_cluster = kmeans(np.transpose(np.array(image_features)), k, max_iterations)
        newDataMatrix = np.zeros((len(image_features), k))
        for i in range(len(image_features)):
            newLT = np.zeros(k)
            for j in range(len(image_features[i])):
                newLT[kmeans_cluster[j]] += image_features[i][j]
            newDataMatrix[i] = newLT
        print(f"K-means cluster shape: {kmeans_cluster.shape}")
        print(f"K-means data shape: {newDataMatrix.shape}")
        print(f"K-means image_features shape: {np.matrix(image_features).shape}")
        right_factor_matrix = np.matmul(np.transpose(np.array(image_features)), newDataMatrix)
        print(f"K-means right_factor_matrix shape: {right_factor_matrix.shape}")
        return newDataMatrix, np.transpose(right_factor_matrix)