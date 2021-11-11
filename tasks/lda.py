import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import scale


# COmpute LDA tasks
class LDA:
    def __init__(self, k_components, image_features):
        self.k_components = k_components
        self.image_feature = image_features
        self.lda_func = LatentDirichletAllocation(n_components=k_components).fit(image_features)

    def get_image_features_latent_semantics(self):
        return self.lda_func.components_
    #kxm

    def transform_data(self):
        self.reduced_matrix = self.lda_func.transform(self.image_feature)
        return self.reduced_matrix[:, :self.k_components]
    #nxk
