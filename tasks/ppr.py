from scipy.spatial import distance
import numpy as np
from numpy import linalg

class PersonalizedPageRank:
    def __init__(self, input_image_objects, test_image_object, num_similar_nodes_to_form_graph = 10, beta = 0.15):
        self.input_image_objects = input_image_objects
        self.test_image_object = test_image_object
        self.num_similar_nodes = num_similar_nodes_to_form_graph
        self.beta = beta
        self.ppr_scores = None
        self.initialize()

    def initialize(self):
        """
        self.image_filenames = []
        self.dict = {}
        self.seed_nodes = []
        for img_obj in self.input_image_objects:
            self.image_filenames.append(img_obj.filename)
            self.dict[img_obj.filename] = img_obj
        self.image_filenames.append(self.test_image_object.filename)
        self.dict[self.test_image_object.filename] = self.test_image_object
        seed_node_index = len(self.image_filenames) - 1
        self.seed_nodes = [seed_node_index]

        self.n = len(self.image_filenames)
        self.image_similarity_matrix = None
        """
        self.image_objects = []
        self.image_objects.extend(self.input_image_objects)
        self.image_objects.append(self.test_image_object)
        self.num_nodes = len(self.image_objects)
        seed_node_index = self.num_nodes - 1
        self.seed_nodes = [seed_node_index]
        self.image_similarity_matrix = None

    def compute_image_similarity_matrix(self):
        self.image_similarity_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dis = self.compute_distance(self.image_objects[i], self.image_objects[j])
                self.image_similarity_matrix[i][j] = dis
                self.image_similarity_matrix[j][i] = dis

    def get_image_similarity_matrix(self):
        if self.image_similarity_matrix is None:
            self.compute_image_similarity_matrix()

        return self.image_similarity_matrix

    def compute_ppr_scores(self, similarity_matrix, seed_nodes, num_similar_nodes = 10, beta = 0.15):

        numNodes = len(similarity_matrix)
        num_similar_nodes = min(num_similar_nodes, numNodes)
        transition_matrix_TG = np.zeros((numNodes, numNodes))

        for node in range(numNodes):
            similar_nodes = self.getTopSimilarObjects(node, num_similar_nodes, similarity_matrix)
            for similar_node in similar_nodes:
                transition_matrix_TG[node][similar_node] = 1 / num_similar_nodes

        print('Calculating Robust Personalized Page Rank scores relative to the seed nodes..')

        # Step 1
        Pi_dict = {}  # key = seed_node, value = Pi_i vector

        for seed_node in seed_nodes:
            Si = np.matrix(np.zeros((numNodes, 1)))
            Si[seed_node] = 1

            a = ((1 - beta) * transition_matrix_TG)
            I = np.identity(transition_matrix_TG.shape[0])
            inv = linalg.inv(I - a)
            Pi_i = inv * beta * Si
            Pi_dict[seed_node] = Pi_i

        # Step 2
        seed_scores_dict = {}
        for seed_node in seed_nodes:
            Pi_i = Pi_dict[seed_node]
            summation = 0
            for node in seed_nodes:
                summation += Pi_i[node].item()
            seed_scores_dict[seed_node] = summation

        # Step 3
        S_crit = []
        max_seed_score = seed_scores_dict[max(seed_scores_dict, key=seed_scores_dict.get)]
        for seed_node in seed_nodes:
            if seed_scores_dict[seed_node] == max_seed_score:
                S_crit.append(seed_node)

        # Step 4
        # RPR-2 Scores (Robust Personalized Page Rank Scores)

        PPR_Scores_matrix = sum([Pi_dict[node] for node in S_crit]) / len(S_crit) # Taking average of all scores in S_crit
        PPR_Scores = [scoreMatrix.item() for scoreMatrix in PPR_Scores_matrix]
        self.ppr_scores = PPR_Scores

    def getTopSimilarObjects(self, node, num_similar_nodes, similarity_matrix):
        n = num_similar_nodes
        nodes_in_order = np.argsort(similarity_matrix[node])
        top_similar_nodes = nodes_in_order[:n]
        return top_similar_nodes

    def compute_distance(self, image_obj_1, image_obj_2):
        dis = float('inf')
        if image_obj_1.latent_features is not None and image_obj_2.latent_features is not None:
            dis = distance.euclidean(image_obj_1.latent_features, image_obj_2.latent_features)
        return dis

    def get_ppr_scores_for_input_images(self):
        if self.ppr_scores is None:
            image_similarity_matrix = self.get_image_similarity_matrix()
            self.compute_ppr_scores(similarity_matrix = image_similarity_matrix, seed_nodes = self.seed_nodes)

        return self.ppr_scores[:len(self.input_image_objects)] #excluding the seed nodes