from .ppr import PersonalizedPageRank
import numpy as np

class PersonalizedPageRankClassifier:
    def __init__(self, input_image_objects, test_image_objects, classification_label = 'type',
                 num_nodes_to_consider_for_classifying=10, num_similar_nodes_to_form_graph = 10, beta = 0.15):
        self.input_image_objects = input_image_objects
        self.test_image_objects = test_image_objects
        self.num_similar_nodes = num_similar_nodes_to_form_graph
        self.beta = beta
        self.num_nodes_to_consider_for_classifying = num_nodes_to_consider_for_classifying
        self.classification_label = classification_label

    def get_classified_labels(self):
        for test_image in self.test_image_objects:
            ppr_obj = PersonalizedPageRank(self.input_image_objects, test_image, self.num_similar_nodes, self.beta)
            ppr_scores = ppr_obj.get_ppr_scores_for_input_images()
            image_indices = np.argsort(ppr_scores)[::-1][:self.num_nodes_to_consider_for_classifying]
            label_dict = {}
            for index in image_indices:
                img_obj = self.input_image_objects[index]
                label_val = getattr(img_obj, self.classification_label)
                if label_val not in label_dict:
                    label_dict[label_val] = 0
                label_dict[label_val] += 1
            print(label_dict)




