from .input_output import *
from .features import *
from .dim_red import perform_dim_red
from numpy import genfromtxt
from .custom_svm import *
from .svm import *
from .ppr_task import perform_ppr_classification
from tasks.decision_trees import *
from sklearn.metrics import confusion_matrix, classification_report #for reporting the model performance
from sklearn.metrics import confusion_matrix #for reporting the model performance
import pandas as pd
from tasks.cache import get_cache_for_input_images

cache_for_input_images = {} #input_folder_path, {'images_with_attributes': , 'images':, 'image_features':, 'latent_semantics_file_path': }
svm_models = {}
decision_tree_models = {}
cache_for_input_images = get_cache_for_input_images() #{input_folder_path}_{feature_model}_{dim_red_technique}_{k}, {'images_with_attributes': , 'images':, 'image_features':, 'latent_semantics_file_path': }

# Entry for tasks 1,2, and 3
def task_1_2_3(task_number, input_folder_path, feature_model, k, test_folder_path, classifier,
               dim_red_technique = 'svd', output_folder = 'output', latent_semantics_file_name = None,
               use_cached_input_images = True):
    key = f'{input_folder_path}_{feature_model}_{dim_red_technique}_{k}'
    latent_semantics = None
    if use_cached_input_images is True and key in cache_for_input_images:
        print('Using existing cache for latent semantics')
        images_with_attributes = cache_for_input_images[key]['images_with_attributes']
        image_features = cache_for_input_images[key]['image_features']
        latent_semantics_file_path = cache_for_input_images[key]['latent_semantics_file_path']
        latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
    else:
        images_with_attributes = get_images_and_attributes_from_folder(input_folder_path)
        images = get_image_arr_from_dict(images_with_attributes)
        image_features = get_flattened_features_for_images(images, feature_model)

        if latent_semantics_file_name is None:
            latent_semantics_file_name = f'{input_folder_path}_{feature_model}_{dim_red_technique}_{k}_latent_semantics.csv'

        latent_semantics_file_path = os.path.join(output_folder, latent_semantics_file_name)
        if os.path.exists(latent_semantics_file_path):
            print('Reading stored csv for latent semantics')
            latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
        else:
            dim_red_technique = dim_red_technique.lower()
            if k != "all":
                left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)
                latent_semantics = right_factor_matrix
                store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)
            else:
                left_factor_matrix = latent_semantics = np.array(image_features)
                store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)

        #Storing into cache
        cache_for_input_images[key] = {'images_with_attributes': images_with_attributes,
                                                     'images': images,
                                                     'image_features': image_features,
                                                     'latent_semantics_file_path': latent_semantics_file_path}

    test_images_with_attributes = get_images_and_attributes_from_folder(test_folder_path)
    test_images = get_image_arr_from_dict(test_images_with_attributes)
    test_image_features = get_flattened_features_for_images(test_images, feature_model)

    if task_number == 1:
        label_name = 'type'
    elif task_number == 2:
        label_name = 'subject_id'
    else:
        label_name = 'image_id'

    classifier = classifier.lower()
    if classifier == 'decision-tree':
        predicted_labels, correct_labels = decision_tree(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features, label_name, f'{key}_{task_number}', k)

    elif classifier == 'svm':
        predicted_labels, correct_labels = svm(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features, label_name, f'{key}_{task_number}', k)

        # string_ = f"{key}: {np.sum(predicted_labels == correct_labels) / len(predicted_labels) * 100:.2f}, {test_folder_path}_Correct: {np.sum(predicted_labels == correct_labels)}."
        # return string_

    elif classifier == 'ppr':
        predicted_labels, correct_labels = ppr(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features, label_name)

    print_classification_stats(predicted_labels, correct_labels) #print false positive rate and false negative rate

def decision_tree(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features, label_name, key, k):
    print('Decision Tree Classifier')

    """
    Write Decision Tree code here
    """

    predicted_labels = []
    correct_labels = process_labels(test_images_with_attributes, label_name)
    image_labels = get_label_arr_from_dict(images_with_attributes, label_name)
    dc = DecisionTreeClassifier()

    if k!="all":
        dataset_features = np.matmul(image_features, np.transpose(latent_semantics))
        images_att_new_space = np.matmul(test_image_features, latent_semantics.transpose())
    else:
        images_att_new_space = test_image_features
        dataset_features = image_features

    if key not in decision_tree_models.keys():
        root = dc.make_tree(dataset_features.tolist())
        print("Tree is formed.")
    else:
        root = decision_tree_models[key]
        print("Using the root from cache.")

    count = 0
    totalcount = 0
    for i in range(len(images_att_new_space)):
        pre = dc.predict(images_att_new_space[i], root)
        if int(pre) == int(correct_labels[i]):
            count+=1
        predicted_labels.append(pre)
        totalcount+=1
    return predicted_labels, correct_labels

def svm(latent_semantics, images_with_attributes, image_features,
                                         test_images_with_attributes, test_image_features, label_name, key, k):
    print('Support Vector Machine Classifier')
    """
        Write SVM code here
    """
    correct_labels = process_labels(test_images_with_attributes, label_name)
    image_features = np.array(image_features)
    test_image_features = np.array(test_image_features)

    # Pre processing inputs
    mean = np.mean(image_features, axis=0)
    image_features -= mean
    test_image_features -= mean
    if k!= "all":
        X_data = np.array(np.matmul(image_features, latent_semantics.transpose()))
        X_test = np.array(np.matmul(test_image_features, latent_semantics.transpose()))
    else:
        X_data = np.array(latent_semantics)
        X_test = np.array(test_image_features)

    if key not in svm_models.keys():
        print("Training the model")
        svm_ = SVM_custom(C=1, tol=0.01, max_iter=100, random_state=0, verbose=1)
        # svm_ = linearSVM()
        original_labels = process_labels(images_with_attributes, label_name)
        svm_.train(X_data, original_labels)
        svm_models[key] = svm_
        print("Stored the model")
    else:
        print(f"Using the model {key} from cache")
        svm_ = svm_models[key]
    # Predict the labels
    predicted_labels = svm_.predict(X_test)
    return np.array(predicted_labels), np.array(correct_labels)


def ppr(latent_semantics, images_with_attributes, image_features,
                                         test_images_with_attributes, test_image_features, label_name):
    print('Personalized Page Rank Classifier')

    #predicted_labels = None
    #correct_labels = None

    predicted_labels, correct_labels = perform_ppr_classification(latent_semantics, images_with_attributes, image_features,
                                         test_images_with_attributes, test_image_features, label_name)
    return predicted_labels, correct_labels

def print_classification_stats(predicted_labels, correct_labels) :
    correct_labels_count = 0
    for i in range(len(predicted_labels)):
        # print(f'{i+1}: True Label -  {correct_labels[i]}, Assigned Label - {predicted_labels[i]}')
        if correct_labels[i] == predicted_labels[i]:
            correct_labels_count += 1
    print('Accuracy is ' + str(correct_labels_count * 100 / len(predicted_labels)) + '%')

    """CM = confusion_matrix(correct_labels, predicted_labels, target_classes = predicted_labels)
    print(CM)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    print('False Positive rate - ' + str(FPR))
    print('False Negative rate - ' + str(FNR))"""

    unique_label = np.unique([correct_labels, predicted_labels])
    cmtx = pd.DataFrame(
        confusion_matrix(correct_labels, predicted_labels, labels=unique_label),
        index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx.to_string())

"""
def task1_2_3(feature_model, filter, image_type, k, dim_red_technique,
            folder_path='input_images', output_folder='output',
            latent_semantics_file_name='task1_latent_semantics.csv'):

    images = get_image_arr_from_dict(images_with_attributes)
    image_features = get_flattened_features_for_images(images, feature_model)
    if feature_model == "cm" and dim_red_technique == "lda":
        feature_max_value = np.max(image_features)
        image_features = image_features + feature_max_value
    left_factor_matrix = core_matrix = right_factor_matrix = None

    dim_red_technique = dim_red_technique.lower()
    left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)
    perform_post_operations(images_with_attributes, left_factor_matrix, right_factor_matrix, output_folder,
                            latent_semantics_file_name, filter)


# Perform operations to save the latent semantics
def perform_post_operations(images_with_attributes, left_factor_matrix, right_factor_matrix, output_folder,
                            latent_semantics_file_name, filter, 
                            subject_weight_matrix_file_name = 'task1_subject_weight_matrix.csv', 
                            type_weight_matrix_file_name = 'task2_type_weight_matrix.csv'):
    latent_semantics = right_factor_matrix
    store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)

    if filter == "type":
        subject_weight_matrix = np.array(get_subject_weight_matrix(images_with_attributes, left_factor_matrix))
        store_array_as_csv(subject_weight_matrix, output_folder, subject_weight_matrix_file_name)        
        sort_print_matrix(subject_weight_matrix, k=len(subject_weight_matrix[0]))
    elif filter == "subject_id":
        type_weight_matrix = np.array(get_type_weight_matrix(images_with_attributes, left_factor_matrix))
        store_array_as_csv(type_weight_matrix, output_folder, type_weight_matrix_file_name)   
        sort_print_matrix(type_weight_matrix, len(type_weight_matrix[0]), "t")


# Print the sorted subject weight or task weight based on the inputs called in the above method.
def sort_print_matrix(subject_weight_matrix, k, one="s"):
    sorted_weights_for_each_latent_semantic = {}
    subject_weight_matrix = np.array(subject_weight_matrix).transpose()
    for i, val in enumerate(subject_weight_matrix):
        latent_sematic = f"l{i}"
        subject_weight_sort = {}
        for j, v in enumerate(val):
            subject = f"{one}{j}"
            subject_weight_sort[subject] = v
        subject_weight_sort = sorted(subject_weight_sort.items(), key=lambda kv: kv[1], reverse=True)
        sorted_weights_for_each_latent_semantic[latent_sematic] = subject_weight_sort
        print(f"{latent_sematic} - {sorted_weights_for_each_latent_semantic[latent_sematic]}")

"""