import math
from tasks.input_output import *
total_buckets = []
def compute_number_of_bits(b, d):
    # v = number of vectors
    # b = number of bits
    # d = number of dimensions
    bits = []

    for j in range(1, d+1):
        if j <= b%d:
            bj = math.floor(b/d) + 1
        else:
            bj = math.floor(b/d)+0
        bits.append(bj)

    return bits

def distribute_partition_points(feature_vector, dim_num, bits):
    #ex: for dim_num = 0 concatenate the first col values of all the data
    col = []
    for i in range(0, len(feature_vector)):
        col.append(feature_vector[i][dim_num])
    col.sort()

    partition_points = []
    partition_points.append(int(col[0]))
    # print("Col size is: ", len(col))
    # print("Bits[dim_num] is: ",bits[dim_num])
    num_points_per_region = len(col)//math.pow(2, bits[dim_num])
   # print("Number of points per region ", num_points_per_region)
    count = 0
    for i in range(1, len(col)-1):
        count+=1
        if count == num_points_per_region:
            partition_points.append(int((col[i] + col[i+1])/2.0))
            count = 0
    partition_points.append(int(col[len(col)-1] + 1.0))
    return partition_points


def bin(n, length):
    i = 1 << length
    ans = ""
    while (i > 0):
        if ((n & i) != 0):
            ans = ans + "1"
            #print("1", end="")
        else:
            ans = ans + "0"
            #print("0", end="")

        i = i // 2
    return ans

def compute_region(vector, partition_points):
    regions = []

    # buckets_list = []
    for j in range(0, len(vector)):
        flag = 0
        region = 0

        for i in range(0, len(partition_points[j])-1):
            total_buckets.append(partition_points[j][i])
            #buckets_list.append(partition_points[j][i])
            if(partition_points[j][i] <= int(vector[j]) and int(vector[j]) < partition_points[j][i+1]):
                regions.append(i)
                flag = 1
                break;
        #Handle edge case

        if flag == 0:
            if vector[j] < partition_points[j][0]:
                total_buckets.append(partition_points[j][i])
                # buckets_list.append(partition_points[j][0])
                regions.append(partition_points[j][0])
            elif vector[j] >= partition_points[j][len(partition_points[j])-1]:
                total_buckets.append(partition_points[j][i])
                # buckets_list.append(partition_points[j][len(partition_points[j])-1])
                regions.append(partition_points[j][len(partition_points[j])-1])


    return regions

def compute_bit_string(vector, partition_points, bits):
    final_rep = ""
    regions = []
    for j in range(0, len(vector)):
        flag = 0

        for i in range(0, len(partition_points[j])-1):

            if partition_points[j][i] <= vector[j] and vector[j] < partition_points[j][i + 1]:
                regions.append(i)
                bit_string = bin(partition_points[j][i], bits[j])
                final_rep = final_rep + bit_string
                flag = 1
                break
        # Handle edge case
        if flag == 0:

            if vector[j] < partition_points[j][0]:
                regions.append(partition_points[j][0])
                bit_string = bin(partition_points[j][0], bits[j])
                final_rep = final_rep + bit_string
            elif vector[j] >= partition_points[j][len(partition_points[j]) - 1]:
                regions.append(partition_points[j][len(partition_points[j]) - 1])
                bit_string = bin(partition_points[j][len(partition_points[j]) - 1], bits[j])
                final_rep = final_rep + bit_string

    return final_rep, regions


def compute_lower_bound(vector, vq, partition_points):
    li = 0
    rq = compute_region(vq, partition_points)
    ri = compute_region(vector, partition_points)

    for j in range(0, len(rq)):
        if ri[j] < rq[j]:
            lij = vq[j] - partition_points[j][ri[j]+1]
        elif ri[j] == rq[j]:
            lij = 0
        else:
            lij = partition_points[j][ri[j]] - vq[j]
        li += lij*lij

    return math.sqrt(li)

def compute_euclidean(v1, v2):
    dist = 0
    for i in range(0, len(v1)):
        dist += (v1[i]-v2[i])*(v1[i]-v2[i])
    return dist

def compute_manhattan(v1, v2):
    dist = 0
    for i in range(0, len(v1)):
        dist += abs(v1[i]-v2[i])
    return dist

def Candidate(d, i, dst, ans, t):
    if d < dst[t-1]:
        dst[t-1] = d
        ans[t-1] = i

    #SortOnDst(ans, dst, t);
    ans = [dis for _, dis in sorted(zip(dst, ans))]
    dst.sort()
    return ans, dst

def simple_search_algorithm(feature_vectors, vq, t, partition_points):
    ans = []
    dst = []
    images_considered = 0
    for k in range(0, t):
        dst.append(float('inf'))
        ans.append(0)

    # total_buckets = []
    d = float('inf')
    idx = 0
    for i in range(0, len(feature_vectors)):
        li = compute_lower_bound(feature_vectors[i], vq, partition_points)
        # total_buckets += buckets
        # for b in buckets:
        #     total_buckets.append(b)


        if li < d or idx < t:
            images_considered += 1
            d = compute_euclidean(vq, feature_vectors[i])
            idx+=1
            ans, dst = Candidate(d, i, dst, ans, t)

    return ans, dst, len(set(total_buckets)), images_considered

def calculate_distances(original, test):
    result = {}
    i = 0
    for img in original:
        result[i] = compute_euclidean(img, test)
        i+=1

    actual_result = sorted(result.items(), key=lambda x: x[1])

    return actual_result


def perform_va_files(folder_path, feature, q_image_name, t, b):
    print("Calculating features vectors of the dataset")
    image_data_attributes = get_images_and_attributes_from_folder(folder_path)
    images = [img['image'] for img in image_data_attributes]
    image_names = [img['filename'] for img in image_data_attributes]
    image_features = get_flattened_features_for_images(images, feature)

    # q_image = get_image_arr_from_file(q_image_name)
    # query_image_features = get_flattened_features_for_images([q_image], feature)

    d = len(image_features[0])
    print("Computing bits per dimension: ")
    bits = compute_number_of_bits(b, d)
    overall_partition_list = []
    """Calculate the partition points for each dimension. These points are added to the overall_partition_points list
    which will contains the partition points for all dimensions."""

    for i in range(d):
        partition_points = distribute_partition_points(image_features, i, bits)
        overall_partition_list.append(partition_points)

    """Calculate approximation vectors for each vector in the Database"""
    approximations = []
    print("Calculating approximations: ")
    for i in range(0, len(image_features)):
        bit_string, regions = compute_bit_string(image_features[i], overall_partition_list, bits)
        approximations.append(bit_string)
    approx_set = set(approximations)

    """Read the query image and compute its feature vecyor"""
    # test_dataset = get_images_and_attributes_from_folder(q_image_name)
    query_image = get_image_object_from_file(q_image_name)
    query_image_features = get_flattened_features_for_images([query_image.image_arr], feature)
    query_image_features = query_image_features[0]
    query_image.features = query_image_features

    query_vector = query_image_features

    print("Performing Simple Search Algorithm: ")
    ans, dst, total_buckets, images_considered = simple_search_algorithm(image_features, query_vector, t, overall_partition_list)
    expected_files = []
    actual_result = calculate_distances(image_features, query_vector)

    idx = 0
    print("Expected outputs are: ")
    for x in actual_result:
        expected_files.append(image_names[x[0]])
        print(image_names[x[0]])
        idx += 1
        if idx == t:
            break
    print("Output vectors are saved to the Output_VAFiles folder: ")
    output = []
    output_features = []
    output.append((query_image.image_arr, "input-" + query_image.filename))

    false_positives = 0
    misses = 0
    correct_images = 0
    for a in ans:
        if image_data_attributes[a]['filename'] not in expected_files:
            false_positives += 1
        else:
            correct_images += 1
        output.append((images[a], image_names[a]))
        output_features.append((images[a], image_names[a], image_features[a]))
    save_images_by_clearing_folder(output, "Output_VAFiles")

    print("Number of unique approximations considered are: ", len(approx_set))
    print("Number of overall approximations considered are ", len(approximations))
    print("Number of bytes required for index structure is: ", len(approximations) * len(approximations[0]) / 8)
    print("Number of buckets searched are: ", total_buckets)
    print("The number of false positives are: ", false_positives)
    print("The miss rate is: ", misses / len(expected_files))
    return output_features


# image_data = get_images_and_attributes_from_folder("4000")
# images = []
# file_names = []
# for img_dict in image_data:
#     images.append(img_dict['image'])
#     file_names.append(img_dict['filename'])
# feature_vetors = []
# print("Calculating features vectors of the dataset")
# for im in images:
#     input_vector = compute_color_moment_of_image(im)
#     feature_vetors.append(input_vector)





# d = len(feature_vetors[0])

# b = 800
# print("Computing bits per dimension: ")
# bits = compute_number_of_bits(b, d)

# overall_partition_list = []
# """Calculate the partition points for each dimension. These points are added to the overall_partition_points list
# which will contains the partition points for all dimensions."""

# for i in range(0, d):
#     partition_points = distribute_partition_points(feature_vetors, i, bits)
#     overall_partition_list.append(partition_points)
#
# """Calculate approximation vectors for each vector in the Database"""
# approximations = []
# print("Calculating approximations: ")
# for i in range(0, len(feature_vetors)):
#     bit_string, regions = compute_bit_string(feature_vetors[i], overall_partition_list, bits)
#     approximations.append(bit_string)
# approx_set = set(approximations)

# """Read the query image and compute its feature vecyor"""
# test_dataset = get_images_and_attributes_from_folder("TestVA")
# actual = []
# test_type = test_dataset[0]['type']
# test_subject_id = test_dataset[0]['subject_id']
# test_image_id = test_dataset[0]['image_id']
# test_file_name = test_dataset[0]['filename']
# actual.append(test_type)
# actual.append(test_subject_id)

# query_vector = compute_color_moment_of_image(test_dataset[0]['image'])
# t = 10
# print("Performing Simple Search Algorithm: ")


# ans, dst, total_buckets, images_considered = simple_search_algorithm(feature_vetors, query_vector, t, overall_partition_list)
#
# expected_files = []
# actual_result = calculate_distances(feature_vetors, query_vector)

# idx = 0
# print("Expected outputs are: ")
# for x in actual_result:
#     expected_files.append(file_names[x[0]])
#     print(file_names[x[0]])
#     idx+=1
#     if idx == t:
#         break
# print("Output images are saved to the Output_VAFiles folder. ")
# output = []
# output.append((test_dataset[0]['image'], "input-"+test_file_name))
# false_positives = 0
# misses = 0
# correct_images = 0
# for a in ans:
#     if image_data[a]['filename'] not in expected_files:
#         false_positives += 1
#     else:
#         correct_images += 1
#     output.append((images[a], file_names[a]))
# save_images_by_clearing_folder(output, "Output_VAFiles")
#
#
# print("Number of unique approximations considered are: ", len(approx_set))
# print("Number of overall approximations considered are ", len(approximations))
# print("Number of bytes required for index structure is: ", len(approximations)*len(approximations[0])/8)
# print("Number of buckets searched are: ", total_buckets)
# print("The number of false positives are: ",(images_considered-correct_images)/images_considered)
# print("The miss rate is: ",(t-correct_images)/t)
