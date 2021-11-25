import math
from input_output import get_images_and_attributes_from_folder
from Feature_models.ColorMoments import compute_color_moment_of_image
from input_output import save_images_by_clearing_folder
def compute_number_of_bits(b, d):
    # v = number of vectors
    # b = number of bits
    # d = number of dimensions
    bits = []

    for j in range(1, d+1):
        if j <= b%d:
            bj = math.floor(b/d) + 1;
        else:
            bj = math.floor(b/d)+0;
        bits.append(bj)

    return bits

def distribute_partition_points(feature_vector, dim_num, bits):
    #ex: for dim_num = 0 concatenate the first col values of all the data
    col = []
    for i in range(0, len(feature_vector)):
        col.append(feature_vector[i][dim_num])
    col.sort();

    partition_points = []
    partition_points.append(int(col[0]))
    print("Col size is: ", len(col))
    print("Bits[dim_num] is: ",bits[dim_num])
    num_points_per_region = len(col)//math.pow(2, bits[dim_num])
    print("Number of points per region ", num_points_per_region)
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
            ans = ans + "1";
            #print("1", end="")
        else:
            ans = ans + "0";
            #print("0", end="")

        i = i // 2
    return ans

def compute_region(vector, partition_points):
    regions = []
    print(len(vector))
    for j in range(0, len(vector)):
        flag = 0
        region = 0
        # print("Length of partition points is: ", len(partition_points[j]))
        for i in range(0, len(partition_points[j])-1):
            if(partition_points[j][i] <= int(vector[j]) and int(vector[j]) < partition_points[j][i+1]):
                regions.append(i)
                flag = 1
                break;
        #Handle edge case
        if(flag == 0):
            if(int(vector[j]) < partition_points[j][0]):
                regions.append(partition_points[j][0])
            elif(int(vector[j]) >= partition_points[j][len(partition_points)-1]):
                regions.append(partition_points[j][len(partition_points)-1])


    return regions

def compute_bit_string(vector, partition_points, bits):
    final_rep = ""
    regions = []
    for j in range(0, len(vector)):
        flag = 0

        for i in range(0, len(partition_points[j])-1):
            if (partition_points[j][i] <= vector[j] and vector[j] < partition_points[j][i + 1]):
                regions.append(i)
                bit_string = bin(partition_points[j][i], bits[j])
                final_rep = final_rep + bit_string
                flag = 1
                break;
        # Handle edge case
        if (flag == 0):
            if (vector[j] < partition_points[j][0]):
                regions.append(partition_points[j][0])
                bit_string = bin(partition_points[j][0], bits[j])
                final_rep = final_rep + bit_string
            elif (vector[j] >= partition_points[j][len(partition_points[j]) - 1]):
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
    print("Value of Li is: ",li)
    return math.sqrt(li)

def compute_euclidean(v1, v2):
    dist = 0
    for i in range(0, len(v1)):
        dist += (v1[i]-v2[i])*(v1[i]-v2[i])
    return dist

def Candidate(d, i, dst, ans, t):
    if d < dst[t-1]:
        dst[t-1] = d
        ans[t-1] = i

    #SortOnDst(ans, dst, t);
    ans = [dis for _, dis in sorted(zip(dst, ans))]
    dst.sort()
    return ans, dst

def simple_search_algorithm(feature_vectors, vq, t, approximations, partition_points):
    ans = []
    dst = []
    for k in range(0, t):
        dst.append(float('inf'))
        ans.append(0)


    d = float('inf')
    for i in range(0, len(feature_vectors)):
        li = compute_lower_bound(feature_vectors[i], vq, partition_points)
        print("Li value is: ",li)
        print("d value of is: ",d)
        if li < d:
            d = compute_euclidean(vq, feature_vectors[i])
            ans, dst = Candidate(d, i, dst, ans, t)
    print("Image ids: ")
    for a in ans:
        print(a)
    return ans, dst


#Test function
#read data from folder path. Output is a list of dictionaries
image_data = get_images_and_attributes_from_folder("Dataset")
images = []
file_names = []
for img_dict in image_data:
    images.append(img_dict['image'])
    file_names.append(img_dict['filename'])
feature_vetors = []
for im in images:
    hog_vector = compute_color_moment_of_image(im)
    feature_vetors.append(hog_vector)

"""Print the feature vectors"""
print("Printing the feature vectors: ")
print("Size of feature vectors is ", len(feature_vetors))
for f in feature_vetors:
    print(f)
    print("\n")



b = 80

d = len(feature_vetors[0])
bits = compute_number_of_bits(b, d)
print("Bits assigned to each dimension are as follows: ")
print("Size of bits vector is ", len(bits))
print(bits)
overall_partition_list = []
#Get partitionpartition_points[len(partition_points)-1] points for each dimension
#overall_partition_list contains patition points list for every dimension.
print("Printing partition points for each dimension")
for i in range(0, d):
    partition_points = distribute_partition_points(feature_vetors, i, bits)
    print(partition_points)
    overall_partition_list.append(partition_points)


#Get approximations for each vector/datapoint
approximations = []
print("Calculating approximations: ")
for i in range(0, len(feature_vetors)):
    bit_string, regions = compute_bit_string(feature_vetors[i], overall_partition_list, bits)
    approximations.append(bit_string)
approx_set = set(approximations)

#call simple search algo

query_vector = feature_vetors[1500]
t = 10
ans, dst = simple_search_algorithm(feature_vetors, query_vector, t, approximations, overall_partition_list)
print("Output vectors: ")
output = []
output.append((images[1500], "input-"+file_names[1500]))
for a in ans:
    print(a)
    output.append((images[a], file_names[a]))
save_images_by_clearing_folder(output, "Output_VAFiles")

