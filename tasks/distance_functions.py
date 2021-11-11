import numpy as np

# Calculation of manhattan distance
def manhattan(val1, val2):
    sum=0
    for i in range(len(val1)):
        sum += abs(val1[i] - val2[i])
    return np.sum(sum)