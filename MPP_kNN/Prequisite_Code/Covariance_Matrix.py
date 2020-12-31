# Covariance Matrix
#%matplotlib inline
import numpy as np
import statistics
import os, subprocess, time


# specify class num to be either (0 or 1) which corresponds to the classification labels
# turns a two dimensional covariance matrix into 1 dimension
def cov_matrix(input_data, label_data, class_num):
    # this is used to parse out the unwanted data points
    # and include only the data points that correspond to input class num (0 or 1)
    intermediate_list = []
    
    # iterator for the labels list array
    m = 0 
    
    # loops through each data point in the training data set
    while m < len(input_data):
        # checks to see if the a specific image is labeled the same as what you want
        if (label_data[m] == class_num):
            
            # creates a list of only the training points with the specified class num label 
            intermediate_list.append(input_data[m])
            
        # iterate
        m = m + 1
    
    # creating the array from the intermeditate list and then flattening to 1-D
    intermediate_array = np.array(intermediate_list)
    intermediate_array = intermediate_array.flatten()

    # creating the covariance matrix
    cov_matrix_output = np.cov(intermediate_array) # may also use np.flatten()

    # saving the covariance matrix that was computed as to save computation time
    if (class_num == 0):
        np.save("class_0_cov_matrix.npy", cov_matrix_output)
    if (class_num == 1):
        np.save("class_1_cov_matrix.npy", cov_matrix_output)
    elif (class_num > 1 or class_num < 0):
        print("The number you entered is not correct as dialed \n Please hang up and try again :)")
        
 
    return (cov_matrix_output)