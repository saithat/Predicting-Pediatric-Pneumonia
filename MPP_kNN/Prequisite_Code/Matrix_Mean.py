# Matrix mean (average)
#%matplotlib inline
import numpy as np
import statistics
import os, subprocess, time

# calculates the matrix mean for every pixel and returns it in an np array
# class num = what class average you are calculating for the training set (0 or 1)
def matrix_mean(input_data, label_data, class_num):

    # create the output np array
    output_avg_array = np.zeros(input_data[0].shape)
    
    # iterator for the labels list array
    m = 0 
    
    # denominator for the average, counts the number of instances for a specified classification label
    count = 0
    
    # loops through each data point in the training data set
    while m < len(input_data):
        # checks to see if the a specific image is labeled the same as what you want
        if (label_data[m] == class_num):
            # for each match of the classifier label, add 1 to the count
            count = count + 1

            # gets the sum of all datapoints from the specified class 
            output_avg_array = output_avg_array + input_data[m]
            
        # iterate
        m = m + 1
    
    
    # checks to see if there are labels for the specified class num
    if (count == 0):
        print("count is = zero. Something went wrong")
        return (0)
    
    # now divide the output by the count
    output_avg_array = output_avg_array / count
    
    # saving the average that was computed as to save computation time
    if (class_num == 0):
        np.save("class_0_avg.npy", output_avg_array)
    if (class_num == 1):
        np.save("class_1_avg.npy", output_avg_array)
    elif (class_num > 1 or class_num < 0):
        print("The number you entered is not correct as dialed \n Please hang up and try again :)")
    
    # return the average
    return (output_avg_array)