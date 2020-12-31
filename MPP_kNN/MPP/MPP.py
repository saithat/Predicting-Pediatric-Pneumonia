#MPP
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
    

# Compares two things and returns the greater classifier number after value comparison
def compare(class_zero_val, class_one_val):
    if class_zero_val > class_one_val:
        return(0)
    else:
        return(1)

# input data and labels are actually separate files
# Xtrain, Xval, Xtest, Ytrain, Yval, Ytest
# The X is the datasets, and the Y is the labes that correspond to the datasets
def MPP(input_data, label_data, case_num):
    # input_data = training data
    # label_data = training labels
     
    # left corresponds to NORMAL
    # right corresponds to PNEUMONIA
    
    # This prior was calculated in reference to the percentage of normal to pneumonia
    # in the training sample distribution
#    prior_prob_left = 0.257                   #     1341/(1341 + 3875)
    prior_prob_left = .27
    prior_prob_right = 1 - prior_prob_left
    
    
    if (case_num == 1):
        MPP_Case1(input_data, label_data, prior_prob_left, prior_prob_right)
    if (case_num == 2):
        MPP_Case2(input_data, label_data, prior_prob_left, prior_prob_right)
    if (case_num == 3):
        MPP_Case3(input_data, label_data, prior_prob_left, prior_prob_right)
    elif(case_num > 3 or case_num < 1):
        print("The number you entered is not correct as dialed \nPlease hang up and try again :)")
    
    
# Performs Optimization Function of a Gaussian (Case 1)
def MPP_Case1(input_data, label_data, prior_prob_left, prior_prob_right):
    
    # AVEREAGES
    mu_zero = np.load('averages\class_0_avg.npy')
    mu_one = np.load('averages\class_1_avg.npy')
    mu_zero_TRNP = mu_zero.transpose()
    mu_one_TRNP = mu_one.transpose()


    #Square of Sigma!
    sig_square = np.std(input_data) ** 2 #(0.24170213842980853) ** 2
    
    output_list = []
    
    i = 0
    # iterate through each data point through exhaustive search and compare
    # each the results of each "side" to assign a label
    while i < len(input_data):
        # class zero calculation
        leftside_P_one = ((mu_zero_TRNP.dot(input_data[i].transpose())) / (sig_square))
        leftside_P_two = ((mu_zero_TRNP.dot(mu_zero)) / (2 * (sig_square)))
        leftside = (leftside_P_one - leftside_P_two) + np.log(prior_prob_left)
        
        # class one calculation
        rightside_P_one = ((mu_one_TRNP.dot(input_data[i].transpose())) / (sig_square))
        rightside_P_two = ((mu_one_TRNP.dot(mu_one)) / (2 * (sig_square)))
        rightside = (rightside_P_one - rightside_P_two) + np.log(prior_prob_right)
        
        # stores the calculated classified value into a list
        output_list.append(compare(leftside, rightside))
    
        # iterate
        i = i + 1
    
    # converting the list into an array
    output_array = np.array(output_list)
    
    # saving the MPP_Case1_output_labels as .npy file
    np.save("MPP_Case1_output_labels.npy", output_array)
    
    return (output_list) 

# Performs Optimization Function of Gaussian (Case 2)
def MPP_Case2(input_data, label_data, prior_prob_left, prior_prob_right):

    # AVEREAGES
    mu_zero = np.load('averages\class_0_avg.npy')
    mu_one = np.load('averages\class_1_avg.npy')
    
    # COVARIANCE MATRIX
    covariance_matrix = np.load('Cov_Matrices\class_0_cov_matrix.npy')
    covariance_matrix = np.array([[covariance_matrix]]) # this is done so np.linalg.inv works
    output_list = []
    
    i = 0
    # iterate through each data point through exhaustive search and compare
    # each the results of each "side" to assign a label
    while i < len(input_data):
        # class zero calculation
        leftside_P_one = (input_data[i] - mu_zero.transpose())
        leftside_P_two = np.linalg.inv(covariance_matrix) # results in the inverse covariance matrix
        leftside_P_two = leftside_P_two[0][0]             # extracts just the number from the array
        leftside_P_three = input_data[i] - mu_zero
        leftside_step = (leftside_P_one.dot(leftside_P_two))
        leftside = ((leftside_step.dot(leftside_P_three))/ -2) + np.log(prior_prob_left)
        
        # class one calculation
        rightside_P_one = (input_data[i] - mu_one.transpose())
        rightside_P_two = np.linalg.inv(covariance_matrix) # results in the inverse covariance matrix
        rightside_P_two = rightside_P_two[0][0]            # extracts just the number from the array
        rightside_P_three = input_data[i] - mu_one
        rightside_step = (rightside_P_one.dot(rightside_P_two)) # multiplying matrices 2 x 1 * 2 x 2 = 
        rightside = ((rightside_step.dot(rightside_P_three))/ -2) + np.log(prior_prob_right)
        
        # stores the calculated classified value into a list
        output_list.append(compare(leftside, rightside))
        
        
        #iterate
        i = i + 1
    
    # converting the list into an array
    output_array = np.array(output_list)
    
    # saving the MPP_Case2_output_labels as .npy file
    np.save("MPP_Case2_output_labels.npy", output_array)
    
    return (output_array)

# Performs Optimization Function of Gaussian (Case 3)
def MPP_Case3(input_data, label_data, prior_prob_left, prior_prob_right):
    
    # AVEREAGES
    mu_zero = np.load('averages\class_0_avg.npy')
    mu_one = np.load('averages\class_1_avg.npy')
    
    # COVARIANCE MATRICES
    covariance_matrix_zero = np.load('Cov_Matrices\class_0_cov_matrix.npy')
    covariance_matrix_one = np.load('Cov_Matrices\class_1_cov_matrix.npy')
    covariance_matrix_zero = np.array([[covariance_matrix_zero]]) # this is done so np.linalg.inv works
    covariance_matrix_one = np.array([[covariance_matrix_one]])   # this is done so np.linalg.inv works
    
    output_list = []
    
    i = 0
    # iterate through each data point through exhaustive search and compare
    # each the results of each "side" to assign a label
    while i < len(input_data):
        # class zero calculation
        leftside_P_one = (input_data[i] - mu_zero).transpose()
        leftside_P_two = np.linalg.inv(covariance_matrix_zero) # results in the inverse covariance matrix
        leftside_P_two = leftside_P_two[0][0]                  # extracts just the number from the matrix
        leftside_P_three = input_data[i] - mu_zero
        leftside_step = (leftside_P_one.dot(leftside_P_two))
        leftside = ((leftside_step.dot(leftside_P_three))/ -2) # the negative 2 is in the formula for this equation!
        leftside = leftside + np.log(prior_prob_left) + ((np.log(np.linalg.det(covariance_matrix_zero))) / -2)
        
        # class one calculation
        rightside_P_one = (input_data[i] - mu_one).transpose()
        rightside_P_two = np.linalg.inv(covariance_matrix_one) # results in the inverse covariance matrix
        rightside_P_two = rightside_P_two[0][0]                # extracts just the number from the matrix
        rightside_P_three = input_data[i] - mu_one
        rightside_step = (rightside_P_one.dot(rightside_P_two)) # multiplying matrices 2 x 1 * 2 x 2 = 
        rightside = ((rightside_step.dot(rightside_P_three))/ -2) # the negative 2 is in the formula for this equation!
        rightside = rightside + np.log(prior_prob_right) + ((np.log(np.linalg.det(covariance_matrix_one))) / -2)
        
        # stores the calculated classified value into a list
        output_list.append(compare(leftside, rightside))
        
        
        #iterate
        i = i + 1
        
    # converting the list into an array
    output_array = np.array(output_list)
    
    # saving the MPP_Case3_output_labels as .npy file
    np.save("MPP_Case3_output_labels.npy", output_array)
    
    return (output_array)

# this function assumes that the .npy files are located in the xray_dataset directory
# change the name of xray_dataset if you have stored the .npy files in a different directory
def load_data(directory = 'xray_dataset'):
    file_list = []
    
    files = os.listdir(directory)
    # goes through the directory and loads them into the file_list
    for f in files:
        file_list.append(f)
    
    # the file_list contains the links to all the npy files that are in the xray_dataset directory
    return (file_list)
    
    

def main():
    
    # Loading data
    # change the directory location to wherever the necessary files are located
    train_dataset = np.load('xray_dataset_samples\sample_train_n=100.npy')
    train_labels = np.load('xray_dataset_samples\labels_train_n=100.npy')
    
    test_dataset = np.load('xray_dataset_samples\sample_test_n=30.npy')
    test_labels = np.load('xray_dataset_samples\labels_test_n=30.npy')
    
    
    # MPP
    # def MPP(input_data, label_data, case_num)
    Case_1 = MPP(test_dataset, test_labels, 1)    
    Case_2 = MPP(test_dataset, test_labels, 2)
    Case_3 = MPP(test_dataset, test_labels, 3)
    
    C1_output = np.load('MPP_Case1_output_labels.npy')
    C2_output = np.load('MPP_Case2_output_labels.npy')
    C3_output = np.load('MPP_Case3_output_labels.npy')
    
    print("C1: ", C1_output)
    print("C2: ", C2_output)
    print("C3: ", C3_output)
if __name__ == "__main__":
    main()
