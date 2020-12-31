#kNN
#%matplotlib inline
import numpy as np
import statistics
import os, subprocess, time


def post_prob_knn(input_list, class_num):
    # make sure input list is a single column of the testing data classifications
    # in these cases class num is either 0 or 1
    
    divisor = len(input_list)
    
    i = 0
    count = 0
    while i < len(input_list):
        if input_list[i] == class_num:
            count += 1
        
        i += 1
    
    return((count / divisor))
    

# calculate the Euclidean distance between two vectors
# inputs have to be arrays!!!!
def test_euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2)): # change to row2
        distance += ((row1[i] - row2[i]) ** 2)
    return np.sqrt(distance)


# calculate the Mahalanobis distance between two vectors
# inputs have to be arrays!!!!
def test_mahalanobis_distance(row1, row2, Inv_cov_matrix):
    distance = 0.0
    for i in range(len(row2)):
        distance += ((row1[i] - row2[i]) ** 2) * Inv_cov_matrix
    return np.sqrt(distance)

# calculate the Manhatten distance between two vectors
# otherwise known as "city-block distance"
# inputs have to be arrays!!!!
def test_manhatten_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2)):
        distance += abs((row1[i]) - row2[i])
    return (distance)

# uses euclidean distance
def calc_dis_matrix(test_list, train_list):
    i = 0
    j = 0
    
    distance_matrix = np.zeros((len(test_list), len(train_list)))    
    
    while i < len(test_list):
        j = 0
        while j < len(train_list):
            distance_matrix[i][j] = test_euclidean_distance(test_list[i], train_list[j])
            
            j += 1
        i += 1
    
    return(distance_matrix)

# uses mahalanobis distance
def calc_dis_matrix_2(test_list, train_list, train_labels):
    i = 0
    j = 0
    
    distance_matrix = np.zeros((len(test_list), len(train_list)))
    
    # calculating the inverse covariance matrices
    # change the load to wherever you have the covariance matrices located
    cov_matrix_cl_0 = np.load('Cov_Matrices\class_0_cov_matrix.npy')
    cov_matrix_cl_0 = np.array([[cov_matrix_cl_0]]) # this is done so np.linalg.inv works
    inv_cov_mtx_cl_0 = np.linalg.inv(cov_matrix_cl_0)   
    
    # change the load to wherever you have the covariance matrices located
    cov_matrix_cl_1 = np.load('Cov_Matrices\class_1_cov_matrix.npy')
    cov_matrix_cl_1 = np.array([[cov_matrix_cl_1]]) # this is done so np.linalg.inv works
    inv_cov_mtx_cl_1 = np.linalg.inv(cov_matrix_cl_1) 
    
    
    while i < len(test_list):
        j = 0
        while j < len(train_list):
            if (train_labels[j] == 0):
                distance_matrix[i][j] = test_mahalanobis_distance(test_list[i], train_list[j], inv_cov_mtx_cl_0)
            if (train_labels[j] == 1):
                distance_matrix[i][j] = test_mahalanobis_distance(test_list[i], train_list[j], inv_cov_mtx_cl_1)
            
            j += 1
        i += 1
    
    return(distance_matrix)

# uses manhatten distance
def calc_dis_matrix_3(test_list, train_list):
    i = 0
    j = 0
    
    distance_matrix = np.zeros((len(test_list), len(train_list)))
    
    while i < len(test_list):
        j = 0
        while j < len(train_list):
            distance_matrix[i][j] = test_manhatten_distance(test_list[i], train_list[j])
            
            j = j + 1
        i = i + 1
    
    return(distance_matrix)



def copy_and_sort_row(input_list_row, row_index_num):
    output_row = input_list_row[row_index_num]
    
    return(np.sort(output_row))


# looks and identifies the index location of the minimum values in the input_list_row!
def search_for_min_value(input_list_row, row_index_num, k):
    
    # copy_sorted_row = np.array(copy_and_sort_row(input_list_row, row_index_num)).reshape(1, len(input_list_row))

    copy_sorted_row = copy_and_sort_row(input_list_row, row_index_num - 1)
    i = 0 # to iterate across the copy_and_sort_row
    
    index_num_output_list = []
    
    while i < k:
        # minimum value is the first value and second is the following and so on
        min_value = copy_sorted_row[i]
        # this looks to find the index of where the min_value is located on input_list_row
        j = 0

        while j < len(input_list_row[row_index_num - 1]): #len(index_list_row)
            if min_value == input_list_row[row_index_num - 1][j]:
                # added the index lis number (j) to the output
                index_num_output_list.append(j)
            #iterator for searching on the input_list_row
            j += 1
        
        # iterate
        i += 1
        
    return(index_num_output_list)
    
# returns the classification output list for after going through kNN at index_num_output_list location
    # index_num_output_list is generated from search_for_min_value
def kNN_output_list(classification_column_train_data, index_num_output_list, output_list):
    
    #output_list = [] # will contain the majority of class0 or class1
    
    # These are for counting if the training classification column
    count_Zero = 0
    count_One = 0
    
    i = 0 # iterator 
    while i < len(index_num_output_list):
        # adds to count zero if at the index number in the classification of the training data is a 0
        if classification_column_train_data[index_num_output_list[i]] == 0:
            count_Zero += 1
        # adds to count zero if at the index number in the classification of the training data is a 1
        if classification_column_train_data[index_num_output_list[i]] != 0.0:
            count_One += 1
            
        # iterate
        i += 1
    
    # Finding the prior probility of the training dataset for each class
    prior_prob_Zero = 0.2570935582822086 # = post_prob_knn(classification_column_train_data, 0)
    prior_prob_One = 0.7429064417177914 # = post_prob_knn(classification_column_train_data, 1)
    
    # Apply Prior Probability to the dataset (the votes)
    # the votes are what determines the output of the pdf = probability density function
    # the probability density function outputs the ratio or likelihood of a given datapoint to belong to
    # any given class.
    # I need to multiply the prior prob to get a better estimation of what the outcome would be
    count_Zero = count_Zero * prior_prob_Zero
    count_One = count_One * prior_prob_One
    
    
    
    # Majority of count_Zero vs count_One takes all
    if count_Zero >= count_One:
        output_list.append(0)
    if count_One > count_Zero:
        output_list.append(1)
    
    # this will return a signle 0 or 1 classification for every row that is done!
    # so, if this function is iterated then you can create a full output_list
    return(output_list)



def main():
    
    # Load in the training lables and training datset
    train_dataset = np.load(r'xray_dataset_samples\train_processed.npy')
    train_labels = np.load(r'xray_dataset_samples\train_labels.npy')
    
    # Changes the type so to prevent overflowing    
    train_dataset = train_dataset = train_dataset.astype('int32')
    
    # For actual calculation
    test_dataset = np.load(r'xray_dataset_samples\test_processed.npy')
    test_labels = np.load(r'xray_dataset_samples\test_labels.npy')
    
    # Load in the testing labels and testing dataset    
    test_dataset = np.load(r'xray_dataset_samples\Old\sample_test_n=30.npy')
    test_labels = np.load(r'xray_dataset_samples\Old\labels_test_n=30.npy')
    
    # Changes the type so to prevent overflowing
    test_dataset = test_dataset = test_dataset.astype('int32')

    
    
    # marks when to start the clock for kNN
    t0 = time.time()
    
    # change this to any distance metric
        # Euclidean
        # Mahalanobis
        # Manhatten (City-Block)
    euclidean_distance_matrix_output = calc_dis_matrix(test_dataset, train_dataset)    
    
    # z specifies the starting value
    z = 1
    # train_kNN_param will the output list for the resultant classification labels after kNN predictions
    train_kNN_param = []
    while z < len(euclidean_distance_matrix_output):
        
        # the 5 represents the k-NN so choose whatever number you want for the number of nearest neighbors
        search_for_min = search_for_min_value(euclidean_distance_matrix_output, z, 5)
        
        # createst the output list of classification labels post kNN predictions
        train_kNN_output_list = kNN_output_list(train_labels, search_for_min, train_kNN_param)
        
        # iterate
        z += 1
    
    # converting the list into an array
    output_array = np.array(train_kNN_output_list)
    
    t1 = time.time()
    
    # saving the kNN_output_labels as .npy file
    # Change this to whatever you would like to name the file
    np.save("kNN_output_labels.npy", output_array)

    # displays the total time that it takes to run kNN    
    total_time = t1 - t0
    print("total time = ", total_time)    
    
if __name__ == "__main__":
    main()
