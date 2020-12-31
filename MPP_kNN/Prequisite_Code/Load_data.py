# Load function
import numpy as np
import os

# this function assumes that the .npy files are located in the xray_dataset directory
# change the name of xray_dataset if you have stored the .npy files in a different directory
def load_data(directory = 'xray_dataset'):
    file_list = []
    
    files = os.listdir(directory)
    # goes through the directory and loads them into the file_list
    for f in files:
        file_list.append(f)
    
    # the first file in the directory is labels
    # thre rest of the files that follow are the sample files
    
    # the file_list contains the links to all the npy files that are in the xray_dataset directory
    return (file_list)


# This function requires that the input be the list created from the load data function
# It will output the data that comes from the links into a np array
def extract_data(file_list, directory = 'xray_dataset'):
    # This data_list will take in the extracted npy information
    data_list = []
    

    
    # i = 0 is the first file which is labels.npy
    # change i = 1 if you want to only load the samples datasets
    i = 0
    # loading the extracted data in the npy files into data_list
    while i < len(file_list):
        # directory will still be 'xray_dataset' unless you change it
        data_list.append(np.load(directory + "\\" + file_list[i]))
        
        i = i + 1
    
    # dataset is the data_list converted into an np array (this will be the returned output)
    dataset = np.array(data_list)
    
    return (dataset)
    

def main():
    file_list = load_data(directory = 'xray_dataset')
    dataset = extract_data(file_list, directory = 'xray_dataset')
    

if __name__ == "__main__":
    main()