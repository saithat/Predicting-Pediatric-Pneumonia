import os, subprocess, time, random
import tensorflow
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from sklearn import model_selection as ms
from skimage.measure import block_reduce

'''
Stats for all images:

x stats: ---------------
mean x   : 970.6890368852459
std x    : 383.35938110959034
max x    : 2713
min x    : 127

y stats: ---------------
mean y   : 1327.880806010929
std y    : 363.46988369451225
max y    : 2916
min y    : 384

Number of samples for each category:
Training NORMAL      = 1341
Training PNEUMONIA   = 3875
Training TOTAL       = 5216

Divisors of 5216 (for batch processing):
2, 4, 8, 16, 32, 163, 326, 652, 1304, 2608

Testing NORMAL       = 234
Testing PNEUMONIA    = 390
Testing TOTAL        = 624

Validation NORMAL    = 8
Validation PNEUMONIA = 8
Validation TOTAL     = 16

Image sizes after pooling = (250, 175)
'''

# Iterate through all directories, process images
def generate_dataset_stats(read_path = "chest_xray"):
    '''
    Generates the statistics for the images in the dataset
    Arguments:
    ----------
    read_path: string
        - Absolute path (or relative path wrt to the current directory) to the dataset

    No return value
    '''

    train_dir = read_path + "/train"
    test_dir = read_path + "/test"
    val_dir = read_path + "/val"

    dir_list = [train_dir, test_dir, val_dir]

    total_count = 0

    sum_sizes = np.zeros((1, 2))
    x_sizes = []
    y_sizes = []

    #num_class_per_dir = {d:[0, 0] for d in dir_list}
    num_class_per_dir = [[0, 0], [0, 0], [0, 0]]

    for d in range(0, len(dir_list)):
        sub_d = dir_list[d] + "/NORMAL"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f
            read_f = tensorflow.io.read_file(img_path)
            decoded_img = tensorflow.image.decode_jpeg(read_f, channels = 1)
            # channels = 1 forces pixels to be grayscale
            x_sizes.append(decoded_img.shape[0])
            y_sizes.append(decoded_img.shape[1])
            total_count += 1

            num_class_per_dir[d][0] += 1

        # Go through pnemonia images
        sub_d = dir_list[d] + "/PNEUMONIA"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f
            read_f = tensorflow.io.read_file(img_path)
            decoded_img = tensorflow.image.decode_jpeg(read_f, channels = 1)
            # channels = 1 forces pixels to be grayscale (should be grayscale already)
            x_sizes.append(decoded_img.shape[0])
            y_sizes.append(decoded_img.shape[1])

            num_class_per_dir[d][1] += 1

    # Stats for x:
    print("x stats: ---------------")
    print("mean x   :", np.mean(x_sizes))
    print("std x    :", np.std(x_sizes))
    print("max x    :", max(x_sizes))
    print("min x    :", min(x_sizes))
    print("\n")

    # Stats for y:
    print("y stats: ---------------")
    print("mean y   :", np.mean(y_sizes))
    print("std y    :", np.std(y_sizes))
    print("max y    :", max(y_sizes))
    print("min y    :", min(y_sizes))
    print("\n")

    # Stats on distributions of samples
    print("Training NORMAL      =", num_class_per_dir[0][0])
    print("Training PNEUMONIA   =", num_class_per_dir[0][1])
    print("Training TOTAL       =", num_class_per_dir[0][1] + num_class_per_dir[0][0])
    print("\n")

    print("Testing NORMAL       =", num_class_per_dir[1][0])
    print("Testing PNEUMONIA    =", num_class_per_dir[1][1])
    print("Testing TOTAL        =", num_class_per_dir[1][1] + num_class_per_dir[1][0])
    print("\n")

    print("Validation NORMAL    =", num_class_per_dir[2][0])
    print("Validation PNEUMONIA =", num_class_per_dir[2][1])
    print("Validation TOTAL     =", num_class_per_dir[2][1] + num_class_per_dir[2][0])
    print("\n")

    print(np.array(num_class_per_dir[:][0]).flatten())

    normals = [num_class_per_dir[0][0], num_class_per_dir[1][0]]
    cancers = [num_class_per_dir[0][1], num_class_per_dir[1][1]]

    x_labels = ["Train", "Test"]
    plt.bar([0, 1], normals,label = 'Normal', bottom = cancers)
    plt.bar([0, 1], cancers, label = 'Cancer')
    plt.xticks([0, 1], x_labels)
    plt.ylabel("Frequency")
    plt.xlabel("Split")
    plt.legend()
    plt.title("Splits in Pneumonia Dataset")
    plt.show()

    labels = ["Normal", "Pneumonia"]
    by_label_train = [num_class_per_dir[0][0], num_class_per_dir[0][1]]
    by_label_test = [num_class_per_dir[1][0], num_class_per_dir[1][1]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie(by_label_train, labels = labels, autopct = '%1.1f%%')
    ax2.pie(by_label_test, labels = labels, autopct = '%1.1f%%')
    ax1.axis('equal')
    ax2.axis('equal')
    ax1.set_title("Training")
    ax2.set_title("Testing")
    fig.suptitle("Proportion of Classes in Dataset")

    plt.show()


# Iterate through all directories, process images
# previous: reshape_size = (970, 1320)
def generate_dataset(read_path = "chest_xray", write_path = "xray_dataset", reshape_size = (700, 1000),
                        normalize = True, save = False):
    '''
    Builds a cleaned dataset from the chest xray data downloaded from Kaggle

    Arguments:
    ----------
    read_path: string, optional
        - Default: "chest_xray"
        - Path from which to read the files
        - Must contain the "train", "test", and "val" directories from the original kaggle dataset
    write_path: string, optional
        - Default: "xray_dataset"
        - Path to write the .npy files to
    reshape_size: 1x2 tuple, optional
        - Default: (970, 1320) (~ the mean size of all images in dataset)
        - Size which you want to resize all of the images to
    normalize: bool, optional
        -Default: True
        - Whether or not to normalize the pixels
        - Normalization is performed for each image
    save: 

    Returns:
    --------
    data: np.array
        - One np.array of all of the image samples read from the initial repo
        - Each row is one sample
    labels: list
        - Label (0 or 1) for each of the rows in data
            - 0 = no pneumonia, 1 = pneumonia
    '''

    train_dir = read_path + "/train"
    test_dir = read_path + "/test"
    val_dir = read_path + "/val"

    dir_list = [train_dir, test_dir, val_dir]

    total_count = 0

    data = []

    labels = []

    for d in dir_list:
        sub_d = d + "/NORMAL"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize)

            if (normalize):
                # Normalize the data for each individual image:
                im_mean = decoded_img.mean()
                decoded_img = (decoded_img - im_mean) / 255

            # Save the np.array to write_path
            if (save):
                np.save(write_path + "/sample_{:05d}.npy".format(total_count), decoded_img)
            data.append(decoded_img.ravel())
            # Set label equal to 0 (i.e. no pneumonia)
            labels.append(0)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

        # Go through pnemonia images
        sub_d = d + "/PNEUMONIA"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize)

            if (normalize):
                # Normalize the data for each individual image:
                im_mean = decoded_img.mean()
                decoded_img = (decoded_img - im_mean) / 255

            if (save):
                # Save the np.array to write_path
                np.save(write_path + "/sample_{:05d}.npy".format(total_count), decoded_img)

            data.append(decoded_img.ravel())
            # Set label equal to 1 (i.e. pneumonia)
            labels.append(1)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

    # Write the labels as an np array
    if (save):
        np.save(write_path + "/labels.npy", np.array(labels))
    
    return (np.array(data), labels) # Return the data itself

def generate_dataset_org_splits(read_path = "chest_xray", write_path = "xray_dataset", reshape_size = (700, 1000),
                        normalize = True, save = False):
    '''
    Builds a cleaned dataset from the chest xray data downloaded from Kaggle

    Arguments:
    ----------
    read_path: string, optional
        - Default: "chest_xray"
        - Path from which to read the files
        - Must contain the "train", "test", and "val" directories from the original kaggle dataset
    write_path: string, optional
        - Default: "xray_dataset"
        - Path to write the .npy files to
    reshape_size: 1x2 tuple, optional
        - Default: (970, 1320) (~ the mean size of all images in dataset)
        - Size which you want to resize all of the images to
    normalize: bool, optional
        -Default: True
        - Whether or not to normalize the pixels
        - Normalization is performed for each image
    save: bool, optional
        - Default: True
        - If true, saves the data to write_path

    Returns:
    --------
    data: np.array
        - One np.array of all of the image samples read from the initial repo
        - Each row is one sample
    labels: list
        - Label (0 or 1) for each of the rows in data
            - 0 = no pneumonia, 1 = pneumonia
    '''

    train_dir = read_path + "/train"
    test_dir = read_path + "/test"
    val_dir = read_path + "/val"

    total_count = 0

    dir_list = [train_dir, test_dir, val_dir]

    for d in range(0, len(dir_list)):

        data = []
        labels = []

        sub_d = dir_list[d] + "/NORMAL"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize)

            if (normalize):
                # Normalize the data for each individual image:
                im_mean = decoded_img.mean()
                decoded_img = (decoded_img - im_mean) / 255

            data.append(decoded_img.ravel())
            # Set label equal to 0 (i.e. no pneumonia)
            labels.append(0)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

        # Go through pnemonia images
        sub_d = dir_list[d] + "/PNEUMONIA"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize)

            if (normalize):
                # Normalize the data for each individual image:
                im_mean = decoded_img.mean()
                decoded_img = (decoded_img - im_mean) / decoded_img.var()

            data.append(decoded_img.ravel())
            # Set label equal to 1 (i.e. pneumonia)
            labels.append(1)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

        # Write this split as one dataframe to it's given file
        if (d == 0):
            # Save as train
            np.save(write_path + "/train_processed.npy", data)
            np.save(write_path + "/train_labels.npy", labels)
    
        if (d == 1):
            #Save as test
            np.save(write_path + "/test_processed.npy", data)
            np.save(write_path + "/test_labels.npy", labels)

        if (d == 2):
            # Save as val
            np.save(write_path + "/val_processed.npy", data)
            np.save(write_path + "/val_labels.npy", labels)
    
    return (np.array(data), labels) # Return the data itself
    

def train_test_split_from_dir(split_proportions = (0.8, 0.2), val = False, dataset_dir_path = "xray_dataset"):
    '''
    Generates a train/test or train/val/test
    Arguments:
    ----------
    split_proportions: tuple
        - Proportion of data to split
        - Doesn't have to sum to zero, just needs to be proportion of data you desire
        - Length is either 2 or 3
            - If val, length should be 3: (<train proportion>, <val proportion>, <test proportion>)
            - If not val, length should be 2 (<train proportion>, <test proportion>)
    val: bool, optional
        - Default: False
        - If True, will generate a validation set
    dataset_dir_path: string, optional
        - Default: "xray_dataset"
        - Path to find the numpy-generated dataset in

    Returns:
    --------
    data: tuple of zips
    if val:
        Returns (train, val, test)
        - Train, validation, and testing data, respectively
    if not val:
        Returns (train, test)
        - Training and testing data, respectively
    '''
    
    num_samples = 5856 # Number of samples that we have

    #Normalize the split_proportions tuple
    tot = sum(split_proportions)
    split_proportions = [i / tot for i in split_proportions]
    
    # Generates a train/val/test split
    x_samples = []

    labels = np.load(dataset_dir_path + '/labels.npy')

    # Build a dataframe with all of the data, use train_test_split
    for i in range(0, num_samples):
        print(i)
        load_s = np.load(dataset_dir_path + "/sample_{:05d}.npy".format(i)).ravel()
        x_samples.append(load_s)

    x_samples = np.array(x_samples)

    # Outputs a zip
    if val:
        te_size = split_proportions[2]
        Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(x_samples, labels, test_size = te_size)

        # Draws the validation set from the train set
        Xtrain, Xval, Ytrain, Yval = ms.train_test_split(Xtrain, Ytrain, test_size = split_proportions[1])

        return (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)
    else:
        # If no validation set specified, just split train, test
        te_size = split_proportions[1]
        Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(x_samples, labels, test_size = te_size)

        return (Xtrain, Xtest, Ytrain, Ytest)

def train_test_split_data(img_data, labels, split_proportions = (0.8, 0.2), val = False):
    '''
    Generates a train/test or train/val/test directly from image data (np.array)
    Arguments:
    ----------
    img_data: np.array
        - Rows should be image data, flattened
        - The output from generate_dataset will work for this
    labels: list or np.array
        - Corresponds to labels for each of the samples in the img_data
        - Must be flattened (i.e. 1 x n for any n value)
    split_proportions: tuple
        - Proportion of data to split
        - Doesn't have to sum to zero, just needs to be proportion of data you desire
        - Length is either 2 or 3
            - If val, length should be 3: (<train proportion>, <val proportion>, <test proportion>)
            - If not val, length should be 2 (<train proportion>, <test proportion>)
    val: bool, optional
        - Default: False
        - If True, will generate a validation set

    Returns:
    --------
    data: tuple of X and Y data
        - If val, the return is (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)
        - If not val, the return is (Xtrain, Xtest, Ytrain, Ytest)
        - The "X..." variables are the actual datasets
        - The "Y..." variables are the labels

    '''
    #Normalize the split_proportions tuple
    tot = sum(split_proportions)
    split_proportions = [i / tot for i in split_proportions]

    if val:
        te_size = split_proportions[2]
        Xtrain, Ytrain, Xtest, Ytest = ms.train_test_split(img_data, labels, test_size = te_size)

        # Draws the validation set from the train set
        Xtrain, Xval, Ytrain, Yval = ms.train_test_split(Xtrain, Ytrain, test_size = split_proportions[1])
        return (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)

    else:
        # If no validation set specified, just split train, test
        te_size = split_proportions[1]
        Xtrain, Xtest, Ytrain, Ytest = ms.train_test_split(img_data, labels, test_size = te_size)

        return (Xtrain, Xtest, Ytrain, Ytest)

def sample_splits(read_path = "chest_xray", sizes = (100, 10, 30), reshape_size = (700, 1000), normalize = True):
    '''
    Builds samples of train, val, and test splits
        - Writes to current directory

    Arguments:
    ----------
    read_path: string
        - Path to find the dataset
    sizes: tuple of length 3
        - Size of each split to sample
        - Form: (train, val, test)
    reshape_size: tuple of length 2
        - Size to reshape each image to
    normalize: bool, optional
        - Default: True
        - Whether or not to normalize the pixels
        - Normalization is performed for each image

    Returns:
    --------
    No return value
    '''
    train_data = [os.path.join(read_path + "/train/NORMAL", f) for f in os.listdir(read_path + "/train/NORMAL")]
    train_data += [os.path.join(read_path + "/train/PNEUMONIA", f) for f in os.listdir(read_path + "/train/PNEUMONIA")]

    val_data = [os.path.join(read_path + "/val/NORMAL", f) for f in os.listdir(read_path + "/val/NORMAL")]
    val_data += [os.path.join(read_path + "/val/PNEUMONIA", f) for f in os.listdir(read_path + "/val/PNEUMONIA")]

    test_data = [os.path.join(read_path + "/test/NORMAL", f) for f in os.listdir(read_path + "/test/NORMAL")]
    test_data += [os.path.join(read_path + "/test/PNEUMONIA", f) for f in os.listdir(read_path + "/test/PNEUMONIA")]

    # Sample from each directory
    train_samples = random.sample(train_data, sizes[0])
    val_samples = random.sample(val_data, sizes[1])
    test_samples = random.sample(test_data, sizes[2])

    total_count = 0

    dsets = [train_samples, val_samples, test_samples]

    for i in range(0, 3): 

        dataset = dsets[i]

        data = []
        labels = []

        for img_path in dataset:

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize)

            if (normalize):
                # Normalize the data for each individual image:
                im_mean = decoded_img.mean()
                decoded_img = (decoded_img - im_mean) / decoded_img.var()

            data.append(decoded_img.ravel())
            if ("PNEUMONIA" in img_path):
                labels.append(1)
            else:
                labels.append(0)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "of", sum(sizes), "samples read")

        if (i == 0):
            categ = "train"
        elif (i == 1):
            categ = "val"
        elif (i == 2):
            categ = "test"

        # Save the samples
        np.save("sample_{category}_n={num}.npy".format(category = categ, num = sizes[i]), np.array(data))
        np.save("labels_{category}_n={num}.npy".format(category = categ, num = sizes[i]), np.array(labels))


def generate_dataset_w_pooling(read_path = "chest_xray", write_path = "xray_dataset", reshape_size = (700, 1000),
                        pool_fn = "max", save = False):
    '''
    Builds a cleaned dataset from the chest xray data downloaded from Kaggle

    Arguments:
    ----------
    read_path: string, optional
        - Default: "chest_xray"
        - Path from which to read the files
        - Must contain the "train", "test", and "val" directories from the original kaggle dataset
    write_path: string, optional
        - Default: "xray_dataset"
        - Path to write the .npy files to
    reshape_size: 1x2 tuple, optional
        - Default: (970, 1320) (~ the mean size of all images in dataset)
        - Size which you want to resize all of the images to
    normalize: bool, optional
        -Default: True
        - Whether or not to normalize the pixels
        - Normalization is performed for each image
    save: 

    Returns:
    --------
    data: np.array
        - One np.array of all of the image samples read from the initial repo
        - Each row is one sample
    labels: list
        - Label (0 or 1) for each of the rows in data
            - 0 = no pneumonia, 1 = pneumonia
    '''
    if (pool_fn == "mean"):
        pool_func = np.mean
    else:
        pool_func = np.max

    train_dir = read_path + "/train"
    test_dir = read_path + "/test"
    val_dir = read_path + "/val"

    total_count = 0

    dir_list = [train_dir, test_dir, val_dir]

    for d in range(0, len(dir_list)):

        data = []
        labels = []

        sub_d = dir_list[d] + "/NORMAL"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize).astype(np.ubyte)

            # Pool twice:
            decoded_img = block_reduce(decoded_img, (2, 2), pool_func)
            decoded_img = block_reduce(decoded_img, (2, 2), pool_func)
            decoded_img = decoded_img.astype(np.ubyte)

            data.append(decoded_img.ravel())
            # Set label equal to 0 (i.e. no pneumonia)
            labels.append(0)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

        # Go through pnemonia images
        sub_d = dir_list[d] + "/PNEUMONIA"
        for f in os.listdir(sub_d): # Iterates over the directory
            img_path = sub_d + "/" + f

            im = Image.open(img_path)
            im = ImageOps.grayscale(im)
            imResize = im.resize(reshape_size, Image.ANTIALIAS)
            decoded_img = np.array(imResize).astype(np.ubyte)
            
            #Pool twice
            decoded_img = block_reduce(decoded_img, (2, 2), pool_func)
            decoded_img = block_reduce(decoded_img, (2, 2), pool_func)
            decoded_img = decoded_img.astype(np.ubyte)

            data.append(decoded_img.ravel())
            # Set label equal to 1 (i.e. pneumonia)
            labels.append(1)

            total_count += 1
            if (total_count % 10 == 0):
                print(total_count, "samples read")

        # Write this split as one dataframe to it's given file
        if (d == 0):
            # Save as train
            np.save(write_path + "/train_processed.npy", data)
            np.save(write_path + "/train_labels.npy", labels)
    
        if (d == 1):
            #Save as test
            np.save(write_path + "/test_processed.npy", data)
            np.save(write_path + "/test_labels.npy", labels)

        if (d == 2):
            # Save as val
            np.save(write_path + "/val_processed.npy", data)
            np.save(write_path + "/val_labels.npy", labels)
    
    return (np.array(data), labels) # Return the data itself

if __name__ == "__main__":
    # Generates dataset stats
    generate_dataset_stats()

    # Generates the dataset with pooling
    generate_dataset_w_pooling(read_path = "chest_xray", write_path = "xray_dataset_pooled", reshape_size = (700, 1000),
                        pool_fn = "max", save = True)
    