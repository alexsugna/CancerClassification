"""
This file contains functions for exploring, reading, and writing aggregate data structures.

Alex Angus

October 16, 2019
"""
import os
import numpy as np

def explore_data(data_path):
    """
    Explores a given dataset for missing data and determines if the number 
    of features for each example is consistent, and if they are in the same order.
    
    params:
        data_path: the directory of the dataset
    
    returns:
        classes: a numpy array of cancer classes
        
    """
    classes = os.listdir(data_path) #list of cancer type example files
    bad_data_paths = []
    for class_file in classes: #iterate through cancer types
        example_path = data_path + class_file + '/' #redefine path
        i = 0
        for example_file in os.listdir(example_path): #iterate through examples
            if 'MANIFEST.txt' in example_file: #skip manifest
                continue
            example_file_path = example_path + example_file + '/' #redefine path
            for example_txt in os.listdir(example_file_path): #iterate through example features
                if(example_txt == 'annotations.txt'): #skip annotations
                    continue
                instance_path = example_file_path + example_txt
                try:
                    instance_features = np.loadtxt(instance_path, skiprows=1, usecols=0, dtype=np.string_)
                    instance_values = np.loadtxt(instance_path, skiprows=1, usecols=(1,2))
                    if i != 0:
                        if(np.shape(instance_features)[0] != prev_features_len): #print if number of features are different
                            print("Lengths are not the same")
                        if not np.array_equal(instance_features, instance_features_reference): #print if features are missordered
                            print("Features are missordered")
                    else: #else it's a bad feature reference, add to bad references
                        instance_features_reference = np.loadtxt(instance_path, skiprows=1, usecols=0, dtype=np.string_)
                        i += 1
                except:
                    bad_data_paths.append(instance_path)
            prev_features_len = np.shape(instance_features)[0]
            prev_instance_features = instance_features #update feature name list
        print("There are " + str(len(bad_data_paths)) + " files with missing data.")
    return classes
    

def combine_data(data_path):
    """
    Saves the data from all datafiles into two ordered text files that can be 
    read in with np.loadtxt as numpy arrays. 
    
    params:
        data_path: the directory of the original datafiles
    """
    
    i = 0
    X = [] #feature array
    y = [] #label array
    for label in os.listdir(data_path): #iterate through cancer types
        instance_files = data_path + label + '/' #update path
        for instance_file in os.listdir(instance_files): #iterate through instance files
            if 'MANIFEST.txt' in instance_file: #skip manifest
                continue
            instance_data_paths = instance_files + instance_file + '/' #update path
            for instance_data_file in os.listdir(instance_data_paths): #iterate through instance files
                if instance_data_file == 'annotations.txt': #skip annotations
                    continue
                instance_path = instance_data_paths + instance_data_file #update path
                if i == 0:
                    instance_features_reference = np.loadtxt(instance_path, skiprows=1, usecols=0, dtype=np.string_) #save initial feature order
                    i += 1
                else:
                    instance_features = np.loadtxt(instance_path, skiprows=1, usecols=0, dtype=np.string_) #save feature values
                        
                instance_values = np.loadtxt(instance_path, skiprows=1, usecols=2, dtype=np.float) #save labels
                X.append(instance_values)
                y.append(label)
    np.save('aggregate_data/X', np.array(X)) #write to file for easy access
    np.save('aggregate_data/y', np.array(y))
    
def get_data():
    """
    Reads the combined array binary files in as numpy arrays.
    """
    X = np.load('aggregate_data/X.npy')
    y = np.load('aggregate_data/y.npy')
    
    return X, y
        