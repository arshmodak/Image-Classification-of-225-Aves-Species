import pandas as pd
import numpy as np
import cv2
import mahotas
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

import os
# from skimage.io import imread_collection
data_dictionary = dict()
labels = dict()
mapping = dict()

def read_data():
    path = r"D:\Northeastern courses\DS 5220\project\dataset"
    for folder in os.listdir(path)[:1]:
        print("reading: " + folder)
        if folder not in data_dictionary:
            data_dictionary[folder] = list()
        if folder not in labels:
            labels[folder] = list()
        for image_class in os.listdir(path + "\\" +folder):
            for image in os.listdir(path + "\\" +folder + '\\' + image_class):
                data_dictionary[folder].append(cv2.imread(path + "\\" +folder + '\\' + image_class + '\\' + image))
                labels[folder].append(image_class)
    
    return 

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(image).mean(axis = 0)
    return feature

def fd_histogram(image):
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram,histogram)
    return histogram.flatten()

def generate_global_features():
    
    le = LabelEncoder() 
    scaler = MinMaxScaler(feature_range = (0,1))
    for folder in data_dictionary:
        print("extracting features for: " + folder)
        new_feature_list = list()
        for image in data_dictionary[folder]:
            one = fd_hu_moments(image)
            two = fd_haralick(image)
            three = fd_histogram(image)
            new_feature_list.append(np.hstack([one,two,three]))
        print("Normalizing extracted feature vectors: "+ folder)
        # data_dictionary[folder] = scaler.fit_transform(new_feature_list)
        data_dictionary[folder] = np.array(new_feature_list)
        labels[folder] = le.fit_transform(labels[folder])
        if folder not in mapping:
            mapping[folder] = list()
        mapping[folder] = pd.DataFrame(le.inverse_transform(labels[folder]))
            
    return
def models():
    # test =  pd.read_csv(r"D:\Northeastern courses\DS 5220\project\output\test_extracted_features.csv", index_col= False)
    # train =  pd.read_csv(r"D:\Northeastern courses\DS 5220\project\output\train_extracted_features.csv", index_col= False)
    # validate =  pd.read_csv(r"D:\Northeastern courses\DS 5220\project\output\valid_extracted_features.csv", index_col= False)
    # mapping =  pd.read_csv(r"D:\Northeastern courses\DS 5220\project\output\ class_mapping.csv", index_col= False)
    # test_y = test.iloc[:,-1]
    # train_y = train.iloc[:,-1]
    # test = test.drop([test.columns[0],test.columns[-1]],axis=1)
    # train = train.drop([test.columns[0],train.columns[-1]],axis=1)
    svm = SVC(kernel = 'linear', gamma = 'auto', random_state= 101)
    svm.fit(pd.DataFrame(data_dictionary['train']), labels['train'])
    y_pred = svm.predict(data_dictionary['test'])
    # print(classification_report(labels['test'],y_pred,target_names = mapping))
    print(accuracy_score(test_y,y_pred))
    return

def write_file():
    path = r"D:\Northeastern courses\DS 5220\project\output"
    # for folder in data_dictionary:
        # npa_images = pd.DataFrame(data_dictionary[folder])
        # npa_labels = pd.DataFrame(labels[folder])
        # npa = pd.concat([npa_images,npa_labels], axis = 1)
        # npa.to_csv(path + "\\" + folder + '_extracted_features.csv')
    mapping['train'][0].unique().to_csv(path + "\\" + ' class_mapping.csv')
        
    return

if __name__ == "__main__":
    
    read_data()
    generate_global_features()
    # write_file()
    models()
                
                