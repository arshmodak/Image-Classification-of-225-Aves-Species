import os
import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


data_dictionary = dict()
labels = dict()
descriptor = dict()
bag_of_vw = dict() #{folder:}

def read_data():
    path = r"D:\Northeastern courses\DS 5220\project\dataset"
    for folder in os.listdir(path)[:-1]:
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

def sift_features(labels ,images):
    sift_vectors = {}
    descriptor_list = []
    
    sift = cv2.xfeatures2d.SIFT_create()
    for key,img in zip(labels,images):
        features = []
        kp, des = sift.detectAndCompute(img,None)
        descriptor_list.extend(des)
        features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

def generate_bag_of_vw():
    
    for key,value in data_dictionary.items():
        if key not in descriptor:
            descriptor[key] = list()
        if key not in bag_of_vw:
            bag_of_vw[key] = list()
        
        descriptor_list,sift_vectors = sift_features(labels[key],data_dictionary[key])
        descriptor[key] = descriptor_list
        bag_of_vw[key] = sift_vectors

    return

# A k-means clustering algorithm who takes 2 parameter which is number 
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words
    


# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


# 1-NN algorithm. We use this for predict the class of test images.
# Takes 2 parameters. images is the feature vectors of train images and tests is the feature vectors of test images
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
def knn(images, tests, train_labels, test_labels):
    num_test = 0
    correct_predict = 0
    class_based = {}
    
    classifier = KNeighborsClassifier(n_neighbours = 1)
    classifier.fit(images,train_labels)
    y_pred = classifier.predict(test)
    print(confusion_matrix(test_labels, y_pred))
    
    # for test_key, test_val in zip(test_labels,tests):
    #     class_based[test_key] = [0, 0] # [correct, all]
    #     predict_start = 0
    #     #print(test_key)
    #     minimum = 0
    #     key = "a" #predicted
    #     for train_key, train_val in  zip(train_labels,images):
    #           if(predict_start == 0):
    #               minimum = distance.euclidean(test_val, train_val)
    #               #minimum = L1_dist(tst,train)
    #               key = train_key
    #               predict_start += 1
    #           else:
    #               dist = distance.euclidean(test_val, train_val)
    #               #dist = L1_dist(tst,train)
    #               if(dist < minimum):
    #                   minimum = dist
    #                   key = train_key
          
    #     if(test_key == key):
    #         correct_predict += 1
    #         class_based[test_key][0] += 1
    #     num_test += 1
    #     class_based[test_key][1] += 1
    #       #print(minimum)
    return [num_test, correct_predict, class_based]

# Calculates the average accuracy and class based accuracies.  
def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))
        


if __name__ == "__main__":
    read_data()
    generate_bag_of_vw()
    visual_words = kmeans(150,descriptor['train'])
    # Creates histograms for train data    
    bovw_train = image_class(descriptor['train'], visual_words) 
    # Creates histograms for test data
    bovw_test = image_class(descriptor['test'], visual_words) 
    results_bowl = knn(bovw_train, bovw_test) 
    # Calculates the accuracies and write the results to the console.       
    # accuracy(results_bowl) 
    