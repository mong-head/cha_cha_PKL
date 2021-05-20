'''
**Parking Lot Space Availability CNN (차차)**

This convolutional neural network performs binary classification task on visual input of a parking lot. 
Possible classes for individual parking lot spaces are empty and occupied.
The segmentation of the image is done 
through the use of coordinates imported from a csv file.
'''

'''
**Code Initialization**

Modules and library imports
'''

import tarfile
tar = tarfile.open("/content/drive/Shared drives/parkinglot/CNR-EXT_FULL_IMAGE_1000x750/CNR-EXT_FULL_IMAGE_1000x750.tar")
tar.extractall(path="/content/drive/Shared drives/parkinglot/CNR-EXT_FULL_IMAGE_1000x750/")

#Future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Utils imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
from datetime import datetime
from PIL import Image
import cv2
import pickle as pkl

#Tensorflow imports
import tensorflow as tf
from tensorflow import keras
try:
    %tensorflow_version 2.x
except Exception:
    pass

#Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import compute_class_weight

#Keras imports
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model

#My modules
from DataAcquisition import *
from ModelDefinition import *
from ModelTraining import *
from ModelTest import *
from DataPreprocess import * 

'''
 **Definition of Global Variables**
 
Initialization of global variables that will be used through the entire code, 
these include the directory of the dataset, the logs, saved models, and input image parameters.
'''
#Global variables definition
file_path = '/content/drive/Shared drives/parkinglot/'
model_path = file_path+'Best Model.hdf5'
logdir = file_path+"IBML-Final-Project/Training Logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#Image parameters
image_width = 150
image_height = 150
channels = 3

'''
 **If __main__ Statement**
 
End of the code!
'''
if __name__ == "__main__":
    '''
    # data acquisition
    X_busy, y_busy = DataAcquisition.busy_acquisition(image_width, image_height, channels)
    X_free, y_free = DataAcquisition.free_acquisition(image_width, image_height, channels)
    X_raw, y_raw = DataAcquisition.concatenate_dataset(X_busy, X_free, y_busy, y_free)
    DataAcquisition.save_dataset(X_raw, y_raw, file_path, X_busy, X_free, y_busy, y_free)
    X_raw, y_raw = load_dataset(file_path+''Dataset/Training/Training Set')
    '''
    
    # data preprocess
    #X_raw = DataPreprocess.blur_and_Adaptive_Threshold(X_raw)
    
    # model training
    #ModelTraining.do_stuff(image_width, image_height, channels, file_path)
    
    # testing
    vacancy = ModelTesting.predict_vacancy(file_path)
