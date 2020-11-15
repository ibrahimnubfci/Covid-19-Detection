import cv2
import glob
import numpy as np
import scipy.io as sio
from scipy.stats import skew
from scipy.stats import kurtosis
import pywt
from skimage.feature import greycomatrix
import scipy.io as sio
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics
import matplotlib.pyplot as plt


import os
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import KernelPCA

from IPython.display import Image
import itertools
import os
from sklearn.metrics import roc_curve, auc

# =============================================================================
# im2double
# =============================================================================
def im2double(img):
    """ convert image to double format """
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out
# =============================================================================
# compute_14_features
# =============================================================================
def compute_14_features(region):
    """ Compute 14 features """
    temp_array=region.reshape(-1)
    all_pixels=temp_array[temp_array!=0]
#    Area
    Area = np.sum(all_pixels)
#    mean
    density = np.mean(all_pixels)
#   Std
    std_Density = np.std(all_pixels)
#   skewness
    Skewness = skew(all_pixels)
#   kurtosis
    Kurtosis = kurtosis(all_pixels)
#   Energy
    ENERGY =np.sum(np.square(all_pixels))
#   Entropy
    value,counts = np.unique(all_pixels, return_counts=True)
    p = counts / np.sum(counts)
    p =  p[p!=0]
    ENTROPY =-np.sum( p*np.log2(p));
#   Maximum
    MAX = np.max(all_pixels)
#   Mean Absolute Deviation
    sum_deviation= np.sum(np.abs(all_pixels-np.mean(all_pixels)))
    mean_absolute_deviation = sum_deviation/len(all_pixels)
#   Median
    MEDIAN = np.median(all_pixels)
#   Minimum
    MIN = np.min(all_pixels)
#   Range
    RANGE = np.max(all_pixels)-np.min(all_pixels)
#   Root Mean Square
    RMS = np.sqrt(np.mean(np.square(all_pixels))) 
#    Uniformity
    UNIFORMITY = np.sum(np.square(p))

    features = np.array([Area, density, std_Density,
        Skewness, Kurtosis,ENERGY, ENTROPY,
        MAX, mean_absolute_deviation, MEDIAN, MIN, RANGE, RMS, UNIFORMITY])
    return features
# =============================================================================
# GLDM
# =============================================================================
def GLDM(img, distance):
    """ GLDM in four directions """
    pro1=np.zeros(img.shape,dtype=np.float32)
    pro2=np.zeros(img.shape,dtype=np.float32)
    pro3=np.zeros(img.shape,dtype=np.float32)
    pro4=np.zeros(img.shape,dtype=np.float32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            if((j+distance)<img.shape[1]):
                pro1[i,j]=np.abs(img[i,j]-img[i,(j+distance)])
            if((i-distance)>0)&((j+distance)<img.shape[1]):
                pro2[i,j]=np.abs(img[i,j]-img[(i-distance),(j+distance)])
            if((i+distance)<img.shape[0]):
                pro3[i,j]=np.abs(img[i,j]-img[(i+distance),j])
            if((i-distance)>0)&((j-distance)>0):
                pro4[i,j]=np.abs(img[i,j]-img[(i-distance),(j-distance)])

    n=256;
    cnt, bin_edges=np.histogram(pro1[pro1!=0], bins=np.arange(n)/(n-1), density=False)
    Out1 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro2[pro2!=0], bins=np.arange(n)/(n-1), density=False)
    Out2 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro3[pro3!=0], bins=np.arange(n)/(n-1), density=False)
    Out3 = cnt.cumsum()
    cnt, bin_edges=np.histogram(pro4[pro4!=0], bins=np.arange(n)/(n-1), density=False)
    Out4 = cnt.cumsum()
    return Out1,Out2,Out3,Out4
# =============================================================================
#   show model
