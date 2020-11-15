import cv2 as cv
import os
import numpy as np
import csv
import glob
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from utils import*
import pywt
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import signal as sg
import glob

label1 = "corona"
label2="normal"
label3="pneumonia"
dirList1 = glob.glob("GAN_Images/"+label1+"/*")
dirList2 = glob.glob("GAN_Images/"+label2+"/*")
dirList3 = glob.glob("GAN_Images/"+label3+"/*")
file = open("C:/Users/Ibrahim kholil/covid/data.csv","a")
file1 = open("C:/Users/Ibrahim kholil/covid/data1.csv","a")
file2 = open("C:/Users/Ibrahim kholil/covid/data2.csv","a")
file3 = open("C:/Users/Ibrahim kholil/covid/data3.csv","a")
file4 = open("C:/Users/Ibrahim kholil/covid/data4.csv","a")





for img_path in dirList1:
    im1 = cv.imread(img_path)
    im1=cv.resize(im1,(512,512))
    im_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
#    img_rescaled=(im_gray-np.min(im_gray))/(np.max(im_gray)-np.min(im_gray))
 #   img_rescaled = cv.cvtColor(img_rescaled,cv.COLOR_BGR2GRAY)

    im = cv.equalizeHist(im_gray)
#cv.imshow('eq',im)

   


   
	
	
    def norm(ar):
        """ Normalize IConvolved image"""
        return 255.*np.absolute(ar)/np.max(ar)



    gray2 = np.copy(im.astype(np.float64))
    (rows, cols) = im.shape[:2]

    #Create space for 16 convolutions for each kernel
    conv_maps = np.zeros((rows, cols,16),np.float64)
   
   #create an array of Laws filter vectors
    filter_vectors = np.array([[1, 4, 6,  4, 1],
                             [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])

    #Perform matrix multiplication of vectors to get 16 kernels
    filters = list()
    for ii in range(4):
        for jj in range(4):
            filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))
            #Preprocess the image
            smooth_kernel = (1/25)*np.ones((5,5))
            gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
            gray_processed = np.abs(gray2 - gray_smooth)


    #Convolve the Laws kernels
    for ii in range(len(filters)):
        conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

    #Create the 9 texture maps
    texture_maps = list()
    texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
    texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
    texture_maps.append(norm(conv_maps[:, :, 10]))
    texture_maps.append(norm(conv_maps[:, :, 15]))
    texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
    texture_maps.append(norm(conv_maps[:, :, 5]))
    texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
    texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))
    texture_map=np.concatenate((compute_14_features(texture_maps[0]), compute_14_features(texture_maps[1]),
                                  compute_14_features(texture_maps[2]),compute_14_features(texture_maps[3]),compute_14_features(texture_maps[4]),compute_14_features(texture_maps[5]),compute_14_features(texture_maps[6]),compute_14_features(texture_maps[7])), axis=0)
        
    
    
    file.write(label1)
    file.write(",")  
    for i in range(56):
        try:
            area = glcm_features[i]
            file.write(str(area))
        except:
            file.write("0")

        file.write(",")

    file.write("\n")

    file1.write(label1)
    file1.write(",")  
    for i in range(112):
        try:
            area1 = wavelet_features[i]
            file1.write(str(area1))
        except:
            file1.write("0")

        file1.write(",")

    file1.write("\n")
    
    
    

    
    file2.write(label1)
    file2.write(",")  
    for i in range(112):
        try:
            area2 = texture_map[i]
            file2.write(str(area2))
        except:
            file2.write("0")

        file2.write(",")

    file2.write("\n")
    
    file3.write(label1)
    file3.write(",")  
    for i in range(56):
        try:
            area3 = gldm_features[i]
            file3.write(str(area3))
        except:
            file3.write("0")

        file3.write(",")
    file3.write("\n")
    
    file4.write(label1)
    file4.write(",")  
    for i in range(14):
        try:
            area4 = fft_features[i]
            file4.write(str(area4))
        except:
            file4.write("0")

 
    file4.write("\n")


    
    
for img_path in dirList2:
    im1 = cv.imread(img_path)
    im1=cv.resize(im1,(512,512))
    im_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
#    img_rescaled=(im_gray-np.min(im_gray))/(np.max(im_gray)-np.min(im_gray))
 #   img_rescaled = cv.cvtColor(img_rescaled,cv.COLOR_BGR2GRAY)

    im = cv.equalizeHist(im_gray)
#cv.imshow('eq',im)

   


   
	
	
    def norm(ar):
        """ Normalize IConvolved image"""
        return 255.*np.absolute(ar)/np.max(ar)



    gray2 = np.copy(im.astype(np.float64))
    (rows, cols) = im.shape[:2]

    #Create space for 16 convolutions for each kernel
    conv_maps = np.zeros((rows, cols,16),np.float64)
   
   #create an array of Laws filter vectors
    filter_vectors = np.array([[1, 4, 6,  4, 1],
                             [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])

    #Perform matrix multiplication of vectors to get 16 kernels
    filters = list()
    for ii in range(4):
        for jj in range(4):
            filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))
            #Preprocess the image
            smooth_kernel = (1/25)*np.ones((5,5))
            gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
            gray_processed = np.abs(gray2 - gray_smooth)


    #Convolve the Laws kernels
    for ii in range(len(filters)):
        conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

    #Create the 9 texture maps
    texture_maps = list()
    texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
    texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
    texture_maps.append(norm(conv_maps[:, :, 10]))
    texture_maps.append(norm(conv_maps[:, :, 15]))
    texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
    texture_maps.append(norm(conv_maps[:, :, 5]))
    texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
    texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))
    texture_map=np.concatenate((compute_14_features(texture_maps[0]), compute_14_features(texture_maps[1]),
                                  compute_14_features(texture_maps[2]),compute_14_features(texture_maps[3]),compute_14_features(texture_maps[4]),compute_14_features(texture_maps[5]),compute_14_features(texture_maps[6]),compute_14_features(texture_maps[7])), axis=0)
        
    
    file.write(label2)
    file.write(",")  
    for i in range(56):
        try:
            area = glcm_features[i]
            file.write(str(area))
        except:
            file.write("0")

        file.write(",")

    file.write("\n")

    file1.write(label2)
    file1.write(",")  
    for i in range(112):
        try:
            area1 = wavelet_features[i]
            file1.write(str(area1))
        except:
            file1.write("0")

        file1.write(",")

    file1.write("\n")
    
    
    

    
    file2.write(label2)
    file2.write(",")  
    for i in range(112):
        try:
            area2 = texture_map[i]
            file2.write(str(area2))
        except:
            file2.write("0")

        file2.write(",")

    file2.write("\n")
    
    file3.write(label2)
    file3.write(",")  
    for i in range(56):
        try:
            area3 = gldm_features[i]
            file3.write(str(area3))
        except:
            file3.write("0")

        file3.write(",")
    file3.write("\n")
    
    file4.write(label2)
    file4.write(",")  
    for i in range(14):
        try:
            area4 = fft_features[i]
            file4.write(str(area4))
        except:
            file4.write("0")

 
    file4.write("\n")

    
for img_path in dirList3:
    im1 = cv.imread(img_path)
    im1=cv.resize(im1,(512,512))
    im_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
#    img_rescaled=(im_gray-np.min(im_gray))/(np.max(im_gray)-np.min(im_gray))
 #   img_rescaled = cv.cvtColor(img_rescaled,cv.COLOR_BGR2GRAY)

    im = cv.equalizeHist(im_gray)
#cv.imshow('eq',im)

   


   
	
	
    def norm(ar):
        """ Normalize IConvolved image"""
        return 255.*np.absolute(ar)/np.max(ar)



    gray2 = np.copy(im.astype(np.float64))
    (rows, cols) = im.shape[:2]

    #Create space for 16 convolutions for each kernel
    conv_maps = np.zeros((rows, cols,16),np.float64)
   
   #create an array of Laws filter vectors
    filter_vectors = np.array([[1, 4, 6,  4, 1],
                             [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])

    #Perform matrix multiplication of vectors to get 16 kernels
    filters = list()
    for ii in range(4):
        for jj in range(4):
            filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))
            #Preprocess the image
            smooth_kernel = (1/25)*np.ones((5,5))
            gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
            gray_processed = np.abs(gray2 - gray_smooth)


    #Convolve the Laws kernels
    for ii in range(len(filters)):
        conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

    #Create the 9 texture maps
    texture_maps = list()
    texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
    texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
    texture_maps.append(norm(conv_maps[:, :, 10]))
    texture_maps.append(norm(conv_maps[:, :, 15]))
    texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
    texture_maps.append(norm(conv_maps[:, :, 5]))
    texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
    texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))
    texture_map=np.concatenate((compute_14_features(texture_maps[0]), compute_14_features(texture_maps[1]),
                                  compute_14_features(texture_maps[2]),compute_14_features(texture_maps[3]),compute_14_features(texture_maps[4]),compute_14_features(texture_maps[5]),compute_14_features(texture_maps[6]),compute_14_features(texture_maps[7])), axis=0)
        
    
     
    
    file.write(label3)
    file.write(",")  
    for i in range(56):
        try:
            area = glcm_features[i]
            file.write(str(area))
        except:
            file.write("0")

        file.write(",")

    file.write("\n")

    file1.write(label3)
    file1.write(",")  
    for i in range(112):
        try:
            area1 = wavelet_features[i]
            file1.write(str(area1))
        except:
            file1.write("0")

        file1.write(",")

    file1.write("\n")
    
    
    

    
    file2.write(label3)
    file2.write(",")  
    for i in range(112):
        try:
            area2 = texture_map[i]
            file2.write(str(area2))
        except:
            file2.write("0")

        file2.write(",")

    file2.write("\n")
    
    file3.write(label3)
    file3.write(",")  
    for i in range(56):
        try:
            area3 = gldm_features[i]
            file3.write(str(area3))
        except:
            file3.write("0")

        file3.write(",")
    file3.write("\n")
    
    file4.write(label3)
    file4.write(",")  
    for i in range(14):
        try:
            area4 = fft_features[i]
            file4.write(str(area4))
        except:
            file4.write("0")

 
    file4.write("\n")


cv.waitKey(0)
cv.destroyAllWindows()
   
    
    

  
