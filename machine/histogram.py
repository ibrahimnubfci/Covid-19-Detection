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
file = open("C:/Users/Ibrahim kholil/covid/dataset.csv","a")
file1 = open("C:/Users/Ibrahim kholil/covid/dataset1.csv","a")
file2 = open("C:/Users/Ibrahim kholil/covid/dataset2.csv","a")
file3 = open("C:/Users/Ibrahim kholil/covid/dataset3.csv","a")
file4 = open("C:/Users/Ibrahim kholil/covid/dataset4.csv","a")
for img_path in dirList1:
    im1 = cv.imread(img_path)
    im1=cv.resize(im1,(512,512))
    im = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    
    
#cv.imshow('eq',im)

    
    ret, thresh = cv.threshold(im,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#cv.imshow('th',thresh)
# noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
#cv.imshow('opening',opening)
# sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
#cv.imshow('dialate',sure_bg)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#cv.imshow('dist',dist_transform)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)
#cv.imshow('dialate1',sure_bg)
# Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(im1,markers)
#cv.imshow('mark',markers)
    im1[markers == -1] = [0,255,0]
#mg2 = color.label2rgb(markers, bg_label=1)
#file.write(label1)


    #cv.imshow('imj',im_gray)
    #markers = cv.watershed(im1,markers)
    #file.write(str(markers))
#p

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = im1.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

# define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
    k = 3
    compactness, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(im1.shape)

# show the image
#plt.imshow(segmented_image)
#plt.show()

# disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(im1)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]

# convert back to original shape
    masked_image = masked_image.reshape(im1.shape)
# show the image
#cv.imshow('knn',masked_image)
#plt.imshow(masked_image)
#plt.show()
    
    masked_image= cv.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    #masked_image = cv.equalizeHist(masked_image)


    img_rescaled=(masked_image-np.min(masked_image))/(np.max(masked_image)-np.min(masked_image)) 
    wavelet_coeffs = pywt.dwt2(img_rescaled,'sym4')
    cA1, (cH1, cV1, cD1) = wavelet_coeffs
    wavelet_coeffs = pywt.dwt2(cA1,'sym4')
    cA2, (cH2, cV2, cD2) = wavelet_coeffs#wavelet features
    wavelet_features=np.concatenate((compute_14_features(cA1), compute_14_features(cH1),compute_14_features(cV1),compute_14_features(cD1)
                                     ,compute_14_features(cA2), compute_14_features(cH2),compute_14_features(cV2),compute_14_features(cD2)), axis=0)

# =============================================================================
    glcms =greycomatrix(masked_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])#GLCM in four directions
    glcm_features=np.concatenate((compute_14_features(im2double(glcms[:, :, 0, 0])), 
                                  compute_14_features(im2double(glcms[:, :, 0, 1])),
                                  compute_14_features(im2double(im2double(glcms[:, :, 0, 2]))),
                                  compute_14_features(glcms[:, :, 0, 3])), axis=0)


    
    
    fft_map=np.fft.fft2(img_rescaled)
    fft_map = np.fft.fftshift(fft_map)
    fft_map = np.abs(fft_map)
    YC=int(np.floor(fft_map.shape[1]/2)+1)
    fft_map=fft_map[:,YC:int(np.floor(3*YC/2))]
    fft_features=compute_14_features(fft_map)#FFT features


    gLDM1,gLDM2,gLDM3,gLDM4=GLDM(img_rescaled,10)#GLDM in four directions
    gldm_features=np.concatenate((compute_14_features(gLDM1), compute_14_features(gLDM2),
                                  compute_14_features(gLDM3),compute_14_features(gLDM4)), axis=0)
    

	
	
    def norm(ar):
        """ Normalize IConvolved image"""
        return 255.*np.absolute(ar)/np.max(ar)



    gray2 = np.copy(masked_image.astype(np.float64))
    (rows, cols) = masked_image.shape[:2]

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










cv.waitKey(0)
cv.destroyAllWindows()
   
    
    

  
