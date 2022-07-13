import os
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers import concatenate
from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.preprocessing.image import transform_matrix_offset_center
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages



K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.



def Utrecht_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    #print("utrecht",np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    #print("133",(image_rows_Dataset/2-rows_standard/2),(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2),(image_cols_Dataset/2+cols_standard/2))
    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_T1 = brain_mask_T1[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def Utrecht_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:,(image_rows_Dataset-rows_standard)/2:(image_rows_Dataset+rows_standard)/2,(image_cols_Dataset-cols_standard)/2:(image_cols_Dataset+cols_standard)/2] = pred[:,:,:,0]
    
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def GE3T_preprocessing(FLAIR_image, T1_image):
    #print("GE3T",np.shape(FLAIR_image))
    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, int(cols_standard/2-image_cols_Dataset/2):int(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = T1_image[:, start_cut:start_cut+rows_standard, :]
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def GE3T_postprocessing(FLAIR_array, pred):
    start_slice = 11
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, (rows_standard-image_cols_Dataset)/2:(rows_standard+image_cols_Dataset)/2,0]

    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred



def main():
    data=""
    for patient  in range(60):
        #patient=2
        flair=True
        t1=True
        if patient < 20: dir = 'raw/Utrecht/'
        elif patient < 40: dir = 'raw/Singapore/'
        else: dir = 'raw/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        #img_shape = (rows_standard, cols_standard, flair+t1)
        print("shape",patient, imgs_test.shape)
        #data.append(imgs_test)
        if isinstance(data, str):#type(data) == "str":
            print("patient",patient)
            data=imgs_test
        else:
            print("patient",patient)
            data=np.concatenate((data, imgs_test), axis=0)
        print(data.shape)
        np.save('./wmh_dataset_preprocessed/train.npy', data)
        print(imgs_test)
        """
        #patient=20
        flair=True
        t1=True
        if patient < 20: dir = 'raw/Utrecht/'
        elif patient < 40: dir = 'raw/Singapore/'
        else: dir = 'raw/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        mask = sitk.ReadImage(dir + '/wmh.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        mask = sitk.GetArrayFromImage(mask)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        #img_shape = (rows_standard, cols_standard, flair+t1)
        print("shape",patient, imgs_test.shape,mask.shape,T1_array.shape,FLAIR_array.shape)
        #data.append(imgs_test)

        #patient=55
        flair=True
        t1=True
        if patient < 20: dir = 'raw/Utrecht/'
        elif patient < 40: dir = 'raw/Singapore/'
        else: dir = 'raw/GE3T/'
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient%20]
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()
        #img_shape = (rows_standard, cols_standard, flair+t1)
        print("shape",patient, imgs_test.shape)
        #data.append(imgs_test)
        """
    
    

if __name__=='__main__':
    main()
