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



def Utrecht_preprocessing(FLAIR_image):

    channel_num = 1
    #print("utrecht",np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    #T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    
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

def GE3T_preprocessing(FLAIR_image):
    #print("GE3T",np.shape(FLAIR_image))
    channel_num = 1
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    #T1_image = np.float32(T1_image)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
   
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
        FLAIR_image = sitk.ReadImage(dir + '/wmh.nii.gz')
        #T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        print(FLAIR_array.shape)
        print(FLAIR_array)
        #T1_array = sitk.GetArrayFromImage(T1_image)
        if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array)
        #if not flair: imgs_test = imgs_test[..., 1:2].copy()
        #if not t1: imgs_test = imgs_test[..., 0:1].copy()
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
        np.save('./wmh_dataset_preprocessed/masks.npy', data)
        #print(imgs_test)
        
    
    

if __name__=='__main__':
    main()
