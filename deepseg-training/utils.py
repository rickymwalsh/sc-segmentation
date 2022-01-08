import pickle
import tables
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.ndimage.measurements import center_of_mass
from skimage.exposure import rescale_intensity
from skimage.util.shape import view_as_blocks

import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(dirname(abspath(__file__))), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# **TODO** : switch to 'channels_first' when running on GPU.
K.set_image_data_format('channels_last')


################################ LOAD DATA ################################

def fetch_data_files(subjects, data_folder, contrast):
    '''             
    Input:
        - subjects: list of subjects
        - data_folder: absolute path of the data folder
        - contrast: 'T2' or 'T2s'
        
    Output: a list of tuples, each containing the absolute path of both the image and its related groundtruth
    '''
    data_files = list()

    # fname = {'t2': 't2_iso_onT2srig_nl.nii.gz', 't2s': 't2sMerge_iso.nii.gz', 'lesion': 'labelLesion_iso_bin.nii.gz'}

    # for s in subjects:
    #     im_fname = os.path.join(data_folder, str(s), 'final', fname[contrast])
    #     gt_fname = os.path.join(data_folder, str(s), 'final', fname['lesion'])
    #     if os.path.isfile(im_fname) and os.path.isfile(gt_fname):
    #         subject_files = [im_fname, gt_fname]
    #         data_files.append(tuple(subject_files))
    # return data_files

    fname = {'t2': 'crop_t2.nii.gz', 't2s': 'crop_t2s.nii.gz', 'lesion': 'crop_lesion.nii.gz'}

    for s in subjects:
        im_fname = os.path.join(
            '/mnt/c/Users/Ricky/Documents/Masters/Second Year/Case Study/ot_da_v0/Data/SCSeg/', str(s), 'SC/res', fname[contrast])
        gt_fname = os.path.join(
            '/mnt/c/Users/Ricky/Documents/Masters/Second Year/Case Study/ot_da_v0/Data/SCSeg/', str(s), 'SC/res', fname['lesion'])
        if os.path.isfile(im_fname) and os.path.isfile(gt_fname):
            subject_files = [im_fname, gt_fname]
            data_files.append(tuple(subject_files))
    return data_files


def normalize_data(data, mean, std):
    '''Normalize data (numpy array) by substracting mean (float) then dividing by std (float).'''
    data -= mean
    data /= std
    return data


def load_3Dpatches(fname_lst, patch_shape, overlap=None):
    '''
    Extract 3D patches from a set of images.
    
    Input:
        - fname_lst: list of list, where each sublist contains the absolute path of both the image and its related groundtruth mask
        - patch_shape: tuple 3 int numbers indicating the size of patches to extract (in voxel)
        - overlap: int indicating the number of voxel overlap between each extracted patch in the third dimension
        
    Return:
        Two numpy arrays (image and groundtruth) with the following dimensions:
            (N, 1, patch_shape[0], patch_shape[1], patch_shape[2])
            where N is the total number of extracted patches.
    '''
    x_size, y_size, z_size = patch_shape
    X, y = [], []
    for fname in fname_lst:
        print(fname[0])
        if os.path.isfile(fname[0]) and os.path.isfile(fname[1]):
            im, gt = Image(fname[0]), Image(fname[1])
            if  np.any(gt.data):
                im_data, gt_data = im.data.astype(np.float32), gt.data.astype(np.int8)

                z_max = im_data.shape[2]

                z_step_keep = range(0, z_max, z_size)
                z_data_crop_max = max(z_step_keep) + z_size

                im_data_crop = np.zeros((x_size, y_size, z_data_crop_max))
                gt_data_crop = np.zeros((x_size, y_size, z_data_crop_max))

                im_data_crop[:, :, :z_max] = im_data
                gt_data_crop[:, :, :z_max] = gt_data
                

               # z_max = im_data.shape[1]

                #z_step_keep = range(0, z_max, z_size)
                #z_data_crop_max = max(z_step_keep) + z_size

                #im_data_crop = np.zeros((x_size, y_size, z_data_crop_max))
                #gt_data_crop = np.zeros((x_size, y_size, z_data_crop_max))
                
               # print x_size,y_size,z_data_crop_max,z_max
                #print gt_data_crop

                #im_data_crop[:, :, :z_max] = im_data[:,:,:48]
                #gt_data_crop[:, :, :z_max] = gt_data[:,:,:48]

                
                #print(im_data_crop.shape)

                z_step_keep = range(0, z_max, overlap) if overlap else range(0, z_max, z_size)
                for zz in z_step_keep:
                    if im_data_crop[:, :, zz:zz+z_size].shape[2] == z_size:
                        #print(im_data_crop.shape)
                        X.append(im_data_crop[:, :, zz:zz+z_size])
                        y.append(gt_data_crop[:, :, zz:zz+z_size])

    return np.expand_dims(np.array(X), axis=1), np.expand_dims(np.array(y), axis=1)


################################ VISUALIZATION ################################

def visualize_data(X, Y):    
    '''Utility function to visualize the processed patches on a slice by slice basis'''
    plt.figure()
    plt.imshow(X, 'gray', interpolation='none')
    if 1 in Y:
        masked = np.ma.masked_where(Y == 0, Y)
        plt.imshow(masked, 'jet', interpolation='none', alpha=0.4)
    plt.show()


################################ TRAINING ################################

def get_callbacks(path2save, fname, learning_rate_drop=None, learning_rate_patience=50):
    model_checkpoint_best = ModelCheckpoint(path2save + '/best_' + fname + '.h5', save_best_only=True)
    tensorboard = TensorBoard(log_dir=path2save + "/logs/{}".format(fname))
    if learning_rate_drop:
        patience = ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=1)
        return [model_checkpoint_best, tensorboard, patience]
    else:
        return [model_checkpoint_best, tensorboard]


def train_model(model, path2save, model_name, training_generator, validation_generator, steps_per_epoch, validation_steps, n_epochs, learning_rate_drop=None, learning_rate_patience=50):
    '''
    Train a Keras model.
    
    Input:
        - model: Keras model that will be trained.
        - path2save: Folder path to save the model.
        - model_name: Model name.
        - training_generator: Generator that iterates through the training data.
        - validation_generator: Generator that iterates through the validation data.
        - steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
        - validation_steps: Number of batches that the validation generator will provide during a given epoch.
        - n_epochs: Total number of epochs to train the model.
        - learning_rate_drop: How much at which to the learning rate will decay.
        - learning_rate_patience: Number of epochs after which the learning rate will drop.
    '''
    model.fit(training_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=n_epochs,
                # validation_data=validation_generator,
                # validation_steps=validation_steps,
                # use_multiprocessing=True,
                # callbacks=get_callbacks(
                #     path2save, 
                #     model_name,
                #     learning_rate_drop=learning_rate_drop,
                #     learning_rate_patience=learning_rate_patience
                #     )
                )
