import nibabel as nib
import numpy as np
from scipy.spatial.distance import dice
from sklearn.metrics import recall_score, precision_score
from skimage import measure
import pandas as pd
import os
from tqdm import tqdm
import json

# Add the spinalcordtoolbox location to the system path.
import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(abspath(__file__)), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

def dice_coefficient(lesion_gt, lesion_seg):
    """ Returns the Dice coefficient for one subject given the ground truth lesion mask and segmentation mask for that subject. """

    # If both the ground truth mask and automated segmentation mask contain no positive (lesion) voxels, 
    # the function will throw an error (division by zero). So first check that this is not the case.
    if any(lesion_seg.ravel()) | any(lesion_gt.ravel()):
        # the dice function gives Dice dissimilarity - subtract from 1 to the get the Dice coefficient.
        return 1-dice(lesion_gt.ravel(), lesion_seg.ravel())
    # If neither mask contains any lesion pixels, this is a good result - the model has not returned a false positive lesion.
    # Thus, return the maximum Dice coefficient score of 1.
    else:
        return 1

def voxel_sensitivity(lesion_gt, lesion_seg):
    return recall_score(lesion_gt.ravel(), lesion_seg.ravel())

def voxel_precision(lesion_gt, lesion_seg):
    return precision_score(lesion_gt.ravel(), lesion_seg.ravel())


models_list = os.listdir(os.path.join('models', 'finetuned'))
tstamps = [int(f[:14]) for f in models_list]
latest_model = models_list[tstamps.index(max(tstamps))]


for contrast in ['t2', 't2s']:
    data_list = [int(f) for f in os.listdir(os.path.join('data','preprocessed'))]
    preprocessed_path = os.path.join('data','preprocessed', str(max(data_list)))

    data_train = np.load(os.path.join(preprocessed_path, f'training_{contrast}_{contrast}.npz'))
    data_valid = np.load(os.path.join(preprocessed_path, f'validation_{contrast}.npz'))
    data_test = np.load(os.path.join(preprocessed_path, f'test_{contrast}.npz'))

    X_train = data_train['im_patches']
    y_train = data_train['lesion_patches']

    X_valid = data_valid['im_patches']
    y_valid = data_valid['lesion_patches']

    X_test = data_test['im_patches']
    y_test = data_test['lesion_patches']

    model = load_trained_model(os.path.join('models', 'finetuned', latest_model, contrast, f'best_{contrast}.h5'))

    pred_train = np.zeros(y_train.shape)
    pred_valid = np.zeros(y_valid.shape)
    pred_test = np.zeros(y_test.shape)

    train_results = np.zeros((y_train.shape[0], 3))
    valid_results = np.zeros((y_valid.shape[0], 3))
    test_results = np.zeros((y_test.shape[0], 3))

    for i in range(X_train.shape[0]):
        pred_train[i,:,:,:,:] = model.predict(X_train[[i]])
        train_results[i, 0] = dice_coefficient(X_train[i], pred_train[i])
        train_results[i, 1] = voxel_sensitivity(X_train[i], pred_train[i])
        train_results[i, 2] = voxel_precision(X_train[i], pred_train[i])

    for i in range(X_valid.shape[0]):
        pred_valid[i,:,:,:,:] = model.predict(X_valid[[i]])
        valid_results[i, 0] = dice_coefficient(X_valid[i], pred_valid[i])
        valid_results[i, 1] = voxel_sensitivity(X_valid[i], pred_valid[i])
        valid_results[i, 2] = voxel_precision(X_valid[i], pred_valid[i])

    for i in range(X_test.shape[0]):
        pred_train[i,:,:,:,:] = model.predict(X_test[[i]])
        test_results[i, 0] = dice_coefficient(X_test[i], pred_test[i])
        test_results[i, 1] = voxel_sensitivity(X_test[i], pred_test[i])
        test_results[i, 2] = voxel_precision(X_test[i], pred_test[i])

    np.savez(f'quick_results_{contrast}.npz', train_results=train_results, valid_results=valid_results, test_results=test_results)