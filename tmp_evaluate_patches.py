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

data_dir = os.path.join('data', 'SCSeg')

# Loads the image with Nibabel.
load_data = lambda fpath: nib.load(fpath).get_fdata()

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

def lesion_sensitivity(lesion_gt, lesion_seg):
    """ Function to return the proportion of ground truth lesions detected.
    input:  the ground truth lesion mask image and automated segmentation mask for one subject.
    returns:  lesion_sensitivity for one subject, and the number of distinct ground truth lesions.
     """
    labels_gt = measure.label(lesion_gt)  # Numbers the connected regions (1, 2, 3, etc.)

    # Get the unique ground truth lesions (connected regions) and how many voxels each one covers.
    labels, count = np.unique(labels_gt, return_counts=True)
    labels_overlap, count_overlap = np.unique(labels_gt*lesion_seg, return_counts=True)

    detected = 0.0
    # Loop over the distinct ground truth lesions. If part of this lesion is present in the segmentation mask, 
    #  check if the overlap is over 25% of the lesion voxels. If so, then the ground truth lesion is considered detected.
    for i,label in enumerate(labels):
        if label in labels_overlap:
            if count_overlap[labels_overlap == label]/count[i] > 0.25:
                detected += 1.0

    # Return the sensitivity, i.e., number of detected lesions divided by the total number of lesions in the ground truth.
    # Also return the total number of ground truth lesions.
    return detected/len(labels), len(labels)

def lesion_precision(lesion_gt, lesion_seg):
    """ Function to return the proportion of ground truth lesions detected.
    input:  the ground truth lesion mask image and automated segmentation mask for one subject.
    returns:  lesion_sensitivity for one subject, and the number of distinct ground truth lesions.
     """
    labels_seg = measure.label(lesion_seg)  # Numbers the connected regions (1, 2, 3, etc.)

    # Get the unique lesions (connected regions) in the automatic segmentation and how many voxels each one covers.
    labels, count = np.unique(labels_seg, return_counts=True)
    # Get the overlap between the segmentation labels and the ground truth mask.
    labels_overlap, count_overlap = np.unique(labels_seg*lesion_gt, return_counts=True)

    TP = 0.0
    # Loop over the distinct segmentation lesions. If part of this lesion was present in the ground truth mask, 
    #  check if the overlap is over 25% of the lesion voxels. If so, then the detected lesion is considered valid.
    for i,label in enumerate(labels):
        if label in labels_overlap:
            if count_overlap[labels_overlap == label]/count[i] > 0.25:
                TP += 1.0

    # Return the precision, i.e., number of valid lesions divided by total number of lesions detected. 
    # Also return the total number of lesions detected.
    return TP/len(labels), len(labels)


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

    pred_train = np.zeros(y_train.shape[0], 1, 48,48,48)
    pred_valid = np.zeros(y_valid.shape[0], 1, 48,48,48)
    pred_test = np.zeros(y_test.shape[0], 1, 48,48,48)

    for i,patch in enumerate(X_train):
        pred_train[i,:,:,:,:] = model.predict(patch)
