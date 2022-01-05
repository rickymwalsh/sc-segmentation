# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:06:27 2022

@author: Ricky
"""

# TODO: 
#   Add comments
#   Set up for all experiments/models.
#   Relate written results to experiment identifier.

import nibabel as nib
import numpy as np
from scipy.spatial.distance import dice
from sklearn.metrics import recall_score, precision_score
from skimage import measure
import pandas as pd
import os
from tqdm import tqdm
import json

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


scores = {}   # Dict to store the results.

# Loop over each subject.
for subj in tqdm(os.listdir(data_dir)):
    # Create the high-level dict structure.
    scores[subj] = {'sct':{'t2':{}, 't2s':{}}, 'finetuned': {}, 'adapted': {}}
    # Filepaths for the segmentations using the SCT models.
    fpaths = {
        'gt': os.path.join(data_dir, subj, \
                           'final', 'labelLesion_iso_bin.nii.gz'),
        't2': os.path.join(data_dir, subj, \
                           'segmentation', 't2_iso_onT2srig_nl_lesionseg.nii.gz'),
        't2s': os.path.join(data_dir, subj, \
                            'segmentation', 't2sMerge_iso_lesionseg.nii.gz')
        }

    # Load the lesion masks.
    lesion_gt = load_data(fpaths['gt'])   # Ground Truth
    lesion_seg_t2 = load_data(fpaths['t2'])     # Automated segmentation of T2 using the T2 model.
    lesion_seg_t2s = load_data(fpaths['t2s'])    # Automated segmentation of T2* using the T2* model.
    
    # Calculate and store the Dice scores.
    scores[subj]['sct']['t2']['dice'] = \
        dice_coefficient(lesion_gt, lesion_seg_t2)
    scores[subj]['sct']['t2s']['dice'] = \
        dice_coefficient(lesion_gt, lesion_seg_t2s)

    # Check if there was at least one lesion in the ground truth. Otherwise these metrics don't apply.
    if np.max(lesion_gt):
        # Calculate and store the various metrics per subject.
        scores[subj]['sct']['t2']['voxel_sensitivity'] = voxel_sensitivity(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['voxel_sensitivity'] = voxel_sensitivity(lesion_gt, lesion_seg_t2s)    

        scores[subj]['sct']['t2']['voxel_precision'] = voxel_precision(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['voxel_precision'] = voxel_precision(lesion_gt, lesion_seg_t2s)    

        scores[subj]['sct']['t2']['lesion_sensitivity'], nlesions_gt = lesion_sensitivity(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['lesion_sensitivity'], _ = lesion_sensitivity(lesion_gt, lesion_seg_t2s)

        scores[subj]['sct']['t2']['lesion_precision'], nlesions_t2 = lesion_precision(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['lesion_precision'], nlesions_t2s = lesion_precision(lesion_gt, lesion_seg_t2s)

        # Record the number of lesions in the ground truth & automated segmentations.
        scores[subj]['nlesions_gt'] = nlesions_gt
        scores[subj]['sct']['t2']['nlesions'] = nlesions_t2
        scores[subj]['sct']['t2s']['nlesions'] = nlesions_t2s

    # If there is no lesion in the Ground Truth segmentation file.
    else:
        # Record the number of lesions as zero.
        scores[subj]['nlesions_gt'] = 0

        # Check if there is at least one lesion returned by the T2 segmentation.
        if np.max(lesion_seg_t2):
            _, nlesions_t2 = measure.label(lesion_seg_t2, return_num=True)
            scores[subj]['sct']['t2']['nlesions'] = nlesions_t2
            scores[subj]['sct']['t2']['true_negative'] = 0  # Used to calculate subject-wise specificity

        else:
            scores[subj]['sct']['t2']['nlesions'] = 0
            scores[subj]['sct']['t2']['true_negative'] = 1  # Used to calculate subject-wise specificity

        # Check if there is at least one lesion returned by the T2* segmentation.
        if np.max(lesion_seg_t2s):
            _, nlesions_t2s = measure.label(lesion_seg_t2s, return_num=True)
            scores[subj]['sct']['t2s']['nlesions'] = nlesions_t2s
            scores[subj]['sct']['t2s']['true_negative'] = 0  # Used to calculate subject-wise specificity

        else:
            scores[subj]['sct']['t2s']['nlesions'] = 0
            scores[subj]['sct']['t2s']['true_negative'] = 1  # Used to calculate subject-wise specificity



with open(os.path.join('results', 'scores.json'), 'w') as f:
    json.dump(scores, f)

# with open(os.path.join('results', 'scores.json'), 'r') as f:
#     scores = json.load(f)

scores = [{"subject":subj, 'results':values} for subj, values in scores.items()]

df = pd.json_normalize(scores)
df.columns = df.columns.str.replace('results.', '', regex=False)

# df.to_csv(os.path.join('results', 'subject_scores.csv'), index=False)

subj_specificity = df[['sct.t2.true_negative', 'sct.t2s.true_negative']] \
    .mean(skipna=True) \
    .to_frame(name='specificity') \
    .rename({
        'sct.t2.true_negative': 'sct.t2.subject_specificity',
        'sct.t2s.true_negative': 'sct.t2s.subject_specificity'
        }) 

summary_stats = df.drop(columns=['subject','sct.t2.true_negative', 'sct.t2s.true_negative'],axis=1) \
    .agg([
        np.nanmedian, 
        lambda x: x.quantile(q=0.25),
        lambda x: x.quantile(q=0.75)]) \
    .T 

summary_stats.columns = ['median', 'q25', 'q75']

summary_stats = summary_stats \
    .append(subj_specificity) \
    .reset_index()

summary_stats['IQR'] = summary_stats['q75'] - summary_stats['q25']

summary_stats[['model', 'data', 'metric']] = summary_stats['index'].str.split('.', 2, expand=True)

summary_stats.to_csv(os.path.join('results', 'summary_stats.csv'), index=False)
