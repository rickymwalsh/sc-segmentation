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

def dice_coefficient(lesion_gt, lesion_seg):
    if any(lesion_seg.ravel()) | any(lesion_gt.ravel()):
        return 1-dice(lesion_gt.ravel(), lesion_seg.ravel())
    else:
        return 1

def voxel_sensitivity(lesion_gt, lesion_seg):
    return recall_score(lesion_gt.ravel(), lesion_seg.ravel())

def voxel_precision(lesion_gt, lesion_seg):
    return precision_score(lesion_gt.ravel(), lesion_seg.ravel())

def lesion_sensitivity(lesion_gt, lesion_seg):
    labels_gt = measure.label(lesion_gt)

    labels, count = np.unique(labels_gt, return_counts=True)
    labels_overlap, count_overlap = np.unique(labels_gt*lesion_seg, return_counts=True)

    detected = 0.0
    for i,label in enumerate(labels):
        if label in labels_overlap:
            if count_overlap[labels_overlap == label]/count[i] > 0.25:
                detected += 1.0

    return detected/len(labels), len(labels)

def lesion_precision(lesion_gt, lesion_seg):
    labels_seg = measure.label(lesion_seg)

    labels, count = np.unique(labels_seg, return_counts=True)
    labels_overlap, count_overlap = np.unique(labels_seg*lesion_gt, return_counts=True)

    TP = 0.0
    for i,label in enumerate(labels):
        if label in labels_overlap:
            if count_overlap[labels_overlap == label]/count[i] > 0.25:
                TP += 1.0

    return TP/len(labels), len(labels)

# Loads the image with Nibabel and then flattens to 1 dimension.
load_data = lambda fpath: nib.load(fpath).get_fdata()

scores = {}

for subj in tqdm(os.listdir(data_dir)[0:5]):
    scores[subj] = {'sct':{'t2':{}, 't2s':{}}, 'finetuned': {}, 'adapted': {}}
    fpaths = {
        'gt': os.path.join(data_dir, subj, \
                           'final', 'labelLesion_iso_bin.nii.gz'),
        't2': os.path.join(data_dir, subj, \
                           'segmentation', 't2_iso_onT2srig_nl_lesionseg.nii.gz'),
        't2s': os.path.join(data_dir, subj, \
                            'segmentation', 't2sMerge_iso_lesionseg.nii.gz')
        }
        
    lesion_gt = load_data(fpaths['gt'])
    lesion_seg_t2 = load_data(fpaths['t2'])
    lesion_seg_t2s = load_data(fpaths['t2s'])
    
    scores[subj]['sct']['t2']['dice'] = \
        dice_coefficient(lesion_gt, lesion_seg_t2)
    scores[subj]['sct']['t2s']['dice'] = \
        dice_coefficient(lesion_gt, lesion_seg_t2s)

    if np.max(lesion_gt):
        scores[subj]['sct']['t2']['voxel_sensitivity'] = \
            voxel_sensitivity(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['voxel_sensitivity'] = \
            voxel_sensitivity(lesion_gt, lesion_seg_t2s)    

        scores[subj]['sct']['t2']['voxel_precision'] = \
            voxel_precision(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['voxel_precision'] = \
            voxel_precision(lesion_gt, lesion_seg_t2s)    

        scores[subj]['sct']['t2']['lesion_sensitivity'], nlesions_gt = \
            lesion_sensitivity(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['lesion_sensitivity'], _ = \
            lesion_sensitivity(lesion_gt, lesion_seg_t2s)

        scores[subj]['sct']['t2']['lesion_precision'], nlesions_t2 = \
            lesion_precision(lesion_gt, lesion_seg_t2)
        scores[subj]['sct']['t2s']['lesion_precision'], nlesions_t2s = \
            lesion_precision(lesion_gt, lesion_seg_t2s)

        scores[subj]['nlesions_gt'] = nlesions_gt
        scores[subj]['sct']['t2']['nlesions'] = nlesions_t2
        scores[subj]['sct']['t2s']['nlesions'] = nlesions_t2s

    # If there is no lesion in the Ground Truth segmentation file.
    else:
        scores[subj]['nlesions_gt'] = 0

        if np.max(lesion_seg_t2):
            _, nlesions_t2 = measure.label(lesion_seg_t2, return_num=True)
            scores[subj]['sct']['t2']['nlesions'] = nlesions_t2
            scores[subj]['sct']['t2']['true_negative'] = 0  # Used to calculate subject-wise specificity

        else:
            scores[subj]['sct']['t2']['nlesions'] = 0
            scores[subj]['sct']['t2']['true_negative'] = 1  # Used to calculate subject-wise specificity

        if np.max(lesion_seg_t2):
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

print(subj_specificity)

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
