# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:06:27 2022

@author: Ricky
"""

import numpy as np
from scipy.spatial.distance import dice
from sklearn.metrics import recall_score, precision_score
from skimage import measure
import pandas as pd
import os
from tqdm import tqdm
import json
from copy import deepcopy
import argparse

# Local Modules.
from spinalcordtoolbox.spinalcordtoolbox.image import Image, add_suffix, zeros_like
from spinalcordtoolbox.spinalcordtoolbox.deepseg_lesion.core import apply_intensity_normalization, segment_3d
from spinalcordtoolbox.spinalcordtoolbox.deepseg_sc.core import find_centerline, crop_image_around_centerline
from spinalcordtoolbox.spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

from preprocessing import normalize_data, crop_images, extract_patches

parser = argparse.ArgumentParser()
parser.add_argument("-model_id", type=str, help="The id of the model to be evaluated (same as its folder name). To evaluate original SCT models set model_id='sct'. ")
parser.add_argument("-adapted", type=int, default=0, help="Whether we are evaluating a domain-adapted model (1) or a regular fine-tuned model (0).")
parser.add_argument("-results_from_file", default=1, type=int, help="Set to 1 if the scores.json file already exists and we want to just calculate summary statistics. \
                                                                    Set to 0 if the segmentations and subject scores need to be generated.")
parser.add_argument("-preprocessed_data", default=None, type=str, help="If the original SCT models are being evaluated, we need to know which data split to use: The timestamp/ID associated with the relevant preprocessed data.")
args = parser.parse_args()

results_from_file = args.results_from_file
model_id = args.model_id
adapted = args.adapted
preprocessed_data_sct = args.preprocessed_data

def dice_coefficient(lesion_gt, lesion_seg):
    # the dice function gives Dice dissimilarity - subtract from 1 to the get the Dice coefficient.
    return 1-dice(lesion_gt.ravel(), lesion_seg.ravel())

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
    labels, count = np.unique(labels_gt[labels_gt>0], return_counts=True)
    # Get the overlap between the ground truth lesions and the model's segmentation.
    seg_overlap = labels_gt*lesion_seg
    # If there is no overlap, return a sensitivity of 0.
    if np.max(seg_overlap) == 0:
        return 0.0, len(labels)
    # Get the unique lesions in the overlapping regions as well as the number of voxels each one covers.
    labels_overlap, count_overlap = np.unique(seg_overlap[seg_overlap>0], return_counts=True)

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
    labels, count = np.unique(labels_seg[labels_seg>0], return_counts=True)
    # Get the overlap between the segmentation labels and the ground truth mask.
    seg_overlap = labels_seg*lesion_gt
    # If there is no overlap, return a precision of 0.
    if np.max(seg_overlap) == 0:
        return 0.0, len(labels)
    # Get the unique lesions in the overlapping regions as well as the number of voxels each one covers.
    labels_overlap, count_overlap = np.unique(seg_overlap[seg_overlap>0], return_counts=True)

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

def get_prediction(model, im_path, contrast):
    # Use the already saved cropped version.
    im_path = add_suffix(im_path, '_crop')

    im = Image(im_path)

    # Apply Nyul-like percentiles normalization.
    im = apply_intensity_normalization(im, contrast)

    # Get prediction. (Apply normalization, patch_extraction, predict & reconstruct the image)
    seg_im = segment_3d(model_fname=None, contrast_type=contrast, im=im, model=model)

    # Return the segmented cropped image and the corresponding Ground Truth mask.
    return seg_im

def retrieve_data_split(data_id):
    data_split_path = os.path.join('data', 'train-test-splits', f'split_{data_id}.json')
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)

    return data_split

def generate_predictions(model_id, data_split, adapted=False):
    for model_contrast in ['t2', 't2s']:
        if model_id=='sct':
            model = load_trained_model(os.path.join('sct_deepseg_lesion_models', f'{model_contrast}_lesion.h5'))
        elif adapted:
            opposite_contrast = 't2s' if model_contrast=='t2' else 't2s'
            model = load_trained_model(os.path.join('models', 'adapted', model_id, f'{opposite_contrast}_to_{model_contrast}', f'best_{opposite_contrast}_to_{model_contrast}.h5'))            
        else:
            model = load_trained_model(os.path.join('models', 'finetuned', model_id, contrast, f'best_{model_contrast}.h5'))

        # We want to also test the t2 model on the t2s data for example, so need all 4 variations of model_contrast->data_contrast
        for data_contrast in ['t2','t2s']:
            # Loop through the subsets (training_t2, training_t2s, validation, test)
            for subset,subdict in data_split.items():
                for subj,values in subdict.items():
                    seg_im = get_prediction(model, values[f'{data_contrast}_path'], model_contrast)

                    seg_dir = os.path.join('data','SCSeg',subj,'segmentation')
                    model_dir = os.path.join(seg_dir, model_id)
                    if not os.path.isdir(model_dir):
                        if not os.path.isdir(seg_dir):
                            os.mkdir(seg_dir)
                        os.mkdir(model_dir)

                    seg_im.save(os.path.join(model_dir,  f'{model_contrast}-model_{data_contrast}-data.nii.gz'))


def compute_scores(model_id, data_split):

    scores = {}

    for subset in data_split:
        for subj in tqdm(data_split[subset]):
            scores[subj] = {'subset':subset} # Create a new sub-dict for each subject.

            for data_contrast in ['t2','t2s']:
                # Read in the ground truth image for this subject for the relevant contrast.
                gt_path = add_suffix(data_split[subset][subj]['lesion_path'], f'_crop_{data_contrast}')
                gt = Image(gt_path).data.astype(np.uint8)

                for model_contrast in ['t2','t2s']:
                    # Read in the segmented image based on the data contrast and the model contrast.
                    seg_path = os.path.join('data','SCSeg',subj,'segmentation',model_id, f'{model_contrast}-model_{data_contrast}-data.nii.gz')
                    seg = Image(seg_path).data.astype(np.uint8)

                    res_tmp = {}
                    res_tmp['n_lesion_voxels_gt'] = int(np.sum(gt))
                    res_tmp['n_lesion_voxels_seg'] = int(np.sum(seg))
                    res_tmp['total_voxels'] = len(gt.ravel())
                    res_tmp['lesion_load_gt'] = float(res_tmp['n_lesion_voxels_gt'])/res_tmp['total_voxels']
                    res_tmp['lesion_load_seg'] = float(res_tmp['n_lesion_voxels_seg'])/res_tmp['total_voxels']

                    # Check if there was at least one lesion in the ground truth. Otherwise these metrics don't apply.
                    if np.max(gt):
                        res_tmp['dice'] = dice_coefficient(gt, seg)

                        res_tmp['voxel_sensitivity'] = voxel_sensitivity(gt, seg)
                        res_tmp['lesion_sensitivity'], res_tmp['nlesions_gt'] = lesion_sensitivity(gt, seg)
                        # Check if there is at least one lesion returned in the segmentation.
                        if np.max(seg):
                            res_tmp['voxel_precision'] = voxel_precision(gt, seg)
                            res_tmp['lesion_precision'], res_tmp['nlesions_seg'] = lesion_precision(gt, seg)
                        else: 
                            res_tmp['nlesions_seg'] = 0


                    # If there is no lesion in the Ground Truth segmentation file.
                    else:
                        # Record the number of lesions as zero.
                        res_tmp['nlesions_gt'] = 0

                        # Check if there is at least one lesion returned by the segmentation.
                        if np.max(seg):
                            _, res_tmp['nlesions_seg'] = measure.label(seg, return_num=True)
                            res_tmp['true_negative'] = 0  # Used to calculate subject-wise specificity

                        else:
                            res_tmp['nlesions_seg'] = 0
                            res_tmp['true_negative'] = 1  # Used to calculate subject-wise specificity

                    scores[subj][f'{model_contrast}-model_{data_contrast}-data'] = deepcopy(res_tmp)

    return scores


def main():

    if model_id=='sct':
        # If the SCT models are to be evaluated, the relevant data to be used should be supplied when calling the script.
        data_id = preprocessed_data_sct

    else:
        if adapted:
            model_dir = os.path.join('models', 'adapted', model_id)
        else:
            model_dir = os.path.join('models', 'finetuned', model_id)

        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        # Retrieve the relevant file name containing the data split used to train the model.
        data_id = config['preprocessed_data_file']

    # Read the dict containing the subject IDs and data in the train/valid/test sets.
    data_split = retrieve_data_split(data_id)

    # Create a directory for the model results
    results_dir = os.path.join('results', model_id)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        print(f'Created directory: {results_dir}')

    if results_from_file==0:
        # Get all of the segmentations and write them to file.
        generate_predictions(model_id, data_split)

        # Get all of the result scores for the model.
        scores = compute_scores(model_id, data_split)

        with open(os.path.join(results_dir, 'scores.json'), 'w') as f:
            json.dump(scores, f)

    else:
        with open(os.path.join(results_dir, 'scores.json'), 'r') as f:
            scores = json.load(f)

    scores = [{"subject":subj, 'results':values} for subj, values in scores.items()]

    df = pd.json_normalize(scores)
    df.columns = df.columns.str.replace('results.', '', regex=False)

    print(df.columns)
    print(df.head())

    df.to_csv(os.path.join(results_dir, 'subject_scores.csv'), index=False)

    subj_specificity = df[['t2-model_t2-data.true_negative', 't2-model_t2s-data.true_negative', 't2s-model_t2-data.true_negative', 't2s-model_t2s-data.true_negative']] \
        .mean(skipna=True) \
        .to_frame(name='specificity') \
        .rename({
            't2-model_t2-data.true_negative': 't2-model_t2-data.subject_specificity',
            't2-model_t2s-data.true_negative': 't2-model_t2s-data.subject_specificity',
            't2s-model_t2-data.true_negative': 't2s-model_t2-data.subject_specificity',
            't2s-model_t2s-data.true_negative': 't2s-model_t2s-data.subject_specificity'
            }) 

    # Define custom aggregation functions.
    def q1(x): return x.quantile(q=0.25)
    def q3(x): return x.quantile(q=0.75)
    def IQR(x): return q3(x) - q1(x)

    df['subset'] = np.where(df['subset']=='test' & df['t2-model_t2-data.nlesions_gt']==0, 'test_no_lesions',
                        np.where(df['subset']=='test', 'test_lesions',
                            df['subset'])

    summary_stats = df \
        .drop(columns=['subject','t2-model_t2-data.true_negative', 't2-model_t2s-data.true_negative', \
                        't2s-model_t2-data.true_negative', 't2s-model_t2s-data.true_negative'], axis=1) \
        .groupby(by='subset') \
        .agg([np.nanmedian, q1, q3, IQR]) \
        .T  

    print(summary_stats.head())

    summary_stats = summary_stats \
        .append(subj_specificity) \
        .reset_index()

    summary_stats[['model--data', 'metric']] = summary_stats['index'].str.split('.', 2, expand=True)

    summary_stats.to_csv(os.path.join(results_dir, 'summary_stats.csv'), index=False)

if __name__=="__main__":
    main()
