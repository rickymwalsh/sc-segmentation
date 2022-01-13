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
parser.add_argument("-model_id", type=str, help="The id of the model to be evaluated (same as its folder name).")
parser.add_argument("-results_from_file", default=1, type=int, help="Set to 1 if the scores.json file already exists and we want to just calculate summary statistics. \
                                                                    Set to 0 if the segmentations and subject scores need to be generated.")
args = parser.parse_args()

results_from_file = args.results_from_file
model_id = args.model_id


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

def generate_predictions(model_id, data_split):
    for contrast in ['t2', 't2s']:
        model = load_trained_model(os.path.join('models', 'finetuned', model_id, contrast, f'best_{contrast}.h5'))

        # Loop through the subsets (training_t2, training_t2s, validation, test)
        for subset,subdict in data_split.items():
            for subj,values in subdict.items():
                seg_im = get_prediction(model, values[f'{contrast}_path'], contrast)

                seg_dir = os.path.join('data','SCSeg',subj,'segmentation')
                model_dir = os.path.join(seg_dir, model_id)
                if not os.path.isdir(model_dir):
                    if not os.path.isdir(seg_dir):
                        os.mkdir(seg_dir)
                    os.mkdir(model_dir)

                seg_im.save(os.path.join(model_dir, contrast + '.nii.gz'))


def compute_scores(model_id, data_split):

    scores = {}

    for subset in data_split:
        for subj in tqdm(data_split[subset]):
            scores[subj] = {'subset':subset, 't2':{}, 't2s':{}}

            for contrast in ['t2','t2s']:

                seg_path = os.path.join('data','SCSeg',subj,'segmentation',model_id, f'{contrast}.nii.gz')
                gt_path = add_suffix(data_split[subset][subj]['lesion_path'], f'_crop_{contrast}')

                seg = Image(seg_path).data.astype(np.uint8)
                gt = Image(gt_path).data.astype(np.uint8)

                res_tmp = {}
                res_tmp['dice'] = dice_coefficient(gt, seg)
                res_tmp['n_lesion_voxels_gt'] = int(np.sum(gt))
                res_tmp['n_lesion_voxels_seg'] = int(np.sum(seg))
                res_tmp['total_voxels'] = len(gt.ravel())
                res_tmp['lesion_load_gt'] = float(res_tmp['n_lesion_voxels_gt'])/res_tmp['total_voxels']
                res_tmp['lesion_load_seg'] = float(res_tmp['n_lesion_voxels_seg'])/res_tmp['total_voxels']

                # Check if there was at least one lesion in the ground truth. Otherwise these metrics don't apply.
                if np.max(gt):
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

                scores[subj][contrast] = deepcopy(res_tmp)

    return scores


def main():

    model_dir = os.path.join('models', 'finetuned', model_id)

    # Retrieve the relevant train-test-split used for this model.
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    data_id = config['preprocessed_data_file']
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

    subj_specificity = df[['t2.true_negative', 't2s.true_negative']] \
        .mean(skipna=True) \
        .to_frame(name='specificity') \
        .rename({
            't2.true_negative': 't2.subject_specificity',
            't2s.true_negative': 't2s.subject_specificity'
            }) 

    summary_stats = df.drop(columns=['subject','t2.true_negative', 't2s.true_negative'],axis=1) \
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

    summary_stats.to_csv(os.path.join(results_dir, 'summary_stats.csv'), index=False)

if __name__=="__main__":
    main()
