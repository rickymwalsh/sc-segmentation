# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:46:19 2021

@author: Ricky
"""

import os
import logging
import warnings
import argparse

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.core import find_centerline, crop_image_around_centerline

logger = logging.getLogger(__name__)

def extract_patches(subj_dir='', results_dir=None, ctr_algo='svm', patch_size=(48,48,48)):
    """
    Extract patches around the centerline.
    :param subj_dir: directory containing the images from which patches will be extracted.
    :param ctr_algo: Which algorithm to use to detect the SC centerline.
    :param patch_size: patch size in the X,Y,Z axes
    :return:
    """
    if results_dir is None:
        results_dir=subj_dir
    
    fnames = {'t2s': 't2sMerge_iso.nii.gz', 
              't2': 't2_iso_onT2srig_nl.nii.gz',
              'lesion': 'labelLesion_iso_bin.nii.gz'
              }
    
    fpaths = {f: os.path.join(subj_dir, fnames[f]) for f in fnames}

    # Check if one of the result files already exists in the directory. If so, skip.
    if os.path.exists(os.path.join(results_dir, 'crop_' + fnames['t2s'] + '.nii.gz')):
        print(os.path.join(results_dir, 'crop_' + fnames['t2s'] + '.nii.gz'), 'already exists. Skipping....')
        return None 
    
    # Ensure all the input files exist, otherwise skip.
    for f in fpaths:
        if not os.path.exists(fpaths[f]):
            warning_txt = 'Failed - {} does not exist.'.format(fpaths[f])
            warnings.warn(warning_txt)
            return None
    
    ims = {'t2s': Image(fpaths['t2s']),
           't2': Image(fpaths['t2']),
           'lesion': Image(fpaths['lesion'])
           }
    
    # find the spinal cord centerline - execute OptiC binary
    logger.info("\nFinding the spinal cord centerline...")
    _, im_ctl, im_labels_viewer = find_centerline(algo=ctr_algo,
                                                    image_fname=fpaths['t2s'],
                                                    contrast_type='t2s',
                                                    brain_bool=False,
                                                    folder_output=None,
                                                    remove_temp_files=False,
                                                    centerline_fname=None)

    # crop image around the spinal cord centerline
    logger.info("\nCropping the image around the spinal cord...")
    
    # Crop all three images using the T2* SC centerline.
    ims_crop = {}
    for f in ims:
        _, _, _, im_crop_nii = crop_image_around_centerline(im_in=ims[f],
                                                        ctr_in=im_ctl,
                                                        crop_size=patch_size[0])        
        ims_crop[f] = im_crop_nii
        
        im_crop_nii.save(path = 'crop_' + f + '.nii.gz')
    
        z_patch_size = patch_size[2]
        z_step_keep = list(range(0, im_crop_nii.data.shape[2], z_patch_size))
        patch_arr = np.empty([len(z_step_keep)] + list(patch_size))
    
        for i,zz in enumerate(z_step_keep):
            if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
                patch_im = np.zeros(patch_size)
                z_patch_extracted = im_crop_nii.data.shape[2] - zz
                patch_im[:, :, :z_patch_extracted] = im_crop_nii.data[:, :, zz:]
            else:
                z_patch_extracted = z_patch_size
                patch_im = im_crop_nii.data[:, :, zz:z_patch_size + zz]
                
            patch_arr[i,:,:,:] = patch_im
            
            # Save image patches to file.
            np.save(os.path.join(results_dir, 'patches_' + f + '.npy'), patch_arr)
    
    
def main():
    parser = argparse.ArgumentParser(description='Extract patches from all of the processed subject images.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help=
                        'Absolute path to the directory containing the data. \
                        This directory should contain a separate folder for each of the subjects.')
                        
    args = parser.parse_args()
    data_dir = args.data_dir
    orig_dir = os.getcwd()
    
    for subj in os.listdir(data_dir):
        parent_dir = '' if data_dir is None else data_dir
        print('Subject', subj)
        res_dir = os.path.join(parent_dir, subj, 'SC', 'res')
        if os.path.isdir(res_dir):
            try:
                os.chdir(res_dir)
                extract_patches(subj_dir='', results_dir=None, ctr_algo='svm', patch_size=(48,48,48))
            except Exception as e:
                warning_txt = 'Error for subject ' + subj + ':\n\n' + str(e)
                warnings.warn(warning_txt)
        else:
            warning_txt='Results directory {} does not exist, please check the supplied data_dir.'.format(res_dir)
            warnings.warn(warning_txt)
        os.chdir(orig_dir)
    
if __name__=="__main__":
    main()