import numpy as np
import os
import json
import random
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Local modules
from config_file import config

from spinalcordtoolbox.spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.spinalcordtoolbox.deepseg_lesion.core import apply_intensity_normalization
from spinalcordtoolbox.spinalcordtoolbox.deepseg_sc.core import find_centerline, crop_image_around_centerline

def normalize_data(data, contrast):
	"""Function to normalize data based on learned mean and std from SCT project."""
	params = {'t2': {'mean': 871.309, 'std': 557.916},
				't2s': {'mean': 1011.31, 'std': 678.985}}

	data -= params[contrast]['mean']
	data /= params[contrast]['std']
	return data

def crop_images(im_image, im_lesion, contrast, crop_size=48):
    """
    Use SCT functions to find the centerline and crop the image and its associated lesion mask around the centerline.
    :param im_image: Image() object to crop
    :param contrast: Constrast of the image. 't2' or 't2s'
    :param crop_size: int denoting the square crop_size around the SC in the axial plane.
    :return: cropped image.
    """

    # find the spinal cord centerline - execute OptiC binary
    _, im_ctl, _ = find_centerline(algo='svm', image_fname=im_image, contrast_type=contrast, brain_bool=False)

    # crop image around the spinal cord centerline
    _,_,_, im_crop_nii = crop_image_around_centerline(im_in=im_image, ctr_in=im_ctl, crop_size=crop_size)
    _,_,_, lesion_crop = crop_image_around_centerline(im_in=im_lesion, ctr_in=im_ctl, crop_size=crop_size)

    return im_crop_nii, lesion_crop

def extract_patches(im_data, patch_size):
	z_patch_size = patch_size[2]
	z_step_keep = list(range(0, im_data.shape[2], z_patch_size))
	patch_arr = np.empty([len(z_step_keep)] + list(patch_size))

	for i,zz in enumerate(z_step_keep):
		if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
			patch_im = np.zeros(patch_size)
			z_patch_extracted = im_data.shape[2] - zz
			patch_im[:, :, :z_patch_extracted] = im_data[:, :, zz:]
		else:
			patch_im = im_data[:, :, zz:z_patch_size + zz]

		patch_arr[i,:,:,:] = patch_im

	# add an extra axis (to represent the single channel)
	return np.expand_dims(patch_arr, axis=1)


def main():
	patch_size = config['patch_size']

	# Read in the list of subjects in each dataset.
	if config['train_test_split'] is None:
		splits = os.listdir(os.path.join('data', 'train-test-splits')) # Get a list of the splits.
		# Get the latest split, based on the saved timestamp.
		tstamps = [int(s[-17:-5]) for s in splits]  # The timestamp is saved at the end of the filename.
		latest_split = splits[tstamps.index(max(tstamps))]
		with open(os.path.join('data','train-test-splits',latest_split), 'r') as f:
			data_split = json.load(f)

		# Save timestamp for use later - link with processed data.
		data_tstamp = max(tstamps)
	else:
		split_file = os.path.join('data','train-test-splits',config['train_test_split'])
		with open(split_file, 'r') as f:
			data_split = json.load(f)
		# Extract and save timestamp for use later - link with processed data
		data_tstamp = int(split_file[-17:-5])

	# For each dataset:
	# 	Loop through each subject.
	# 		Apply preprocessing
	# 			1. Detect centerline & Crop image around SC (48mm)
	# 			1b. Save image in original folder. 
	# 			2. Intensity normalization (with percentiles)  (apply_intensity_normalization)
	# 			3. Mean/Std. Dev normalization
	# 			4. Patch Extraction
	# 	Save combined processed patches to .npz

	for subset in data_split:
		for contrast in ['t2','t2s']:
			print(f'Processing {subset} ({contrast}) data')
			im_patches = []; lesion_patches = []
			for subj in tqdm(data_split[subset]):
				# Path where the cropped (48x48) images are saved.		
				im_cropped_path = add_suffix(data_split[subset][subj][contrast+'_path'], '_crop')
				lesion_cropped_path = add_suffix(data_split[subset][subj]['lesion_path'], f'_crop_{contrast}')
				
				# Check if the cropped images already exist.
				if os.path.isfile(im_cropped_path) & os.path.isfile(lesion_cropped_path):
					im_cropped = Image(im_cropped_path)
					lesion_cropped = Image(lesion_cropped_path)

				else:
					# Read the image and its associated lesion mask.
					im_orig = Image(data_split[subset][subj][contrast+'_path'])
					im_lesion = Image(data_split[subset][subj]['lesion_path'])

					# Crop the image and its associated lesion mask around the SC.
					im_cropped, lesion_cropped = crop_images(
						im_orig, im_lesion, contrast=contrast, crop_size=patch_size[0])

					# Save the cropped images to file.
					im_cropped.save(im_cropped_path)
					lesion_cropped.save(lesion_cropped_path)

				# Apply Nyul-like percentiles normalization.
				im_cropped = apply_intensity_normalization(im_cropped, contrast)

				# Apply mean & std. deviation normalization
				im_cropped.data = normalize_data(im_cropped.data, contrast)

				# Extract the individual patches.
				im_patches.append(extract_patches(im_cropped.data, patch_size))
				lesion_patches.append(extract_patches(lesion_cropped.data, patch_size))

			# Combine the list of patches
			im_patches = np.concatenate(im_patches)
			lesion_patches = np.concatenate(lesion_patches)

			# Save to .npz archive.
			if not os.path.isdir(os.path.join('data', 'preprocessed', str(data_tstamp))):
				os.mkdir(os.path.join('data', 'preprocessed', str(data_tstamp)))
			outfile = os.path.join('data', 'preprocessed',str(data_tstamp), f'{subset}_{contrast}.npz')
			np.savez(outfile, im_patches=im_patches, lesion_patches=lesion_patches)

if __name__=='__main__':
	main()