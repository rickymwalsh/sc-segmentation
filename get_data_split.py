
import numpy as np
import os
import json
import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add the spinalcordtoolbox location to the system path.
import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(abspath(__file__)), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image

def get_data_split(train_prop=0.7, valid_prop=0.1, train_on_lesion_only=True, random_state=None):

	# Get a list of the subjects and their relevant centres
	subjects_df = pd.read_csv(os.path.join('data', 'subject-centres.csv'))

	if train_on_lesion_only:
		relevant_subjects_df = subjects_df[subjects_df['has_lesion']==1]
		nonlesion_subjects = subjects_df[subjects_df['has_lesion']==0]
	else:
		relevant_subjects_df = subjects_df

	# Get the numbers of subjects based on the given proportions.
	len_train = int(round(len(relevant_subjects_df)*train_prop, 0))
	len_valid = int(round(len(relevant_subjects_df)*valid_prop, 0))

	print("Training: {}, Validation: {}, Test: {}, Test (Non-lesion): {}".format(
		len_train, len_valid, len(relevant_subjects_df)-len_train-len_valid, len(nonlesion_subjects)))

	# Centre 4 has only one subject with a lesion which affects the stratified split below. So take it out first and add back later.
	centre4 = relevant_subjects_df[relevant_subjects_df['centerId']==4]
	relevant_subjects_df = relevant_subjects_df[relevant_subjects_df['centerId']!=4]

	train_subjects, valid_test_subjects = train_test_split(
		relevant_subjects_df, 
		train_size=len_train, 
		stratify=relevant_subjects_df.centerId,
		random_state=seed)

	# Similarly, 9 has only five subjects with lesions, and four of them go into the training data above.
	# Treat similarly to centre4 above.
	centre9 = valid_test_subjects[valid_test_subjects['centerId']==9]
	valid_test_subjects = valid_test_subjects[valid_test_subjects['centerId']!=9]

	validation_subjects, test_subjects = train_test_split(
		valid_test_subjects,
		train_size=len_valid,
		stratify=valid_test_subjects.centerId,
		random_state=seed)

	train_t2, train_t2s = train_test_split(
		train_subjects,
		train_size=0.5,
		stratify=train_subjects.centerId,
		random_state=seed)

	test_subjects = pd.concat([test_subjects, centre4, centre9, nonlesion_subjects])

	split_dict = {'training_t2':{}, 'training_t2s':{}, 'validation':{}, 'test':{}}

	for k,df in zip(split_dict.keys(), [train_t2, train_t2s, validation_subjects, test_subjects]):
		for i,row in df.iterrows():
			split_dict[k][str(row.patientId)] = {
				'centerId': str(row.centerId), 
				'has_lesion': str(row.has_lesion),
				't2_path': os.path.join('data', 'SCSeg', str(row.patientId), 'final', 't2_iso_onT2srig_nl.nii.gz'),
				't2s_path': os.path.join('data', 'SCSeg', str(row.patientId), 'final', 't2sMerge_iso.nii.gz'),
				'lesion_path': os.path.join('data', 'SCSeg', str(row.patientId), 'final', 'labelLesion_iso_bin.nii.gz')
			}

	# Write the data split to file.
	tstamp = datetime.now().strftime('%Y%m%d%H%M')
	outfile = os.path.join('data', 'train-test-splits', 'split_' + tstamp + '.json')
	with open(outfile, 'w') as f:
		json.dump(split_dict, f)
	print(outfile)	

	return outfile

