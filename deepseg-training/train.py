import os
import numpy as np
import pandas as pd
import pickle
import random
# import commands
from sklearn.utils import shuffle

# Local modules
from config_file import config
from utils import fetch_data_files, visualize_data, normalize_data, load_3Dpatches, train_model
from generator import get_training_and_validation_generators

# from msct_image import Image  # msct_image is deprecated - use spinalcordtoolbox.image instead 
# Add the spinalcordtoolbox location to the system path.
import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(dirname(abspath(__file__))), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image
# from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']

## IMPORT PARAMETERS FROM CONFIG FILE
# config['data_dict'] = pickle file containing a dictionary with at least the following keys: subject and contrast_foldname
# This dict is load as a panda dataframe and used by the function utils.fetch_data_files
# IMPORTANT NOTE: the testing dataset is not included in this dataframe

# TODO: Create data_dict (containing only training & validation data.)
# DATA_PD = pd.read_pickle(config['data_dict'])
DATA_PD = pd.read_json(config['data_dict'])
print(DATA_PD.columns)
DATA_FOLD = config["data_dir"]  # where the preprocess data are stored
MODEL_FOLD = config["path2save"]  # where to store the trained models

MEAN_TRAIN_T2, STD_TRAIN_T2 = 871.309, 557.916  # Mean and SD of the training dataset of the original paper

## CONVERT INPUT IMAGES INTO AN HDF5 FILE
len_train = int(config["training_split"] * len(DATA_PD.columns)) # 80% of the dataset is used for the training 
subj_train = random.sample(list(DATA_PD.columns), len_train)
subj_valid = [s for s in DATA_PD.columns if s not in subj_train] # the remaining images are used for the validation

# TODO: Adjust for new directory structure - return a list of tuples of relative filepaths (image, mask)
training_files = fetch_data_files(subj_train, DATA_FOLD, 't2')
validation_files = fetch_data_files(subj_valid, DATA_FOLD, 't2')

print(training_files)

# ## EXTRACT 3D PATCHES
# # The extracted patches are stored as pickle files (one for training, one for validation).
# # If these files already exist, we load them directly (i.e. do not re run the patch extraction).
# pkl_train_fname = DATA_FOLD + 'lesion_train_data_t2.pkl'
# print(pkl_train_fname)
# if not os.path.isfile(pkl_train_fname):
#     X_train, y_train = load_3Dpatches(fname_lst=training_files,patch_shape=config["patch_size"],overlap=config["patch_overlap"]) 
#     X_train = normalize_data(X_train, MEAN_TRAIN_T2, STD_TRAIN_T2)
#     X_train, y_train = shuffle(X_train, y_train, random_state=2611)
#     print(X_train.shape)
#     with open(pkl_train_fname, 'wb') as fp:
#         pickle.dump(np.array([X_train, y_train]), fp)
# else:
#     with open (pkl_train_fname, 'rb') as fp:
#         X_train, y_train = pickle.load(fp)


# pkl_valid_fname = DATA_FOLD + 'lesion_valid_data_t2.pkl'
# print(pkl_valid_fname)

# if not os.path.isfile(pkl_valid_fname):
#     X_valid, y_valid = load_3Dpatches(fname_lst=validation_files,
#                                         patch_shape=config["patch_size"],
#                                         overlap=0)
    
#     X_valid = normalize_data(X_valid, MEAN_TRAIN_T2, STD_TRAIN_T2)
    
#     with open(pkl_valid_fname, 'wb') as fp:
#         pickle.dump(np.array([X_valid, y_valid]), fp)
# else:
#     with open (pkl_valid_fname, 'rb') as fp:
#         X_valid, y_valid = pickle.load(fp)

# print 'Number of Training patches:\n\t' + str(X_train.shape[0])
# print 'Number of Validation patches:\n\t' + str(X_valid.shape[0])

# ## LOAD TRAINED MODEL
# model_fname = os.path.join(commands.getoutput('$SCT_DIR').split(': ')[2], 'data', 'deepseg_lesion_models', 't2_lesion.h5')
# model = load_trained_model(model_fname)

# ## GET TRAINING AND VALIDATION GENERATORS
# train_generator, nb_train_steps = get_training_and_validation_generators(
#                                                     [X_train, y_train],
#                                                     batch_size=config["batch_size"],
#                                                     augment=True,
#                                                     augment_flip=True)

# print(train_generator,nb_train_steps)


# validation_generator, nb_valid_steps = get_training_and_validation_generators(
#                                                     [X_valid, y_valid],
#                                                     batch_size=1,
#                                                     augment=False,
#                                                     augment_flip=False)
# print(validation_generator,nb_valid_steps)

# for g, g_name in zip([train_generator, validation_generator], ['train_visu', 'valid_visu']):
#     print '\n' + g_name
#     X_visu_, y_visu_ = g.next()
#     idx_random = random.randint(0, X_visu_.shape[-1])
#     visualize_data(X=X_visu_[0,0,:,:,idx_random], Y=y_visu_[0,0,:,:,idx_random])

# ## RUN NET ---> Cell to change --> Change it to fine-tuning / transfer learning...etc.

# train_model(model=model,
#             path2save=config["path2save"],
#             model_name=config["model_name"],
#             training_generator=train_generator,
#             validation_generator=validation_generator,
#             steps_per_epoch=nb_train_steps,
#             validation_steps=nb_valid_steps,
#             n_epochs=config["n_epochs"],
#             learning_rate_drop=config["learning_rate_drop"],
#             learning_rate_patience=config["learning_rate_patience"]
#            )