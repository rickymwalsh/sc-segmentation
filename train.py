import os

# Local modules
from config_file import config

# Set visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']

import numpy as np
import pandas as pd
import pickle
import random
from datetime import datetime
import json
# import commands
from sklearn.utils import shuffle

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

# from utils import fetch_data_files, visualize_data, normalize_data, load_3Dpatches, train_model
from generator import get_training_and_validation_generators

# Add the spinalcordtoolbox location to the system path.
import sys
from os.path import dirname, abspath, join as oj
path_to_sct = oj(dirname(abspath(__file__)), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

K.set_image_data_format('channels_first')

def get_callbacks(path2save, fname, learning_rate_drop=None, learning_rate_patience=50):
    model_checkpoint_best = ModelCheckpoint(path2save + '/best_' + fname + '.h5', save_best_only=True)
    tensorboard = TensorBoard(log_dir=path2save + "/logs/{}".format(fname))
    if learning_rate_drop:
        patience = ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=1)
        return [model_checkpoint_best, tensorboard, patience]
    else:
        return [model_checkpoint_best, tensorboard]


if config['preprocessed_data_file'] is None:
    data_list = [int(f) for f in os.listdir(os.path.join('data','preprocessed'))]
    preprocessed_path = os.path.join('data','preprocessed', str(max(data_list)))
else:
    preprocessed_path = os.path.join('data','preprocessed', config['preprocessed_data_file'])


# Record the time the model was trained.
config['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')

# Create a separate directory for each new model (experiment) trained.
model_dir = os.path.join(config["finetuned_models"], config['timestamp'] + '_' + config["model_name"])
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Record the used config.
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

for contrast in ['t2', 't2s']:
    print(contrast, '....')

    if not os.path.isdir(os.path.join(model_dir, contrast)):
        os.mkdir(os.path.join(model_dir, contrast))
    
    # Load the whole set of preprocessed data.
    data_train = np.load(os.path.join(preprocessed_path, f'training_{contrast}_{contrast}.npz'))
    data_valid = np.load(os.path.join(preprocessed_path, f'validation_{contrast}.npz'))

    X_train = data_train['im_patches']
    y_train = data_train['lesion_patches']

    X_valid = data_valid['im_patches']
    y_valid = data_valid['lesion_patches']

    X_train, y_train = shuffle(X_train, y_train, random_state=config['seed'])

    print('Number of Training patches:\n\t' + str(X_train.shape[0]))
    print('Number of Validation patches:\n\t' + str(X_valid.shape[0]))

    ## Get training and validation generators.
    train_generator, nb_train_steps = get_training_and_validation_generators(
                                                        [X_train, y_train],
                                                        batch_size=config["batch_size"],
                                                        augment=True,
                                                        augment_flip=True)

    print(train_generator,nb_train_steps)

    validation_generator, nb_valid_steps = get_training_and_validation_generators(
                                                        [X_valid, y_valid],
                                                        batch_size=1,
                                                        augment=False,
                                                        augment_flip=False)
    print(validation_generator,nb_valid_steps)

    model_fname = os.path.join('sct_deepseg_lesion_models', f'{contrast}_lesion.h5')
    model = load_trained_model(model_fname)
    ## Test model is working.
    # print("Test:", model.predict(X_train[[0]]).shape)

    model.fit(train_generator,
                steps_per_epoch=nb_train_steps,
                epochs=config["n_epochs"],
                validation_data=validation_generator,
                validation_steps=nb_valid_steps,
                use_multiprocessing=True,
                callbacks=get_callbacks(
                    os.path.join(model_dir, contrast), 
                    contrast,
                    learning_rate_drop=config["learning_rate_drop"],
                    learning_rate_patience=config["learning_rate_patience"]
                    )
                )

    # TODO: Add in model checkpoint.

# TODO: Add JDOT model - use Class saved in other module.

