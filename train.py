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
import argparse
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

# Local modules
from generator import get_training_and_validation_generators

from spinalcordtoolbox.spinalcordtoolbox.image import Image
from spinalcordtoolbox.spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

# Parse whether we should fine-tune on same contrast or adapt to other contrast.
parser = argparse.ArgumentParser()
parser.add_argument("-adapt", type=int,default=0, help="Should the models be finetuned on images of the same contrast (0) or adapted to the other contrast (1).")
args = parser.parse_args()
adapt = args.adapt

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
    config['preprocessed_data_file'] = str(max(data_list))
else:
    preprocessed_path = os.path.join('data','preprocessed', config['preprocessed_data_file'])


# Record the time the model was trained.
config['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')

# Create a separate directory for each new model (experiment) trained.
if adapt:
    model_dir = os.path.join(config["adapted_models"], config['timestamp'] + '_' + config["model_name"])
else:
    model_dir = os.path.join(config["finetuned_models"], config['timestamp'] + '_' + config["model_name"])
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Record the used config.
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

for contrast in ['t2', 't2s']:
    print(contrast, '....')

    opposite_contrast = 't2' if contrast=='t2s' else 't2s'

    if adapt:
        if not os.path.isdir(os.path.join(model_dir, f'{opposite_contrast}_to_{contrast}')):
            os.mkdir(os.path.join(model_dir, f'{opposite_contrast}_to_{contrast}'))
    else:
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
                                                        batch_size=config["batch_size"],
                                                        augment=False,
                                                        augment_flip=False)
    print(validation_generator,nb_valid_steps)

    # Load relevant trained model.
    if adapt:
        if config['ft_model'] is not None:
            model_fname = os.path.join('models','finetuned',config['ft_model'], opposite_contrast, f'best_{opposite_contrast}.h5')
        else:
        # Otherwise take the most recent one.
            ft_models = os.listdir(os.path.join('models','finetuned'))
            model_tstamps = [int(f[:14]) for f in ft_models]
            ft_model_name = ft_models[model_tstamps.index(max(model_tstamps))]
            model_fname = os.path.join('models','finetuned',ft_model_name, opposite_contrast, f'best_{opposite_contrast}.h5')
    else:
        model_fname = os.path.join('sct_deepseg_lesion_models', f'{contrast}_lesion.h5')

    model = load_trained_model(model_fname)
    ## Test model is working.
    # print("Test:", model.predict(X_train[[0]]).shape)

    K.set_value(model.optimizer.learning_rate, 2.5e-5)

    model.fit(train_generator,
                steps_per_epoch=nb_train_steps,
                epochs=config["n_epochs"],
                validation_data=validation_generator,
                validation_steps=nb_valid_steps,
                callbacks=get_callbacks(
                    os.path.join(model_dir, contrast), 
                    contrast,
                    learning_rate_drop=config["learning_rate_drop"],
                    learning_rate_patience=config["learning_rate_patience"]
                    )
                )


