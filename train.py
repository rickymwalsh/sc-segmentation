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
path_to_sct = oj(dirname(abspath(__file__)), 'spinalcordtoolbox')
sys.path.append(path_to_sct)

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']

if config['preprocessed'] is None:
    data_list = [int(f) for f in os.listdir(os.path.join('data','preprocessed'))]
    preprocessed_path = os.path.join('data','preprocessed', str(max(data_list)))
else:
    preprocessed_path = os.path.join('data','preprocessed', config['preprocessed'])


for contrast in ['t2', 't2s']:
    print(contrast, '....')
    X_train, y_train = np.load(preprocessed_path, f'training_{contrast}_{contrast}.npz')

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


    # # for g, g_name in zip([train_generator, validation_generator], ['train_visu', 'valid_visu']):
    # #     print('\n' + g_name)
    # #     X_visu_, y_visu_ = next(g)
    # #     idx_random = random.randint(0, X_visu_.shape[-1])
    # #     visualize_data(X=X_visu_[0,0,:,:,idx_random], Y=y_visu_[0,0,:,:,idx_random])


    model.fit(train_generator,
                steps_per_epoch=nb_train_steps,
                epochs=config["n_epochs"],
                validation_data=validation_generator,
                validation_steps=nb_valid_steps,
                use_multiprocessing=True,
                callbacks=get_callbacks(
                    config["path2save"], 
                    config["model_name"],
                    learning_rate_drop=config["learning_rate_drop"],
                    learning_rate_patience=config["learning_rate_patience"]
                    )
                )
    # Add in model checkpoint.

