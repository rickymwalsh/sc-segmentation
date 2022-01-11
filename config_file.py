import os

config = dict()
config["seed"] = 260
config["training_split"] = 0.8
config["train_test_split"] = None	# Can specify the file containing a specific train/test split 
config["preprocessed_data_file"] = None 	# Can specify the directory containing a specific set of preprocessed data.
config["patch_size"] = (48, 48, 48)  # Size of the patches to extract
config["patch_overlap"] = 24  # Size of the overlap between the extracted patches along the third dimension
config["batch_size"] = 32  # Size of the batches that the generator will provide
config["n_epochs"] = 260  # Total number of epochs to train the model
config["augment"] = True  # If True, training data will be distorted on the fly so as to avoid over-fitting
config["augment_flip"] = True  # if True and augment is True, then the data will be randomly flipped along the x, y and z axis
config["learning_rate_drop"] = 0.5  # How much at which to the learning rate will decay
config["learning_rate_patience"] = 10  # Number of epochs after which the learning rate will drop

config["gpu_id"] = '-1'
config["data_dir"] = '../data/SCSeg/'
# Temporary fix to run the training script **TODO: Create data dict with training subjects.
config["data_dict"] = '../results/scores.json'  # pickle file containing a dictionary with at least the following keys: subject and contrast_foldname

# Model name containing the main parameters, which is useful for the hyperparm optimization
config["model_name"] = '_'.join([config["gpu_id"],
                                 str(config["batch_size"]),
                                 str(config["n_epochs"]),
                                 str(config["learning_rate_drop"]),
                                 str(config["learning_rate_patience"])
                                ])

config["finetuned_models"] = '../models/finetuned'  # Relative path of the folder where the fine-tuned models are saved
config["adapted_models"] = '../models/adapted'  # Relative path of the folder where the adapted models are saved
