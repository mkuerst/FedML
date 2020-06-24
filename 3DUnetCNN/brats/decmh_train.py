import os
import glob
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import h5py

import fetch_data

from keras import backend as K
K.common.set_image_dim_ordering('th')

#NEEDED TO USE MULTIPLE GPUS
from keras.utils import multi_gpu_model 

# needed to be able to import the unet3d module
sys.path.append('/local/home/kumichae/3DUnet/3DUnetCNN')

from unet3d.model import unet_model_3d, isensee2017_model, isensee_global_model, isensee_multiheads_model, \
    isensee_soft_mh_model, is_decmh_model, is_tl_decmh_model
from unet3d.training import load_old_model, train_model
from unet3d.generator import get_training_and_validation_generators
from unet3d.data import write_data_to_file, open_data_file

# DEFINE VISIBLE GPUS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# get rid of logging warning etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed





all_tasks = ['Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus', 'Task05_Prostate',
             'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
# This determines what shape the images will be cropped/resampled to. (CHANGED 144,144,144)
config["image_shape"] = (144,144,48)
# switch to None to train on the whole image (CHANGED 64,64,64)
config["patch_shape"] = None
config["labels"] = (1, 2)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["MRI"]  # ["t1", "t1ce", "flair", "t2"]
# change this if you want to only use some of the modalities
config["training_modalities"] = config["all_modalities"]
config["nb_channels"] = 1
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
# if False, will use upsampling instead of deconvolution
config["deconvolution"] = False

config["n_base_filters"] = 16
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 200  # 500  # cutoff the training after this many epochs
# learning rate will be reduced after this many epochs if the validation loss is not improving
config["patience"] = 10
# 50  # training will be stopped after this many epochs without the validation loss improving
config["early_stop"] = 50
config["initial_learning_rate"] = 0.00001
# factor by which the learning rate will be reduced
config["learning_rate_drop"] = 0.5
# portion of the data that will be used for training
config["validation_split"] = 0.8


# data shape must be a cube. Augments the data by permuting in various directions
config["permute"] = False
config["flip"] = False # augments the data by randomly flipping an axis during
config["distort"] = None#0.25  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]


# if > 0, during training, validation patches will be overlapping
config["validation_patch_overlap"] = 0
# (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["training_patch_start_offset"] = (0, 0, 0)
# if True, then patches without any target will be skipped
config["skip_blank"] = True

config["data_file"] = os.path.abspath("brats_data.h5")
config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")

# If True, will previous files. If False, will use previously written files.
config["overwrite"] = True

datasets_path = '/local/home/kumichae/3DUnet/datasets'


def fetch_training_data_files(task):
    # FOR LOCAL MODEL
    # training_data_files, \
    # config['labels'], config['n_labels'] = fetch_data.prepare_data(task[1])
    # config['labels'] = (1,2)
    # config['n_labels'] = len(config['labels'])

    # FOR GLOBAL MODEL
    # Seconds param makes sure to only train on one dataset with global labels though, index for whta dataset to train on
    # 0 := Hippo | 1 := Prostate | 2 := Heart
    # Change to None to train an all
    training_data_files, config['labels'], config['n_labels'] = fetch_data.prepare_global_data(task, None, True)
    
    config['batch_size'] = 1
    config['patch_shape'] = None
    config['image_shape'] = (64,64,48)
    if "patch_shape" in config and config["patch_shape"] is not None:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
    else:
        config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

    print("input shape: ")
    print(config["input_shape"])
    return training_data_files


def main(tasks, overwrite=False):
    use_existing = True

    # DECMH MODEL
    task_gl = 'global(64,64,48)_is_tl_decmh_bal'
    config["data_file"] = os.path.abspath(task_gl+".h5")
    config["model_file"] = os.path.abspath(task_gl+"_model.h5")
    config["training_file"] = os.path.abspath(task_gl+"_training_ids.pkl")
    config["validation_file"] = os.path.abspath(task_gl+"_validation_ids.pkl")
    
    log_dir = config["model_file"].replace("_model.h5", "_logs")

    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files = fetch_training_data_files(tasks)

    if not os.path.exists(config["data_file"]) or not use_existing:
            write_data_to_file(
                training_files, config["data_file"], image_shape=config["image_shape"])
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        # strategy = tf.distribute.MirroredStrategy()
        # strategy = strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # with strategy.scope():
        # model,metrics = unet_model_3d(input_shape=config["input_shape"],
        #                     pool_size=config["pool_size"],
        #                     n_labels=config["n_labels"],
        #                     initial_learning_rate=config["initial_learning_rate"],
        #                     deconvolution=config["deconvolution"])
        #with tf.device("/cpu:0"):
        model = is_tl_decmh_model(input_shape=config['input_shape'],n_labels=config['n_labels'])
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False,rankdir='TB', \
                                    expand_nested=False, dpi=96)
        print("saved model graph")
        #model = multi_gpu_model(model,gpus=2)    
        # model.compile(optimizer=opt,loss=loss,metrics=ldcm)
        # from keras.optimizers import Adam
        # from unet3d.metrics import weighted_dice_coefficient_loss, dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
        # model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coefficient_loss, metrics=metrics) 

    # get training and testing generators
    
    nr_hippo = 120+25
    nr_prost = 120+13
    nr_heart = 120+4

    print(data_file_opened.root.label.shape)
    print(data_file_opened.root.label[0][0])
    print(data_file_opened.root.label[nr_hippo][0])
    print(data_file_opened.root.label[nr_hippo+nr_prost][0])

    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

        # train_generator = tf.data.Dataset.from_generator(train_generator,output_types=(tf.float32, tf.uint8))
        # validation_generator = tf.data.Dataset.from_generator(validation_generator,output_types=(tf.float32, tf.uint8))

        # train_generator = strategy.experimental_distribute_dataset(train_generator)
        # validation_generator= strategy.experimental_distribute_dataset(validation_generator)

    if not os.path.exists('./GeneratorImages'):
        os.makedirs('GeneratorImages')

    cwd = os.getcwd()
    affine = np.diag(np.ones(4))

    print(data_file_opened.root.truth.shape)
    hdf5_truth_img = data_file_opened.root.truth[0][0]
  

    hdf5_truth_img = nib.Nifti1Image(hdf5_truth_img,affine)
    hdf5_truth_img.to_filename(os.path.join(cwd,"GeneratorImages/hdf5_truth_img"))
    print("saved hdf5_truth_img")
  
    # for i in range(10):
    #     features, targets = next(train_generator)
    #     features = features[0]
    #     targets = np.hstack((targets[0],targets[1],targets[2])) 
    #     # print(features.shape)
    #     # print(targets.shape)
    #     j = 0
    #     for j in range(targets.shape[1]):
    #         targets_image = nib.Nifti1Image(targets[0][j], affine)
    #         targets_image.to_filename(os.path.join(cwd,"GeneratorImages/{}_targets_lab{}.nii".format(i,j)))

    #     features_image = nib.Nifti1Image(features[0][0], affine)
    #     features_image.to_filename(os.path.join(cwd,"GeneratorImages/{}_features.nii".format(i)))
    #     features, targets = next(validation_generator)
        
    #     features = features[0]
    #     targets = np.hstack((targets[0],targets[1],targets[2])) 
    #     # print(features.shape)
    #     # print(targets.shape)

    #     k = 0
    #     for k in range(targets.shape[1]):
    #         targets_image = nib.Nifti1Image(targets[0][k], affine)
    #         targets_image.to_filename(os.path.join(cwd,"GeneratorImages/{}_targets_validation_ḷab{}.nii".format(i,k)))

    #     features_image = nib.Nifti1Image(features[0][0], affine)
    #     features_image.to_filename(os.path.join(cwd,"GeneratorImages/{}_features_validation.nii".format(i)))
    
    print("saved generator images")



    # run training
    train_model(data_file=config['data_file'],model=model,
                model_file=config["model_file"],
                log_dir=log_dir,
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    main(['Task04_Hippocampus','Task05_Prostate','Task02_Heart'], overwrite=config["overwrite"])


# Standard Isensee params : droput=0.3 | depth = 5 | learning rate 5e-4