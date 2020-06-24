import math
from functools import partial
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
import h5py

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient, 
                            label_wise_dice_coefficient,get_label_dice_coefficient_function, get_label_dice_loss)

K.common.set_image_dim_ordering('th')


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file,log_dir,logging_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))

    log_dir = log_dir+ datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
  
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0,
                                    write_graph=False, write_images=False, write_grads=False,
                                    embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    
    callbacks.append(tensor_board)

    return callbacks


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'label_0_dice_coef': get_label_dice_coefficient_function(label_index=0),
                      'label_1_dice_coef': get_label_dice_coefficient_function(label_index=0),
                      'label_2_dice_coef': get_label_dice_coefficient_function(label_index=1),
                      'label_3_dice_coef': get_label_dice_coefficient_function(label_index=0),
                      'label_4_dice_coef': get_label_dice_coefficient_function(label_index=1),
                      'label_5_dice_coef': get_label_dice_coefficient_function(label_index=0),
                      'label_02_dice_loss': get_label_dice_loss([0,0,0],1),
                      'label_01_dice_loss': get_label_dice_loss([0,0,0],1),
                      'label_5_dice_loss': get_label_dice_loss([0,0,0],1),
                      'out12_label_1_dice_coef': get_label_dice_coefficient_function(0),
                      'out12_label_2_dice_coef': get_label_dice_coefficient_function(1),
                      'out34_label_3_dice_coef': get_label_dice_coefficient_function(0),
                      'out34_label_4_dice_coef': get_label_dice_coefficient_function(1),
                      'out5_label_5_dice_coef': get_label_dice_coefficient_function(0)}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def train_model(data_file,model, model_file,log_dir, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """

    model.fit(training_generator, #generator=
                        steps_per_epoch=steps_per_epoch,
                        use_multiprocessing=True,
                        workers = 0, #should solve thread unsafe h5 file problems but ...
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=get_callbacks(model_file, log_dir=log_dir,
                                                logging_file=os.path.basename(log_dir)+'.log',
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))
    # with strat.scope()
    #     model.fit(training_generator,
    #                         steps_per_epoch=steps_per_epoch,
    #                         use_multiprocessing=True,#ADDED THIS --> crashes the whole thing when on cpu
    #                         #workers = 2, #should solve thread safe h5 file problems but uses only one gpu now
    #                         epochs=n_epochs,
    #                         validation_data=validation_generator,
    #                         validation_steps=validation_steps,
    #                         callbacks=get_callbacks(model_file, log_dir=log_dir,
    #                                                 logging_file=os.path.basename(log_dir)+'.log',
    #                                                 initial_learning_rate=initial_learning_rate,
    #                                                 learning_rate_drop=learning_rate_drop,
    #                                                 learning_rate_epochs=learning_rate_epochs,
    #                                                 learning_rate_patience=learning_rate_patience,
    #                                                 early_stopping_patience=early_stopping_patience))
