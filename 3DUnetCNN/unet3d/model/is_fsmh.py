from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Dense, Flatten, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import keras

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss
from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient, get_label_dice_loss, get_wdl


import keras
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)




# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)
def is_fsmh_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    ins1 = Input(input_shape)
    ins2 = Input(input_shape)
    ins3 = Input(input_shape)
    #weights = Input((input_shape[0],3,))


    output_layer12 = create_branch(ins1,3)
    output_layer34 = create_branch(ins2,3)
    output_layer5 = create_branch(ins3,2)

    activation12 = Activation(activation_name,name='out12')(output_layer12)
    activation34 = Activation(activation_name,name='out34')(output_layer34)
    activation5 = Activation(activation_name,name='out5')(output_layer5)


    outputs = [activation12, activation34, activation5]

    # label_wise_dice_metrics = {'out12': [get_label_dice_coefficient_function(0),get_label_dice_coefficient_function(1), get_label_dice_coefficient_function(2)], \
    #                             'out34': [get_label_dice_coefficient_function(0),get_label_dice_coefficient_function(1), get_label_dice_coefficient_function(2)], \
    #                                 'out5': [get_label_dice_coefficient_function(0), get_label_dice_coefficient_function(1)]}
    label_wise_dice_metrics = [[get_label_dice_coefficient_function(0),get_label_dice_coefficient_function(1), get_label_dice_coefficient_function(2)], \
                                [get_label_dice_coefficient_function(0),get_label_dice_coefficient_function(1), get_label_dice_coefficient_function(2)], \
                                [get_label_dice_coefficient_function(0), get_label_dice_coefficient_function(1)]]
    
    # loss = [weighted_dice_coefficient_loss]
    # loss = {'out12': weighted_dice_coefficient_loss, 'out34': weighted_dice_coefficient_loss, 'out5': weighted_dice_coefficient_loss }
    loss = [weighted_dice_coefficient_loss, weighted_dice_coefficient_loss,weighted_dice_coefficient_loss]
    model = Model(inputs=[ins1,ins2,ins3], outputs=outputs) 
    #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss, metrics=label_wise_dice_metrics)

    return model, optimizer(lr=initial_learning_rate), loss, label_wise_dice_metrics


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



def create_branch(inputs,n_labels,dropout_rate=0.3,depth=5,n_base_filters=16,n_segmentation_levels=3):
    current_layer = inputs
    # This list will contain each summed feature and context extraction for each level
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
    

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    return output_layer