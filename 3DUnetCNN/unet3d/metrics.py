from functools import partial

from tensorflow.keras import backend as K
from keras import losses
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)



def weighted_dice_coefficient(y_true, y_pred, axis=(-3,-2,-1), smooth=1e-5):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    
    return K.mean(2. * (K.sum(y_true * y_pred, axis
                              ) + smooth)/(K.sum(y_true, axis
                                                            ) + K.sum(y_pred, axis
                                                                               ) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred,weight=1, index=0):
    
    print("got weight "+str(weight)+" from dataset "+str(index))
    return weight*(1-weighted_dice_coefficient(y_true, y_pred))

def get_wdl(y_true,y_pred,weight=[1,1,1],index=0):
    f = partial(weighted_dice_coefficient_loss,weight=weight, index=index)
    f.__setattr__('__name__', 'task_{0}_wdl'.format(index))
    return f

def label_wise_dice_coefficient(y_true, y_pred, label_index,weight=[1,1,1],index=0):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])

def get_label_dice_coefficient_function(label_index,weight=[1,1,1],index=0):
    f = partial(label_wise_dice_coefficient, label_index=label_index,weight=weight, index=index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f



def label_wise_weighted_dice_loss(y_true, y_pred, low, label_index,weights, task_index):
    return weighted_dice_coefficient_loss(y_true[:, low:label_index], y_pred[:, low:label_index])

def get_label_dice_loss(weight,index):
    f = partial(weighted_dice_coefficient_loss, weight=weight, index=index)
    f.__setattr__('__name__', 'label_{0}_dice_loss'.format(index))
    return f

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
