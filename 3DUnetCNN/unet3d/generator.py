import os
import copy
from random import shuffle
import itertools

import numpy as np
import tensorflow as tf
from functools import partial

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y

from progressbar import *

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    Rotate,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    GaussNoise  
)
# RandRotate90(p=0.5)
# aug = GaussNoise((0,0.0000001),p=1)
aug = Compose([
    OneOf([
    HorizontalFlip(p=0.6),        
    VerticalFlip(p=0.6),              
    Rotate(limit=25,p=0.6)],p=1),
    OneOf([
        ElasticTransform(p=0.6, alpha=25, sigma=30*0.05 , alpha_affine=30*0.03 ),
        GridDistortion(p=0.6,num_steps=5, distort_limit=0.5),
        OpticalDistortion(p=0.6, distort_limit=0.8, shift_limit=0.5)                  
        ], p=1)])#,GaussNoise((0.,0.01),p=0.4)])
        #,
    #Crop(p=0.7),    
    #RandomBrightnessContrast(p=0.9,limit=0.02)])   
    #RandomGamma(p=0.9)])

# from h5imagegenerator import HDF5ImageGenerator
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

all_transforms = get_train_transform((128,128,48))


def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)
    print("run data_generator on training_set...")

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,
                                        val=False)


    # training_generator = tf.data.Dataset.from_generator(partial(data_generator,data_file, training_list,
    #                                     batch_size=batch_size,
    #                                     n_labels=n_labels,
    #                                     labels=labels,
    #                                     augment=augment,
    #                                     augment_flip=augment_flip,
    #                                     augment_distortion_factor=augment_distortion_factor,
    #                                     patch_shape=patch_shape,
    #                                     patch_overlap=0,
    #                                     patch_start_offset=training_patch_start_offset,
    #                                     skip_blank=skip_blank,
    #                                     permute=permute), output_types=(tf.float32, tf.uint8),
    #                                     output_shapes=(tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None])))#,args=training_list)#(tf.TensorShape([None,None,None, None, None]), tf.TensorShape([None,None,None,None,None])))
    print("Succes")
    print("run data_generator on validation_set...")                                        
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          val=True)
    

    # validation_generator = tf.data.Dataset.from_generator(partial(data_generator,data_file, validation_list,
    #                                       batch_size=validation_batch_size,
    #                                       n_labels=n_labels,
    #                                       labels=labels,
    #                                       patch_shape=patch_shape,
    #                                       patch_overlap=validation_patch_overlap,
    #                                       skip_blank=skip_blank), output_types=(tf.float32, tf.uint8),
    #                                       output_shapes=(tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None])))#,args=validation_list)#(tf.TensorShape([None,None,None, None, None]), tf.TensorShape([None,None,None,None,None])))
    print("Succes")

    # Set the number of training and testing samples per epoch correctly
    print("Determine number of training steps...")
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                                                   skip_blank=skip_blank,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)

    print("Determine number of validation steps...")
    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)


    # training_generator = tf.data.Dataset.from_tensors(training_generator)
    # training_generator = tf.data.Dataset.TFRecordDataset(training_generator)
    # validation_generator  = tf.data.Dataset.from_tensors(validation_generator)
    # validation_generator = tf.data.Dataset.TFRecordDataset(validation_generator)


    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = global_split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        print("Successfully created valiadation split...")
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def global_split_list(input_list, split=0.8, shuffle_list=True): 
    # if shuffle_list:
    #     shuffle(input_list)
    nr_hippo = 145 #(only has 1 mod 260)
    nr_prost = 133 #(32: 1 mod | 64: 2 mods)
    nr_heart = 124 #(only has 1 mod 20)

    split_hippo = 120#int(split*nr_hippo)
    split_prost = 145+120#nr_hippo+int(split*nr_prost)
    split_heart = 145+133+120#nr_hippo+nr_prost+int(split*nr_heart)

    training = input_list[:split_hippo]+input_list[nr_hippo:split_prost]+input_list[(nr_hippo+nr_prost):split_heart]
    testing = input_list[split_hippo:nr_hippo]+input_list[split_prost:(nr_hippo+nr_prost)]+input_list[split_heart:]

    if len(testing)+len(training) == nr_hippo+nr_prost+nr_heart:
        print("Creating GLOBAL Train/Validation split succeeded!")

    else:
        print("Creating GLOBAL Train/Validation split failed! --> generator.py -> global_split_list")

    if shuffle_list:
        shuffle(training)
        shuffle(testing)

    return training, testing




def split_list(input_list, split=0.8, shuffle_list=True): 
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing



def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False, val=False, album=False):

    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        tl_list= list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            task_label = add_data(x_list, y_list,tl_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
                     skip_blank=skip_blank, permute=permute)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                x = np.asarray(x_list)
                y = np.asarray(y_list)
                tl_list = np.asarray(tl_list)
                x = x.astype(np.float32)
                y = y.astype(np.uint8)
                if not val:
                    if album:
                        augmented = aug(image=x[0][0], mask=y[0][0])
                        x_list[0][0] = augmented['image']
                        y_list[0][0] = augmented['mask']

                        # data_dict = {'data': x[0][0], 'seg': y[0][0]}
                        # data_dict = all_transforms(data_dict)
                        # x_list[0][0] = data_dict['data']
                        # y_list[0][0] = data_dicht['seg']
                ones = np.ones(y_list[0][0].shape)
                ones = np.array([ones])     
                x,y = convert_data(x_list, y_list, n_labels=n_labels, labels=labels)

                # y = [np.array([y[0][0:2]]),np.array([y[0][2:4]]),np.array([y[0][4:]])]
                # print(x.shape)
                # print(y.shape)       
                # print(tl_list.shape)
                # x1 = np.zeros(x.shape)
                # x2 = np.zeros(x.shape)
                # x3 = np.zeros(x.shape)
                # im_shape = x.shape[-3:]
                # shape =tuple([batch_size]+[3]+list(im_shape))
                # y12 = np.zeros(shape)
                # y34 = np.zeros(shape)
                # shape = tuple([batch_size]+[2]+list(im_shape))
                # y5 = np.zeros(shape)
                # for i in range(batch_size):
                #     y12[i] = np.vstack((ones,y[i][1:2],y[i][2:3]))
                #     y34[i] = np.vstack((ones,y[i][3:4],y[i][4:5]))
                #     y5[i] = np.vstack((ones,y[i][5:]))
                #     # print(y12.shape)
                #     task_label = tl_list[i]
                #     if task_label[0][0]:
                #         y12[i] = y[i][0]
                #     elif task_label[0][1]:
                #         y34[i] = y[i][0]
                #     elif task_label[0][2]:
                #         y5[i] = y[i][0]

                    
                #     x1[i] = task_label[0][0]*x[i]+1e-16
                #     x2[i] = task_label[0][1]*x[i]+1e-16
                #     x3[i] = task_label[0][2]*x[i]+1e-16



                y12 = np.vstack((ones,y[0][1:2],y[0][2:3]))
                y34 = np.vstack((ones,y[0][3:4],y[0][4:5]))
                y5 = np.vstack((ones,y[0][5:]))
                if task_label[0][0]:
                    y12[0] = y[0][0]
                elif task_label[0][1]:
                    y34[0] = y[0][0]
                elif task_label[0][2]:
                    y5[0] = y[0][0]

                w1 = task_label[0][0]
                w2 = task_label[0][1]
                w3 = task_label[0][2]

                x1 = task_label[0][0]*x+1e-16
                x2 = task_label[0][1]*x+1e-16
                x3 = task_label[0][2]*x+1e-16
                # x = [x1,x2,x3]
                y = [np.array([y12]),np.array([y34]),np.array([y5])]
                
                # y = [np.array([y12]),np.array([y34]),np.array([y5])]
                yield [x,np.array([w1]),np.array([w2]),np.array([w3])],y
                x_list = list()
                y_list = list()
                tl_list= list()


def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        i = 0
        pbar = ProgressBar(maxval=len(index_list))
        pbar.start()
        for index in index_list:
            pbar.update(i)
            i = i+1
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        pbar.finish()
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list,tl_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)

    if augment:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)
        

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)

    tl_list.append(data_file.root.label[index])
    return data_file.root.label[index]


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


