import os
import numpy as np
import json
import sys
import pathlib
from progressbar import *
import h5py
import nilearn

# lib to load nifti images
import nibabel as nib

import SimpleITK as sitk
import numpy as np


 

# for printing the whole array
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

# path to datasets
datasets_path = '/local/home/kumichae/3DUnet/datasets'

def prepare_data(task_name):
    print('Preparing task: '+task_name)

    #get data from json file
    json_path = os.path.join(datasets_path,task_name+'/dataset.json')
    json_file = open(json_path, 'r')
    data= json.load(json_file)
    tr_data = data['training']
    modality = data['modality'].values()

    print('Modalities: '+str(modality))
    print('Labels: '+str(data['labels']))

    # set up progression bar
    pbar = ProgressBar(maxval=len(tr_data))
    pbar.start()

    # path to task dataset
    path_dataset = os.path.join(datasets_path, task_name)
    tr_img_dirs = []

    # build list of tuples of paths to all train_images/train_labels
    # 3D data needs to be split into slices
    if(len(modality) > 1): 
        i = 0
        #modality = ['mod'] # only read first modality for faster training
        for row in tr_data:
            pbar.update(i)
            str_tr = os.path.join(path_dataset, row['image'].replace("./", ""))
            str_la = os.path.join(path_dataset,row['label'].replace("./", ""))
            img = nib.load(str_tr)
            seperated_file_dirs = []
            j = 0
            for mod in modality:
                seperated_file_dirs = []
                str_mod = str_tr.replace(".nii.gz","_"+str(j)+".nii.gz")
                seperated_file_dirs.append(str_mod)
                seperated_file_dirs.append(str_la)
                seperated_file_dirs = tuple(seperated_file_dirs)

                tr_img_dirs.append(seperated_file_dirs)
                if not os.path.exists(str_mod):
                    img_ = nib.Nifti1Image(img.get_fdata()[:,:,:,j], img.affine, img.header)
                    img_.header['dim'][0] = 3
                    nib.save(img_, str_mod)
                j = j+1

            i = i+1
        
    # Has only one modality --> no need to split data into slices
    else:
        i = 0
        for row in tr_data:
            pbar.update(i)
            str_tr = os.path.join(path_dataset, row['image'].replace("./", ""))
            str_la = os.path.join(path_dataset,row['label'].replace("./", ""))
            #img = nib.load(str_tr)
            #print (img.header)
            tr_img_dirs.append((str_tr, str_la))
            i = i+1
    
    pbar.finish()

    # configure params for 3dunet
    nb_channels = 1#len(modality)

    labels = tuple(list(map(int,(list(data['labels'].keys()))))[1:])
    print("labels :", labels)
    n_labels = len(labels)

    # patch shape needs to be reversed from nnUnet paper
    pool_size = (2,2,2)
    batch_size = 1
    
    image_shape = None
    patch_shape = None

#config params:         all_mod  |tr_mod  | image_shape  | labels | n_labels
    return tr_img_dirs, modality, modality, image_shape, labels, n_labels,\
           nb_channels, patch_shape,              nb_channels, batch_size, pool_size
          #nb_channels |patch_shape |input_shape |truth_channel|batch_size|pool_size
    
# GET DIECTORIES FOR GLOBAL MODEL
def prepare_global_data(task_names,index):
    global_labels = []
    global_n_labels = 0
    global_tr_dirs = []

    j = 0
    for task_name in task_names:
        tr_tuples, _, _ , _, labels, n_labels, _, _, _, _, _ = prepare_data(task_name)
        label_dirs = [lab[-1] for lab in tr_tuples]
        labels = list(labels)
        labels = [label+len(global_labels) for label in labels]
        global_labels = global_labels + labels

        global_label_dirs = prepare_global_labels(label_dirs,labels)

        if j==index or index == None:
            for i in range(len(tr_tuples)):
                inter = list(tr_tuples[i])
                inter[-1] = global_label_dirs[i]
                inter = tuple(inter)
                global_tr_dirs.append(inter)
        j = j+1

    #print(global_tr_dirs)
    n_global_labels = len(global_labels)
    global_labels = tuple(global_labels)
    print('global_labels: '+str(global_labels))

    print("saving "+global_tr_dirs[15][-1] +" into GeneratorImages")
    gl_truth_img = nib.load(global_tr_dirs[15][-1])
    nib.save(gl_truth_img,os.path.join(os.getcwd(),"GeneratorImages/gl_truth_img.nii"))

    return global_tr_dirs, global_labels, n_global_labels


# CHANGE LABELS TO GLOBAL ORDER
def prepare_global_labels(label_dirs,labels):
    new_label_dirs = []
    print('Changing the labels in the ground truths')
    pbar = ProgressBar(maxval=len(label_dirs))
    pbar.start()
    j = 0

    for label_dir in label_dirs:
        j = j+1
        pbar.update(j)
        
        labeled_img = nib.load(label_dir)
        global_label_dir = label_dir.replace(".nii.gz","_"+'global'+".nii.gz")
        new_label_dirs.append(global_label_dir)
        
        if not os.path.exists(global_label_dir):
            data = labeled_img.get_fdata()
            i = 1
            for label in labels:
                new_labeled_img = np.where(data==i, label, data)
                data = new_labeled_img
                i = i+1
        
            img_ = nib.Nifti1Image(data, labeled_img.affine)#,labelled_img.header)  #labelled_img.affine
            nib.save(img_, global_label_dir)
    
    pbar.finish()
    #print(new_label_dirs)
    return new_label_dirs




def threshold_based_crop(image):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.                                 
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])


def resample_img(data,reference_size=[128,128,48],isLabel=False):
    dimension = data[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for img in data:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = [128]*dimension # Arbitrary sizes, smallest size that yields desired results. 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
    # spacing will not yield the correct coordinates resulting in a long debugging session. 
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    resampled_images = []
    for img in data:
        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)
        # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth 
        # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that 
        # no new labels are introduced.
        if not isLabel:
            resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
        else:
            resampled_img = sitk.Resample(img, reference_image, centered_transform, sitk.sitkNearestNeighbor, 0.0)

        resampled_images.append(resampled_img)

    return resampled_images

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

transforms = get_train_transform((128,128,48))





def main():
    tr_img_dirs, _ , _ = prepare_data('Task02_Heart') 

    for i in range(3):
        img = nib.load(tr_img_dirs[i][0])
        data = img.get_fdata()
        transforms = get_train_transform(data.shape)

        data_dict['data'] = data
        data_dict = transforms(data_dict)

        augmented = data_dict["data"]

        img_ = nib.Nifti1Image(augmented, img.affine, img.header)
        nib.save(img_, "BG_IMG_augmented")
        print("saved augmented IMG !!!!!!!!!!!")

if __name__ == "__main__":
    main()
