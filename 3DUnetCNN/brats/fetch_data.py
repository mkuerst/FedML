import os
import numpy as np
import json
import sys
import pathlib
from progressbar import *
import h5py
import nilearn
from random import shuffle

# lib to load nifti images
import nibabel as nib


# for printing the whole array
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)


# sizes from nnUnet paper - order: patch_size, batch_size, pool_size
sizes = {'Task01_BrainTumour': [(128,128,128),2,(5,5,5)], 'Task02_Heart': [(80,192,128),2,(4,5,5)], 'Task03_Liver': [(128,128,128),2,(5,5,5)],
         'Task04_Hippocampus': [(40,56,40),9,(3,3,3)], 'Task05_Prostate': [(20,192,192),4,(2,5,5)], 'Task06_Lung': [(112,128,128),2,(4,5,5)], 
         'Task07_Pancreas' : [(96,160,128),2,(4,5,5)]}

# path to datasets
datasets_path = '/local/home/kumichae/3DUnet/datasets'

def prepare_data(task_name,balance=False):
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
                #print(img.get_fdata()[:,:,:,j].shape)
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
            tr_img_dirs.append((str_tr, str_la))
            i = i+1
    
    pbar.finish()

    # configure params for 3dunet
    nb_channels = 1

    labels = tuple(list(map(int,(list(data['labels'].keys()))))[1:])
    print("labels :", labels)
    n_labels = len(labels)

    shuffle(tr_img_dirs)

    if balance:
        tr_length = int(len(tr_img_dirs)*0.8)
        if task_name == 'Task04_Hippocampus':
            tr_img_dirs = tr_img_dirs[:120]+tr_img_dirs[120:145] # 120 tr | 25 val
        elif task_name == 'Task05_Prostate':
            tr_img_dirs = (tr_img_dirs[:51]*3)[:120]+tr_img_dirs[51:] # 51 tr | 13 val 
        elif task_name == 'Task02_Heart':
            tr_img_dirs = (tr_img_dirs[:16]*8)[:120]+tr_img_dirs[16:] # 16 tr_images | 4 val

    #shuffle(tr_img_dirs)
    return tr_img_dirs, labels, n_labels








# GET DIECTORIES FOR GLOBAL MODEL
def prepare_global_data(task_names,index,balance=False):
    global_labels = []
    global_n_labels = 0
    global_tr_dirs = []

    j = 0
    for task_name in task_names:
        tr_tuples, labels, n_labels = prepare_data(task_name,balance)
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
    global_labels.insert(0,0)

    
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



def main():
    y = prepare_global_data(['Task04_Hippocampus'])

if __name__ == "__main__":
    main()

# Notes:
# GLOBAL LABELS: {1: Anterior, 2: Posterior (Hippocampus)| 3: PZ, 4: TZ (Prostate)| 5: left atrium (Heart)}

# -- display directory sizes of datasets @ /local/home/kumichae/3DUnet/datasets --
# du -bsh *

# -- BEWARE NIFTI IMAGES HAVE REVERSE ODER OF DIMENSIONS -- (nchannels,x,y,z)

# -- Start benchmarking on datasets Hippocampus (27 MB) -> Prostate (229 MB) -> Heart (435 MB) -> Spleen (1 GB) --


#-- conda install pygpu -- make sure to have pygpu installed

#-- pip install progressbar2 -- 

# -- pip install ablumentations -- 

# -- create and activate environment --
# conda env create -f environment.yml --force
# conda env update -f environment.yml --prune
# source/conda activate 3dunet -- environment.yml dependencies
# source deactivate
# -- deactivate environment -- 

# -- gpustat -i -- gpu stats

# -- COPY STUFF AROUND -- FROM --> TO
# rsync -avz -e 'ssh' kumichae@isegpu2.inf.ethz.ch:/local/home/kumichae/3DUnet/3DUnetCNN/brats/GeneratorImages /home/root1/CODE/3DUnetCNN/brats

# -- tensorboard --logdir /local/home/kumichae/3DUnet/3DUnetCNN/brats -- for tensorboard visuals

# tmux new -s id
# tmux detach-client -s id
# tmux attach -t id
# tmux  kill-session -t id







# IN tensorflow_backend.py line 505 ->
        # else:
        #     #_LOCAL_DEVICES = tf.config.experimental_list_devices()
        #     devices = tf.config.list_logical_devices()
        #     _LOCAL_DEVICES = [x.name for x in devices]

# IN callbacks.py line 1531 ->
#    self._log_write_dir = distributed_file_utils.write_dirpath(
#         self.log_dir)#, self.model._get_distribution_strategy())  # pylint: disable=protected-access

#pip install git+https://www.github.com/keras-team/keras-contrib.git