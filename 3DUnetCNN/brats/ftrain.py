import os
import glob
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
import h5py
import collections

import fetch_data

from tensorflow.keras import losses, metrics, optimizers
from keras import backend as K
K.common.set_image_dim_ordering('th')

#NEEDED TO USE MULTIPLE GPUS
from keras.utils import multi_gpu_model 

# needed to be able to import the unet3d module
sys.path.append('/local/home/kumichae/3DUnet/3DUnetCNN')
sys.path.append('/local/home/kumichae/3DUnet')

from unet3d.model import unet_model_3d, isensee2017_model, isensee_global_model, isensee_multiheads_model, isensee_soft_mh_model, is_fsmh_model
from unet3d.training import load_old_model, train_model
from unet3d.generator import get_training_and_validation_generators, global_split_list, convert_data
from unet3d.data import write_data_to_file, open_data_file

from tff_virtual import *
from tff_virtual.source.fed_prox import FedProx
from tff_virtual.source.fed_process import FedProcess
# DEFINE VISIBLE GPUS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.compat.v1.disable_eager_execution()

# get rid of logging warning etc.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
def conv(data,truth,task_label, n_labels=config['n_labels'], labels=config['labels']):
    ones = np.ones(truth.shape[-3:])
    ones = np.array([ones])   
    zeros = np.array([np.zeros(truth.shape[-3:])])
    x,y = convert_data(data, truth, n_labels=config['n_labels'], labels=config['labels'])

    y12 = np.vstack((ones,y[0][1:2],y[0][2:3]))
    y34 = np.vstack((ones,y[0][3:4],y[0][4:5]))
    y5 = np.vstack((ones,y[0][5:],zeros))
    if task_label[0][0]:
        y12[0] = y[0][0]
    elif task_label[0][1]:
            y34[0] = y[0][0]
    elif task_label[0][2]:
        y5[0] = y[0][0]

    x1 = tf.convert_to_tensor(task_label[0][0]*x+1e-16, dtype=tf.float32)
    x2 = tf.convert_to_tensor(task_label[0][1]*x+1e-16, dtype=tf.float32)
    x3 = tf.convert_to_tensor(task_label[0][2]*x+1e-16, dtype=tf.float32)
    #print(x1.shape)
    y12 = tf.convert_to_tensor(np.array([y12]),dtype=tf.int8)
    y34 = tf.convert_to_tensor(np.array([y34]),dtype=tf.int8)
    y5 = tf.convert_to_tensor(np.array([y5]),dtype=tf.int8)
    # print(x1.shape)
    # print(y12.shape)
    # y = [np.array([y12]),np.array([y34]),np.array([y5])]  
    y = [y12,y34,y5]
    # y = np.vstack((y12,y34,y5))
    # y = tf.convert_to_tensor(y, dtype= tf.int8)
    # print(y.shape)
    return [x1,x2,x3], y

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

    # SMH FEDERATED MODEL
    task_gl = 'global(64,64,48)_is_fsmh_bal'
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
        #with tf.device("/cpu:0"):
        model, opt, loss,ldcm = is_fsmh_model(input_shape=config['input_shape'],n_labels=config['n_labels'])
        #model = multi_gpu_model(model,gpus=2)    
        # model.compile(optimizer=opt,loss=loss,metrics=ldcm)



    nr_hippo = 120
    nr_prost = 120
    nr_heart = 120
    total = nr_hippo+nr_prost+nr_heart
    split = 10

    index_list = list(range(402))
    tr_list, val_list = global_split_list(index_list)
    images_per_client = int(total/split)
    client_train_dataset = collections.OrderedDict()
    client_val_dataset = collections.OrderedDict()
    tr_client_ids = []
    val_client_ids = []

    # TRAIN DATA
    for i in range(1, split+1):
        client_name = "client_" + str(i)
        tr_client_ids.append(client_name)
        start = images_per_client * (i-1)
        end = images_per_client * i
        # start = tr_list[start]
        # end = tr_list[end] 
        data = [data_file_opened.root.data[index] for index in tr_list[start:end]]
        tl = [data_file_opened.root.label[index] for index in tr_list[start:end]]
        truth = [data_file_opened.root.truth[index] for index in tr_list[start:end]]
        data_l = []
        truth_l = []
        for j in range(len(data)):
            x,y = conv(np.array([data[j]]),np.array([truth[j]]),tl[j])
            data_l.append(x)
            truth_l.append(y)

        # print(len(data_l))
        # print(len(truth_l))
        print(f"Adding data from {start} to {end} for client : {client_name}")
        client_data = collections.OrderedDict((('data', data_l), ('truth', truth_l)))#, ('task_label', tl)))
        client_train_dataset[client_name] = client_data

    # VAL DATA    
    for i in range(len(val_list)):
        client_name = "client_" + str(i)
        val_client_ids.append(client_name)
        index = val_list[i]

        data = data_file_opened.root.truth[index]
        tl = data_file_opened.root.label[index]
        truth = data_file_opened.root.truth[index]

        data, truth = conv(np.array([data]),np.array([truth]),tl)
        # print(tf.shape(data))
        # print(tf.shape(truth))
        print(f"Adding data {index} for client: {client_name} as validation")
        client_data = collections.OrderedDict((('data', data), ('truth', truth)))#, ('task_label', tl)))
        client_val_dataset[client_name] = client_data
    
    
    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    val_dataset = tff.simulation.FromTensorSlicesClientData(client_val_dataset)

    SHUFFLE_BUFFER = images_per_client
    BATCH_SIZE = 1
    PREFETCH_BUFFER = 1
    NUM_EPOCHS = 100
    
    def preprocess(dataset):

        def batch_format_fn(element):
            data = tf.reshape(element['data'],[3,64,64,48])
            truth = tf.reshape(element['truth'],[3,3,64,64,48])
            print(data.shape)
            print(truth.shape)
            #task_label = element['task_label']
            #data = [data[0],data[1],data[2]]
            return collections.OrderedDict(
                x=data,
                y=truth)

        return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
            BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


    def make_federated_data(client_data, client_ids):
        # for x in client_ids:
        #     print(tf.shape(client_data.create_tf_dataset_for_client(x)))
        return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    
    sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
    preprocessed_sample_dataset = preprocess(sample_dataset)

    federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)
    federated_val_data = make_federated_data(val_dataset, val_dataset.client_ids)

    # print(tf.shape(federated_train_data))
    print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
  
  
  
    from tensorflow import reshape, nest
  
    #sample_batch = nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))
  
  
    # def model_fn():
    #     return tff.learning.from_keras_model(
    #         model,
    #         input_spec=preprocessed_sample_dataset.element_spec,
    #         #dummy_batch = preprocessed_sample_dataset,
    #         loss=loss,
    #         metrics=ldcm)
      

    # iterative_process = tff.learning.build_federated_averaging_process(
    #     model_fn,
    #     client_optimizer_fn= lambda: opt,
    #     server_optimizer_fn= lambda: optimizers.SGD(learning_rate=5e-4))

    # print(str(iterative_process.initialize.type_signature))

    # state = iterative_process.initialize()

    # tff_train_acc = []
    # tff_val_acc = []
    # tff_train_loss = []
    # tff_val_loss = []

    # eval_model = None
    # for round_num in range(1, 10+1):
    #     state, tff_metrics = iterative_process.next(state, federated_train_data)
    # #     eval_model = create_keras_model()
    # #     eval_model.compile(optimizer=optimizers.Adam(learning_rate=client_lr),
    # #                     loss=losses.SparseCategoricalCrossentropy(),
    # #                     metrics=[metrics.SparseCategoricalAccuracy()])

    # #     tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    # #     ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
    # #     print('round {:2d}, metrics={}'.format(round_num, tff_metrics))
    # #     print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")
    # #     tff_train_acc.append(float(tff_metrics.sparse_categorical_accuracy))
    # #     tff_val_acc.append(ev_result[1])
    # #     tff_train_loss.append(float(tff_metrics.loss))
    # #     tff_val_loss.append(ev_result[0])

    # # metric_collection = {"sparse_categorical_accuracy": tff_train_acc,
    # #                     "val_sparse_categorical_accuracy": tff_val_acc,
    # #                     "loss": tff_train_loss,
    # #                     "val_loss": tff_val_loss}
    
  
  
  
  
  
  
  
  
  
  
  
  
  
    from tff_virtual.source.federated_devices import ClientSequential, ServerSequential, ClientModel, ServerModel
  
    cmodel = ClientModel(model)
  
  
  
  
    def compile_model(opt, loss, metrics):
        cmodel.compile(optimizer=opt,
                    loss=loss,
                    metrics=ldcm)
        return cmodel

    def model_fn(model, train_size):
        # model, opt, loss,ldcm = is_fsmh_model(input_shape=config['input_shape'],n_labels=config['n_labels'])
        # return tff.learning.from_keras_model(
        #     model,
        #     input_spec=preprocessed_sample_dataset.element_spec,
        #     loss=loss,
        #     metrics=ldcm)   
        return compile_model(opt,loss,ldcm)
    

    # BATCH_SIZE = 1
    # SHUFFLE_BUFFER = 500
    LEARNING_RATE = 5e-4
    NUM_CLIENTS = 10
    EPOCHS_PER_ROUND = 1
    NUM_ROUNDS = 100
    CLIENTS_PER_ROUND = 5

    train_size = [np.array([images_per_client])for data in client_train_dataset]
    test_size = [np.array([1]) for data in client_val_dataset]
    logdir = './FedLogs'

    fed_process = FedProcess(model_fn, NUM_CLIENTS)
    fed_prox = FedProx(model_fn, NUM_CLIENTS)
    fed_prox.fit(federated_train_data,
                        num_rounds=NUM_ROUNDS,
                        clients_per_round=CLIENTS_PER_ROUND,
                        epochs_per_round=EPOCHS_PER_ROUND,
                        train_size=train_size,
                        test_size=test_size,
                        logdir=logdir,
                        federated_test_data=federated_val_data)














    # iterative_process = tff.learning.build_federated_averaging_process(
    #     model_fn(preprocessed_sample_dataset,model),
    #     client_optimizer_fn=lambda: opt,
    #     server_optimizer_fn=lambda: optimizers.SGD(learning_rate=5e-4))

    # state = iterative_process.initialize()

    # tff_train_acc = []
    # tff_val_acc = []
    # tff_train_loss = []
    # tff_val_loss = []



    # print(data_file_opened.root.label.shape)
    # print(data_file_opened.root.label[0][0])
    # print(data_file_opened.root.label[nr_hippo][0])
    # print(data_file_opened.root.label[nr_hippo+nr_prost][0])

   

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
    # evaluation = tff.learning.build_federated_evaluation(model_fn(preprocessed_sample_dataset))
    # NUM_ROUNDS = 10
    # eval_model = None
    # for round_num in range(1, NUM_ROUNDS+1):
    #     state, tff_metrics = iterative_process.next(state, federated_train_data)
    #     # eval_model, opt, loss,ldcm = isensee_soft_mh_model(input_shape=config['input_shape'],n_labels=config['n_labels'])
    #     # eval_model.compile(optimizer=opt,
    #     #                 loss=loss,
    #     #                 metrics=ldcm)

    #     # tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    #     # ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
    #     # print('round {:2d}, metrics={}'.format(round_num, tff_metrics))
    #     # print(f"Eval wdl : {ev_result[0]} and Eval dc : {ev_result[1]}")
    #     # tff_train_acc.append(float(tff_metrics.sparse_categorical_accuracy))
    #     # tff_val_acc.append(ev_result[1])
    #     # tff_train_loss.append(float(tff_metrics.loss))
    #     # tff_val_loss.append(ev_result[0])
    #     val_metrics = evaluation(state.model, federated_val_data)
    # metric_collection = {"sparse_categorical_accuracy": tff_train_acc,
    #                     "val_sparse_categorical_accuracy": tff_val_acc,
    #                     "loss": tff_train_loss,
    #                     "val_loss": tff_val_loss}



    data_file_opened.close()


if __name__ == "__main__":
    main(['Task04_Hippocampus','Task05_Prostate','Task02_Heart'], overwrite=config["overwrite"])



