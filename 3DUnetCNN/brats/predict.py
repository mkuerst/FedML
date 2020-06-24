import os
import sys
import tensorflow as tf

from train import config
from unet3d.prediction import run_validation_cases, predict_from_data_file_and_write_image

import tables
sys.path.append('/local/home/kumichae/3DUnet/3DUnetCNN')
from unet3d.utils import pickle_load
from unet3d.training import load_old_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 'GlobalModel/global(144,144,48)_mdl1e_fs_1mod'
tf.compat.v1.disable_eager_execution()
from keras.models import load_model

task_gl = 'global(64,64,48)_is_decmh_bal'

def main():
    config['training_modalities'] = ['MRI']
    config['labels'] = (1,2,3,4,5)
        
    config["data_file"] = os.path.abspath(task_gl+".h5")
    config["model_file"] = os.path.abspath(task_gl+"_model.h5")
    config["training_file"] = os.path.abspath(task_gl+"_training_ids.pkl")
    config["validation_file"] = os.path.abspath(task_gl+"_validation_ids.pkl")



    prediction_dir = os.path.abspath("prediction")
    data_file = tables.open_file(config["data_file"], "r")


    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=False,
                         output_dir=prediction_dir
                         )

    data_file.close()

if __name__ == "__main__":
    main()
