#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:33:07 2025

@author: thomas.jacquemont
"""

list_of_package_directory = ['/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Scripts/Analysis/Deep_learning/Models', '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Scripts/Analysis/Deep_learning/Utils']

import sys
import os
for directory_path in list_of_package_directory:
    sys.path.append(directory_path)
import dl_utils
import dl_models_classic_CNN

###############################################################################
datasets = {
"ARAT" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_nomalisee_arat_binaire.pkl",
"FM" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont//PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_nomalisee_fm_binaire.pkl"
}
working_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Cluster_working_directory/TensorFlow'
motor_score = "FM"
data_augmentation = False
epochs = 150

classic_cnn_model_FM_data_augmentation_false = dl_utils.training_pipeline(datasets[motor_score], dl_models_classic_CNN.classic_cnn_model, working_directory, data_augmentation=data_augmentation, epochs=epochs)
# classic_cnn_resnet_model_FM_data_augmentation_false = dl_utils.training_pipeline(datasets[motor_score], dl_models_classic_CNN.cnn_resnet_model, working_directory, data_augmentation=data_augmentation, epochs=epochs)