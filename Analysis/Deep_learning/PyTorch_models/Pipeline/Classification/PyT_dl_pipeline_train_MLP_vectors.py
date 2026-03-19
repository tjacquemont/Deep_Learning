#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:35:03 2025

@author: thomas.jacquemont
"""

import sys
list_of_package_directory = [
    '/network/iss/home/thomas.jacquemont/PREP_AVC/Script/Deep_Learning'
]

for directory_path in list_of_package_directory:
    sys.path.append(directory_path)
import os
from PyTorch_models.Utils.Classification import PyT_dl_utils_vectors_only
from PyTorch_models.Models import PyT_dl_models_MLP_vectors
import torch
import torch.nn as nn
import argparse


# Définition des parsers d'arguments
parser = argparse.ArgumentParser(description='Script d\'entraînement d\'un modèle CNN pour la classification Deep Learning.')

# Paramètres à passer en ligne de commande
parser.add_argument('--motor_score', type=str, default="FM",
                    help='Score moteur à utiliser (e.g., "ARAT_bin", "FM_bin", "FMprop_bin").')
parser.add_argument('--data_augmentation', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Activer l\'augmentation de données (True/False).')
parser.add_argument('--early_stop', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Activer l\'arrêt anticipé (True/False).')
parser.add_argument('--patience', type=float, default=None,
                    help='Patience de l\'arrêt anticipé.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Nombre d\'époques d\'entraînement.')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Taux de dropout pour le modèle.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Taille du batch pour l\'entraînement.')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Taux d\'apprentissage.')
parser.add_argument('--regularizers_l2', type=float, default=0.000001, 
                    help='Poids de régularisation L2.')
parser.add_argument('--model_name', type=str, default="ClassicCNNModel_5Conv_NoAveragePool_withDropout",
                    help='Nom du modèle CNN à utiliser.')
parser.add_argument('--working_directory_base', type=str,
                    default='/network/iss/home/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/',
                    help='Répertoire de travail de base pour les sorties du modèle. Le score moteur sera ajouté.')


# Parser les arguments ---
args = parser.parse_args()

# Assigner les arguments parsés aux variables du script ---
motor_score = args.motor_score
data_augmentation = args.data_augmentation
early_stop = args.early_stop
patience = args.patience
epochs = args.epochs
dropout_rate = args.dropout_rate
batch_size = args.batch_size
learning_rate = args.learning_rate
regularizers_l2 = args.regularizers_l2
model_name = args.model_name
working_directory_base = args.working_directory_base 

# Détecter si CUDA est disponible (ne change pas, car c'est une détection système)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

###############################################################################

datasets = {
    "ARAT_bin_J3": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_arat_binaire.pkl",
    "FM_bin_J3": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fm_binaire.pkl",
    "FMprop_bin_J3" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fmprop_binaire.pkl",
    "ARAT_quat_J3": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_arat_quaternaire.pkl",
    "FM_quat_J3": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fm_quaternaire.pkl",
    "FMprop_quat_J3" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fmprop_quaternaire.pkl",
    "ARAT_bin_J7": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_arat_binaire.pkl",
    "FM_bin_J7": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fm_binaire.pkl",
    "FMprop_bin_J7" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fmprop_binaire.pkl",
    "ARAT_quat_J7": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_arat_quaternaire.pkl",
    "FM_quat_J7": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fm_quaternaire.pkl",
    "FMprop_quat_J7" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fmprop_quaternaire.pkl"    
    }

deep_learning_models = {
    "Sequential_3FC_2Dropout" : PyT_dl_models_MLP_vectors.Sequential_3FC_2Dropout,
    "Sequential_3FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_3FC_End_Dropout,
    "Sequential_4FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_4FC_End_Dropout,
    "Sequential_5FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_5FC_End_Dropout,
    "Sequential_6FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_6FC_End_Dropout,
    "Sequential_3FC_with_BatchNorm_Mid_Dropout" : PyT_dl_models_MLP_vectors.Sequential_3FC_with_BatchNorm_Mid_Dropout,
    "Sequential_4FC_with_BatchNorm_Mid_Dropout" : PyT_dl_models_MLP_vectors.Sequential_4FC_with_BatchNorm_Mid_Dropout,
    "Sequential_5FC_with_BatchNorm_Mid_Dropout" : PyT_dl_models_MLP_vectors.Sequential_5FC_with_BatchNorm_Mid_Dropout
    }

# determining the number of modele outputs based on the type of motor score categorisation
if "bin" in motor_score:
    num_classes = 1
elif "quat" in motor_score:
    num_classes = 4
else: 
    num_classes = 1

############################### INPUTS ########################################

working_directory = os.path.join(working_directory_base, motor_score)
resume_csv_path = os.path.join(working_directory, 'Resume_DeepLearrning_Performances.csv')
print(resume_csv_path)

is_multiclass = num_classes > 2
loss = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()

metric_list=['accuracy']

# Model instanciation 
Cnn_Model = deep_learning_models[model_name]

model_working_directory = working_directory + '/' + f'{model_name}_{motor_score}_data_augmentation_{data_augmentation}_batch_size_{batch_size}_epochs_{epochs}_earlystop_{early_stop}_{patience}_learning_rate_{learning_rate}_regularizers_l2_{regularizers_l2}_dropout_{dropout_rate}'
if not os.path.exists(model_working_directory):
    os.makedirs(model_working_directory)
    print(f"Created model working directory: {model_working_directory}")
 

MLP_model_trained = PyT_dl_utils_vectors_only.training_pipeline(
    pickle_path=datasets[motor_score],
    model_function=lambda vector_features_dim,dropout_rate, num_classes: Cnn_Model(vector_features_dim, dropout=dropout_rate, num_classes=num_classes),
    model_name=model_name, # Only used for visualizer 
    num_classes=num_classes,
    working_dir=model_working_directory,
    performance_csv_path=resume_csv_path,
    data_augmentation=data_augmentation,
    epochs=epochs,
    batch_size=batch_size,
    dropout=dropout_rate,
    learning_rate=learning_rate,
    regularizers_l2=regularizers_l2,
    early_stop=early_stop,
    patience=patience,
    device=device
)
