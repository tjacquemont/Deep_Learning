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
from PyTorch_models.Utils.Classification import PyT_dl_utils
from PyTorch_models.Models import PyT_dl_models_classic_CNN, PyT_dl_models_CNN_ResNet
import torch
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
parser.add_argument('--epochs', type=int, default=5,
                    help='Nombre d\'époques d\'entraînement.')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Taux de dropout pour le modèle.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Taille du batch pour l\'entraînement.')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Taux d\'apprentissage.')
parser.add_argument('--regularizers_l2', type=float, default=0.001, # <--- NOUVELLE VARIABLE AJOUTÉE
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
    "ARAT_bin": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_arat_binaire.pkl",
    "FM_bin": "/network/iss/cenir/analyse/irm/users/thomas.jacquemont//PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fm_binaire.pkl",
    "FMprop_bin" : "/network/iss/cenir/analyse/irm/users/thomas.jacquemont//PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fmprop_binaire.pkl"
}

input_channels = 1  
deep_learning_models = {
    "ClassicCNNModel_5Conv_NoAveragePool_withDropout" : PyT_dl_models_classic_CNN.ClassicCNNModel_5Conv_NoAveragePool_withDropout,
    "ClassicCNNModel_5Conv_NoAveragePool_withDropout_ReluBeforBatchNorm" : PyT_dl_models_classic_CNN.ClassicCNNModel_5Conv_NoAveragePool_withDropout_ReluBeforBatchNorm,
    "ClassicCNNModel_5Conv_NoAveragePool_NoDropout" : PyT_dl_models_classic_CNN.ClassicCNNModel_5Conv_NoAveragePool_NoDropout,
    "CNNResNetModel_3ResBlock_WithDropOut" : PyT_dl_models_CNN_ResNet.CNNResNetModel_3ResBlock_WithDropOut,
    "CNNResNetModel_5ResBlock_WithDropOut" : PyT_dl_models_CNN_ResNet.CNNResNetModel_5ResBlock_WithDropOut
    }

############################### INPUTS ########################################

working_directory = os.path.join(working_directory_base, motor_score)
resume_csv_path = os.path.join(working_directory, 'Resume_DeepLearrning_Performances.csv')
print(resume_csv_path)

# Model instanciation 
Cnn_Model = deep_learning_models[model_name]

model_working_directory = working_directory + '/' + f'{model_name}_{motor_score}_data_augmentation_{data_augmentation}_batch_size_{batch_size}_epochs_{epochs}_earlystop_{early_stop}_learning_rate_{learning_rate}_regularizers_l2_{regularizers_l2}_dropout_{dropout_rate}'
if not os.path.exists(model_working_directory):
    os.makedirs(model_working_directory)
    print(f"Created model working directory: {model_working_directory}")
 

classic_cnn_model_FM_data_augmentation_false = PyT_dl_utils.training_pipeline(
    pickle_path=datasets[motor_score],
    model_function=lambda input_channels, dropout_rate: Cnn_Model(input_channels=input_channels, dropout=dropout_rate),
    model_name=model_name, # Only used for visualizer 
    working_dir=model_working_directory,
    performance_csv_path=resume_csv_path,
    data_augmentation=data_augmentation,
    epochs=epochs,
    batch_size=batch_size,
    dropout=dropout_rate,
    learning_rate=learning_rate,
    regularizers_l2=regularizers_l2,
    early_stop=early_stop,
    device=device
)
