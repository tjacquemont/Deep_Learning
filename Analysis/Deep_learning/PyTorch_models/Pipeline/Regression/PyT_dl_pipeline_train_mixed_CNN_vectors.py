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
from PyTorch_models.Utils.Regression import PyT_dl_utils_mixte_models
from PyTorch_models.Models import PyT_dl_models_mixed_CNN_vector, PyT_dl_models_classic_CNN, PyT_dl_models_MLP_vectors
import torch
import torch.nn as nn
import json
import argparse

def parse_filters(filters_json):
    if not filters_json:  # gère None et chaîne vide
        return None
    
    try:
        filters_dict = json.loads(filters_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Impossible de décoder le JSON des filtres : {filters_json}\nErreur : {e}")
    
    parsed = {}
    for key, cond in filters_dict.items():
        if isinstance(cond, (int, float)):  # égalité simple
            parsed[key] = lambda x, val=cond: x == val
        elif isinstance(cond, list):  # inclusion
            parsed[key] = lambda x, val=cond: x in val
        elif isinstance(cond, str):
            if cond.startswith(">="):
                threshold = float(cond[2:])
                parsed[key] = lambda x, t=threshold: x >= t
            elif cond.startswith("<="):
                threshold = float(cond[2:])
                parsed[key] = lambda x, t=threshold: x <= t
            elif cond.startswith(">"):
                threshold = float(cond[1:])
                parsed[key] = lambda x, t=threshold: x > t
            elif cond.startswith("<"):
                threshold = float(cond[1:])
                parsed[key] = lambda x, t=threshold: x < t
            else:
                parsed[key] = lambda x, val=cond: str(x) == str(val)
        else:
            raise ValueError(f"Condition non supportée pour {key}: {cond}")
    return parsed


# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Entraînement CNN multimodal (IRM + vecteurs).")

parser.add_argument('--label', type=str, required=True,
                    help="Nom du label moteur (ex: FM, ARAT, FMprop).")
parser.add_argument('--irm_list', type=str, nargs='*', default=None,
                    help="Modalités IRM à inclure (ex: T1 DTI_FA).")
parser.add_argument('--vect_list', type=str, nargs='*', default=None,
                    help="Variables vectorielles à inclure (ex: Age Sexe).")
parser.add_argument('--filters', type=str, default=None,
                    help="Filtres JSON appliqués sur les colonnes VECT. Exemple: '{\"Type_Ischemic\": 1, \"Age\": \">60\"}'")
parser.add_argument('--pickle_path', type=str, required=True,
                    help="Chemin vers le dataset pickle unifié.")
parser.add_argument('--working_directory_base', type=str,
                    default='/network/iss/home/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/',
                    help="Répertoire de base pour les résultats.")
parser.add_argument('--data_augmentation', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--flip_augmentation', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--early_stop', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--patience', type=int, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--loss', type=str, default="MSELoss")
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--regularizers_l2', type=float, default=0.001)
parser.add_argument('--model_name', type=str, default="ClassicCNNModel_5Conv_NoAveragePool_withDropout")

args = parser.parse_args()

# --- VARIABLES ---
label = args.label
irm_list = args.irm_list
vect_list = args.vect_list
filters = parse_filters(args.filters)
pickle_path = args.pickle_path
working_directory_base = args.working_directory_base

flip_augmentation = args.flip_augmentation
data_augmentation = args.data_augmentation
early_stop = args.early_stop
patience = args.patience
epochs = args.epochs
loss = args.loss
dropout_rate = args.dropout_rate
batch_size = args.batch_size
learning_rate = args.learning_rate
regularizers_l2 = args.regularizers_l2
model_name = args.model_name

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

###############################################################################

if len(irm_list)==0:
    input_channels = 1
else :
    input_channels = len(irm_list)
    
    
deep_learning_models = {
    "Mixed_CNN_5Conv_with_Vectors" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_5Conv_with_Vectors,
    "Mixed_CNN_5Conv_with_Vectors_end_Dropout" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_5Conv_with_Vectors_end_Dropout,
    "Mixed_CNN_5Conv_with_Vectors_and_2Dropouts" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_5Conv_with_Vectors_and_2Dropouts,
    "Mixed_CNN_5Conv_with_Vectors_end_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_5Conv_with_Vectors_end_Dropout_and_4FC,
    "Mixed_CNN_3Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_3Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_3FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_3FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_4FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_GatedFusion" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_GatedFusion,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC,
    "Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_BidirectionalCrossAttention_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_BidirectionalCrossAttention_Mid_Dropout_and_4FC,
    "DenseNet_121_2_first_bloc_2Conv_with_MultiheadAttention_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.DenseNet_121_2_first_bloc_2Conv_with_MultiheadAttention_Mid_Dropout_and_4FC,
    "Mixed_CNN_1Conv_with_AveragePool_Vectors_LayerNorm_CrossAttention_Mid_Dropout_and_4FC" : PyT_dl_models_mixed_CNN_vector.Mixed_CNN_1Conv_with_AveragePool_Vectors_LayerNorm_CrossAttention_Mid_Dropout_and_4FC,
    "ClassicCNNModel_5Conv_NoAveragePool_withDropout_3FC" : PyT_dl_models_classic_CNN.ClassicCNNModel_4Conv_NoAveragePool_withDropout_3FC,
    "ClassicCNNModel_4Conv_withAveragePool_withDropout_3FC" : PyT_dl_models_classic_CNN.ClassicCNNModel_4Conv_withAveragePool_withDropout_3FC,
    "ClassicCNNModel_5Conv_SparsePooling_withDropout_3FC" : PyT_dl_models_classic_CNN.ClassicCNNModel_5Conv_SparsePooling_withDropout_3FC,
    "CNN3D_Attention_5Conv_NoAdaptivePool_withDropout_4FC" : PyT_dl_models_classic_CNN.CNN3D_Attention_5Conv_NoAdaptivePool_withDropout_4FC,
    "CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_4FC" : PyT_dl_models_classic_CNN.CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_4FC,
    "CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_2FC" : PyT_dl_models_classic_CNN.CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_2FC,
    "CNN3D_CrossAttention_5Conv_AdaptivePool_withDropout_2FC" : PyT_dl_models_classic_CNN.CNN3D_CrossAttention_5Conv_AdaptivePool_withDropout_2FC,
    "CNN3D_1024_features_CrossAttention_Optimized" : PyT_dl_models_classic_CNN.CNN3D_1024_features_CrossAttention_Optimized,
    "DenseNet_121_2_first_bloc_3Conv_RMI_only" : PyT_dl_models_classic_CNN.DenseNet_121_2_first_bloc_3Conv_RMI_only,
    "Sequential_3FC_2Dropout" : PyT_dl_models_MLP_vectors.Sequential_3FC_2Dropout,
    "Sequential_3FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_3FC_End_Dropout,
    "Sequential_4FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_4FC_End_Dropout,
    "Sequential_5FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_5FC_End_Dropout,
    "Sequential_6FC_End_Dropout" : PyT_dl_models_MLP_vectors.Sequential_6FC_End_Dropout
    }

loss_functions = {
   "WeightedMSELoss" : PyT_dl_utils_mixte_models.WeightedMSELoss(),
   "EWMSELoss" : PyT_dl_utils_mixte_models.EWMSELoss(),
   "MSELoss" : nn.MSELoss()
    }

############################### INPUTS ########################################

working_directory = os.path.join(working_directory_base, f"{label}")
resume_csv_path = os.path.join(working_directory, 'Resume_DeepLearrning_Performances.csv')

model_working_directory = os.path.join(
    working_directory,
    f"{model_name}_{label}_irm_{'-'.join(irm_list)}_vect_{'-'.join(vect_list)}"
)

os.makedirs(model_working_directory, exist_ok=True)
print(f"Working directory: {model_working_directory}")

# --- ENTRAINEMENT ---
Cnn_Model = deep_learning_models[model_name]

trained_model = PyT_dl_utils_mixte_models.training_pipeline(
    whole_dataset_path=pickle_path,
    label_name=label,
    irm_modalities=irm_list,
    vector_features=vect_list,
    filters=filters,
    model_function=lambda input_channels, vector_features_nb, input_image_shape, dropout_rate: Cnn_Model(
        input_channels, vector_features_nb, input_image_shape, dropout=dropout_rate
    ),
    model_name=model_name,
    working_dir=model_working_directory,
    performance_csv_path=resume_csv_path,
    flip_augmentation=flip_augmentation,
    data_augmentation=data_augmentation,
    epochs=epochs,
    loss_fn=loss_functions[loss],
    batch_size=batch_size,
    dropout=dropout_rate,
    learning_rate=learning_rate,
    regularizers_l2=regularizers_l2,
    metric_list=['mae', 'rmse', 'r2'],
    early_stop=early_stop,
    patience=patience,
    device=device
)

