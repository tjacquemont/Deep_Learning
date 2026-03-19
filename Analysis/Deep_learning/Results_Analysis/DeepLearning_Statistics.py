#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:11:04 2025

@author: thomas.jacquemont
"""

# -------------------------------
# Chargement données et modèles
# -------------------------------

import sys
list_of_package_directory = [
    '/network/iss/home/thomas.jacquemont/PREP_AVC/Script/Deep_Learning'
]

for directory_path in list_of_package_directory:
    sys.path.append(directory_path)
import os
import pickle
from PyTorch_models.Models import PyT_dl_models_mixed_CNN_vector, PyT_dl_models_classic_CNN, PyT_dl_models_MLP_vectors
from PyTorch_models.Utils.Regression import PyT_dl_utils_mixte_models
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
    
    
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

def get_label_name(model_dir, label_list=['FM_imput_M3', 'FM_M3_J7_Max_Recovery_ratio', 'ARAT_imput_M3']):
    label_name = None
    
    for label in label_list:
        if label in os.path.basename(model_dir):
            label_name = label
    return label_name
        

def load_patient_data(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_test_patient_ID(model_dir, model_name):
    df_indices = pd.read_csv(os.path.join(model_dir, f"{model_name}_indices.csv"), sep=';')
    df_list = pd.read_csv(os.path.join(model_dir, "patient_list.csv"))
    test_idx = df_indices.query("set == 'test'")["index"].values
    patient_ids = df_list['id'].values[[x for x in test_idx]]
    
    return patient_ids

class NormalizeImg:
    """Normalise une image 3D avec mean et std donnés."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, im):
        return (im - self.mean) / self.std


def extract_and_normalized_patient_data(dataset, list_id, irm_modalities, vector_features, label_name, norm_params_path):
    """
    Extrait les données des patients et applique resize + normalisation.
    
    - dataset : liste des patients
    - list_id : IDs à charger
    - irm_modalities : liste des modalités IRM
    - vector_features : liste des features vectorielles
    - label_name : nom du label
    - norm_params_path : chemin vers normalization_params.npz
    """
    
    # --- Charger les paramètres de normalisation
    params = np.load(norm_params_path)
    mean_im, std_im = params["mean_im"], params["std_im"]
    mean_vec, std_vec = params["mean_vec"], params["std_vec"]
    
    # --- Transformations pour les images
    transform_img = transforms.Compose([
        PyT_dl_utils_mixte_models.Resize3D((96, 96, 96)),
        NormalizeImg(mean_im, std_im)
    ])
    
    dataset_entries = []

    for subj in dataset:
        if subj["id"] in list_id:
            
            # --- Label
            label_val = subj["VECT"].get(label_name)

            # --- IRM : 
            subj_imgs_list = [subj["IRM"].get(irm) for irm in irm_modalities] 
            subj_imgs_tensor = torch.from_numpy(np.stack(subj_imgs_list, axis=0)).float()
            subj_imgs_tensor = transform_img(subj_imgs_tensor)
            
            # --- VECT : normalisation
            subj_vect = torch.tensor([
                (subj["VECT"].get(v) - mean_vec[i]) / std_vec[i]
                for i, v in enumerate(vector_features)
            ], dtype=torch.float32)
            
            dataset_entries.append({
                "imgs": subj_imgs_tensor,  
                "vec": subj_vect,           
                "label": torch.tensor(float(label_val), dtype=torch.float32)
            })

    return dataset_entries


def load_model(model_path, model_class, input_channels, vector_features_nb, input_image_shape, device="cpu"):
    model = model_class(input_channels, vector_features_nb, input_image_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def Pipeline_Model_Estimation_Computation(model_dir):
    
    label_name = get_label_name(model_dir)
    
    model_name = os.path.basename(model_dir).split('_'+label_name)[0]
    model_class = deep_learning_models[model_name]

    irm_modalities = ['IRM_Diffusion_Norm', 'ADC_Norm']
    vector_features = ['Age', 'SAFE_J7', 'FM_J7', 'NIHSS_J7', 'PEM_plus1_ipsi']

    # Loading Validation Data and normalization    
    dataset = load_patient_data(pickle_path)
    list_validation_patient_id = load_test_patient_ID(model_dir, model_name)
    norm_params_path = os.path.join(model_dir, 'normalization_params.npz')
    test_dataset = extract_and_normalized_patient_data(dataset, list_validation_patient_id, irm_modalities, vector_features, label_name, norm_params_path)

    # Loadind Model
    # Preparing entries
    
    X_vec = np.array([entry["vec"] for entry in test_dataset], dtype=np.float32)
    vectors_features_nb = len(X_vec[0])
    X_img = [entry["imgs"] for entry in test_dataset]
    input_image_shape = np.stack(X_img[0], axis=0).shape
    if len(irm_modalities)==0:
        input_channels = 1
    else :
        input_channels = len(irm_modalities)
        
    # Model loading
    model_path = os.path.join(model_dir, 'best_model.pth')
    model = load_model(model_path, model_class, input_channels, vectors_features_nb, input_image_shape)
    model.eval() 

    y_true, y_pred = [], []

    for patient in test_dataset:
        y_true_patient = patient['label'].item()
        inputs_img = patient['imgs'].unsqueeze(0)   
        inputs_vec = patient['vec'].unsqueeze(0)    
    
        with torch.no_grad():
            y_pred_patient = model(inputs_img, inputs_vec).cpu().numpy().flatten()[0]
    
        y_true.append(float(y_true_patient))
        y_pred.append(float(y_pred_patient))
        

    model_results = pd.DataFrame({'Y_true' : y_true,
                                  'Y_pred' : y_pred })

    model_results.to_csv(os.path.join(model_dir, 'test_patients_model_prediction.csv'), sep=';', index=False)

#################################### Run in clusters ##########################
# Label & Model Directory Definition ###########
pickle_path = '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/PREP_Whole_Database_FA_template.pkl'
dataset = load_patient_data(pickle_path)

model_dir_list = ['/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/ARAT_imput_M3/Sequential_5FC_End_Dropout_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/FM_imput_M3/Sequential_5FC_End_Dropout_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Whole_Cohorte/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/FM_M3_J7_Max_Recovery_ratio/Sequential_5FC_End_Dropout_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/ARAT_imput_M3/Sequential_5FC_End_Dropout_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/FM_imput_M3/Sequential_5FC_End_Dropout_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Left/FM_M3_J7_Max_Recovery_ratio/Sequential_5FC_End_Dropout_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/ARAT_imput_M3/Sequential_5FC_End_Dropout_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/FM_imput_M3/Sequential_5FC_End_Dropout_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Lesion_side/IRM_Diffusion_and_ADC_norm_without_Flip_Augmentation/J7/Right/FM_M3_J7_Max_Recovery_ratio/Sequential_5FC_End_Dropout_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi'
                  ]

for model_dir in model_dir_list:
    Pipeline_Model_Estimation_Computation(model_dir)
    
    
#################################### STATISTICS ###############################
# 95 CI computation and Permutation Test

def get_label_and_model_name(model_dir, label_list=['FM_imput_M3', 'FM_M3_J7_Max_Recovery_ratio', 'ARAT_imput_M3']):
    label_name = None
    model_name = None
    
    for label in label_list:
        if label in os.path.basename(model_dir):
            label_name = label
    model_name = os.path.basename(model_dir).split('_'+label_name)[0]
    return label_name, model_name

def get_label_and_population(model_dir, population_list=['Whole_Cohorte','Lesion_side'], label_list=['FM_imput_M3', 'FM_M3_J7_Max_Recovery_ratio', 'ARAT_imput_M3']):
    population_name = None
    lesion_side = None
    label_name = None
    
    # Détection de population
    for population in population_list:
        if population in model_dir:
            population_name = population
            break
    
    # Détection du côté lésionnel
    if 'Right' in model_dir:
        population_name = 'Lesion_side'
        lesion_side = 'Right'
    elif 'Left' in model_dir:
        population_name = 'Lesion_side'
        lesion_side = 'Left'
    
    # Détection du label
    for label in label_list:
        if label in model_dir:
            label_name = label
            break
            
    return label_name, population_name, lesion_side
            

def bootstrap_metrics(model_dir, n_bootstrap=50000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_path = os.path.join(model_dir, 'test_patients_model_prediction.csv')
    y_data = pd.read_csv(y_path, sep=';')
    y_true = y_data['Y_true'].values
    y_pred = y_data['Y_pred'].values
    
    n = len(y_true)
    
    mae_vals, mse_vals, r2_vals = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_true_bs, y_pred_bs = y_true[idx], y_pred[idx]
        
        mae_vals.append(mean_absolute_error(y_true_bs, y_pred_bs))
        mse_vals.append(mean_squared_error(y_true_bs, y_pred_bs))
        r2_vals.append(r2_score(y_true_bs, y_pred_bs))
    
    def ci(arr):
        return np.percentile(arr, [2.5, 97.5])
    
    results = {
        "MAE_mean": np.mean(mae_vals), "MAE_CI_low": ci(mae_vals)[0], "MAE_CI_high": ci(mae_vals)[1],
        "MSE_mean": np.mean(mse_vals), "MSE_CI_low": ci(mse_vals)[0], "MSE_CI_high": ci(mse_vals)[1],
        "R2_mean": np.mean(r2_vals),   "R2_CI_low": ci(r2_vals)[0],   "R2_CI_high": ci(r2_vals)[1],
    }
    return results, y_true, y_pred


def permutation_test(y_true, y_pred1, y_pred2, metric, n_perm=100000, random_state=42):
    rng = np.random.default_rng(random_state)
    metric_model_1 = metric(y_true, y_pred1)
    metric_model_2 = metric(y_true, y_pred2)
    observed_differences = metric_model_1 - metric_model_2
    n = len(y_true)
    
    diffs = []
    for _ in range(n_perm):
        mask = rng.integers(0, 2, size=n)
        mix1 = np.where(mask, y_pred1, y_pred2)
        mix2 = np.where(mask, y_pred2, y_pred1)
        diffs.append(metric(y_true, mix1) - metric(y_true, mix2))
    
    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_differences))
    return metric_model_1, metric_model_2, p_value


def cohens_d(y_true, y_pred1, y_pred2, metric):
    """Calcule le Cohen's d apparié entre deux modèles."""
    # On calcule les erreurs individuelles
    if metric == mean_absolute_error:
        diffs = np.abs(y_true - y_pred1) - np.abs(y_true - y_pred2)
    elif metric == mean_squared_error:
        diffs = (y_true - y_pred1)**2 - (y_true - y_pred2)**2
    elif metric == r2_score:
        # Pour R², on calcule les scores individuels de prédiction (résidus au carré inverses)
        # On convertit la métrique pour que d>0 signifie une meilleure performance pour le modèle 2
        diffs = ((y_true - y_pred2)**2 - (y_true - y_pred1)**2)
    else:
        raise ValueError("Metric non prise en charge pour Cohen's d")
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    d = mean_diff / std_diff if std_diff != 0 else np.nan
    return d


def compare_models(parent_dir, perm_results_output_file, n_bootstrap=50000, n_perm=50000, random_state=42):
    
    results_list = []
    preds_dict = {}
    
    for model_dir in [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]:
        if not os.path.exists(os.path.join(model_dir, 'test_patients_model_prediction.csv')):
            print(f'test_patients_model_prediction.csv file does not exist for model {model_dir}')
            continue
        
        label_name, model_name = get_label_and_model_name(model_dir)
        
        res, y_true, y_pred = bootstrap_metrics(model_dir, n_bootstrap=n_bootstrap, random_state=random_state)
        
        res["model_name"] = model_name
        results_list.append(res)
        preds_dict[model_name] = (y_true, y_pred)
    
    results_df = pd.DataFrame(results_list)
    cols = ["model_name"] + [c for c in results_df.columns if c != "model_name"]
    results_df = results_df[cols]
    results_df.to_csv(os.path.join(parent_dir, "bootstrap_results.csv"), sep=';', index=False)
    
    # ---- Étape 2 : Comparaison par permutation
    model_names = list(preds_dict.keys())
    label_name, population_name, lesion_side = get_label_and_population(parent_dir)
    if os.path.exists(perm_results_output_file):
        perm_output_df = pd.read_csv(perm_results_output_file, sep=';')
    else:
        perm_output_df = pd.DataFrame()
    perm_results  = []
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            y_true1, y_pred1 = preds_dict[m1]
            y_true2, y_pred2 = preds_dict[m2]
            
            # sécurité : vérifier que Y_true est identique
            assert np.allclose(y_true1, y_true2), "Y_true mismatch entre modèles"
            y_true = y_true1
            
            result_row = {"Modele_1": m1, "Modele_2": m2, "Population" : population_name, "Lesion_Side" : lesion_side, "Label" : label_name}
            
            for metric_name, metric_func in zip(["MAE", "MSE", "R2"], [mean_absolute_error, mean_squared_error, r2_score]):
                metric_model_1, metric_model_2, p_value = permutation_test(y_true, y_pred1, y_pred2,
                                              metric_func, n_perm=n_perm,
                                              random_state=random_state)
                result_row[f"{metric_name}_model_1"] = metric_model_1
                result_row[f"{metric_name}_model_2"] = metric_model_2
                result_row[f"{metric_name}_pvalue"] = p_value
                
                d_value = cohens_d(y_true, y_pred1, y_pred2, metric_func)
                result_row[f"{metric_name}_Cohen_d"] = d_value

            perm_results.append(result_row)
    
    perm_df = pd.DataFrame(perm_results)
    if os.path.exists(perm_results_output_file):
        perm_output_df = pd.concat([perm_output_df, perm_df], ignore_index=True)
    else:
        perm_output_df = perm_df
    perm_output_df.to_csv(perm_results_output_file, sep=';', index=False)

    return results_df, perm_output_df


################################ Run in python shell ########################
parent_model_dir_list = ['/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/ARAT_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_M3_J7_Max_Recovery_ratio',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/ARAT_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_M3_J7_Max_Recovery_ratio',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/ARAT_imput_M3',
                         '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_M3_J7_Max_Recovery_ratio']

perm_results_output_file = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Permutation_Results.csv'

for parent_model_dir in parent_model_dir_list:
    compare_models(parent_model_dir, perm_results_output_file)