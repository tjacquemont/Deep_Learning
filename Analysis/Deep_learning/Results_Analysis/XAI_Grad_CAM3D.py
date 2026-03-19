#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:09:30 2025

@author: thomas.jacquemont
"""

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
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import nibabel as nib
import subprocess

def run_command(command):
#    """Executes a shell command and raises an exception if it fails."""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        raise
        
####################### Model Specifics for Grad-CAM ############################

class Mixed_CNN_4Conv_AveragePoolOut_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_AveragePoolOut_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.image_features_dim = 256
        self.norm_image = nn.LayerNorm(256)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        
        # Projection tabulaire vers même dim
        self.tabular_proj = nn.Linear(vector_features_nb, 256)

        # Cross-Attention module
        self.cross_attention = PyT_dl_models_mixed_CNN_vector.CrossAttention(embed_dim=256)

        # Fusion + MLP
        self.fc1 = nn.Linear(256 + 256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x_pooled = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x_pooled = x_pooled.view(x_pooled.size(0), -1)

        x_img = self.norm_image(x_pooled)
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)

        # Cross-attention
        x_attn = self.cross_attention(x_img, x_tab_proj)

        # Fusion
        x_fused = torch.cat([x_img, x_attn], dim=1)

        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PureCNN_4Conv_with_AveragePoolOut_LayerNorm_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim=None, input_image_shape=(96,96,96),
                 dropout=0.3, num_classes=1):
        super(PureCNN_4Conv_with_AveragePoolOut_LayerNorm_Dropout_and_4FC, self).__init__()

        # --- Bloc CNN 3D ---
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)  # 96→48
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)  # 48→24
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)  # 24→12
        )

        # --- Conv4 sans pooling global (pour Grad-CAM) ---
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        # --- Pooling global séparé (utilisé uniquement dans le forward) ---
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # --- Normalisation + FC layers ---
        self.norm_image = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular=None):
        # --- Passage convolutionnel ---
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)       # ← Hook Grad-CAM ici (sortie 12x12x12)

        # --- Pooling global pour la classification ---
        x_pooled = self.global_pool(x)     # (B,256,1,1,1)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)  # (B,256)

        # --- Normalisation + FC head ---
        x_norm = self.norm_image(x_pooled)
        x_norm = F.relu(self.fc1(x_norm))
        x_norm = self.dropout(x_norm)
        x_norm = F.relu(self.fc2(x_norm))
        x_norm = F.relu(self.fc3(x_norm))
        return self.fc4(x_norm)

################### LOADING MODELS AND VALIDATION DATASET #####################
    
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
    subject_id_list = []

    for subj in dataset:
        if subj["id"] in list_id:
            
            subject_id_list += [subj["id"]]
            
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

    return dataset_entries, subject_id_list


def load_model(model_path, model_class, input_channels, vector_features_nb, input_image_shape, device="cpu"):
    model = model_class(input_channels, vector_features_nb, input_image_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

############################## XAI GRAD-CAM ###################################

def generate_gradcam_3d(model, x_img, x_tabular, target_layer_name="conv4", retain_graph=False):
    model.eval()

    # Récupère la couche cible
    target_layer = dict(model.named_modules())[target_layer_name]
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Enregistre les hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(x_img, x_tabular)
    score = output.squeeze()  # score scalaire (ex : prédiction du score moteur)

    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=retain_graph)

    # Récupère les activations et gradients
    acts = activations[0]   # (B, C, D, H, W)
    grads = gradients[0]    # (B, C, D, H, W)

    # Moyenne des gradients par canal
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
    cam = (weights * acts).sum(dim=1, keepdim=True)    # (B, 1, D, H, W)
    cam = F.relu(cam)

    # Normalisation
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # Nettoyage
    forward_handle.remove()
    backward_handle.remove()

    return cam


def generate_smoothgradcam_3d(model, x_img, x_tabular, target_layer_name="conv4", n_samples=20, noise_sigma=0.1):
    cams = []
    x_img = x_img.clone()
    
    for i in range(n_samples):
        noise = torch.randn_like(x_img) * noise_sigma * x_img.std()
        cam = generate_gradcam_3d(model, x_img + noise, x_tabular, target_layer_name, retain_graph=True)
        cams.append(cam)
    
    smooth_cam = torch.stack(cams).mean(dim=0)
    smooth_cam -= smooth_cam.min()
    smooth_cam /= (smooth_cam.max() + 1e-8)
    return smooth_cam


def save_img_as_nifti(cam_tensor, reference_img_path, output_path):
    ref = nib.load(reference_img_path)
    cam_np = cam_tensor.squeeze().cpu().numpy()
    cam_img = nib.Nifti1Image(cam_np, affine=ref.affine, header=ref.header)
    nib.save(cam_img, output_path)
    print(f"✔ Image sauvegardée : {output_path}")


def Pipeline_Compute_SmoothGradCAM3D(model_dir, model_class, target_layer_name="conv4", reference_img_path='/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Images_Database/Other_data/FA_template.nii'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    label_name = get_label_name(model_dir)
    
    model_name = os.path.basename(model_dir).split('_'+label_name)[0]

    irm_modalities = ['IRM_Diffusion_Norm', 'ADC_Norm']
    vector_features = ['Age', 'SAFE_J7', 'FM_J7', 'NIHSS_J7', 'PEM_plus1_ipsi']
    pickle_path = '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/PREP_Whole_Database_FA_template.pkl'

    # Loading Validation Data and normalization    
    dataset = load_patient_data(pickle_path)
    list_validation_patient_id = load_test_patient_ID(model_dir, model_name)
    norm_params_path = os.path.join(model_dir, 'normalization_params.npz')
    test_dataset, subject_id_list = extract_and_normalized_patient_data(dataset, list_validation_patient_id, irm_modalities, vector_features, label_name, norm_params_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
    
    for idx, batch in enumerate(test_loader):
        
        subject_id = subject_id_list[idx]
        x_img = batch["imgs"].to(device)
        x_tabular = batch["vec"].to(device)
        target = batch["label"].to(device)
        
        # Calcul de la carte Grad-CAM ou SmoothGrad-CAM
        cam = generate_smoothgradcam_3d(model, x_img, x_tabular, target_layer_name=target_layer_name)
    
        # Interpolation à la taille IRM d’origine
        cam_up = F.interpolate(cam, size=(96,96,96), mode='trilinear', align_corners=False)
    
        # Sauvegarde Patient and CAM pour inspection
        patient_img_output_dir = os.path.join(model_dir, 'Validation_Images', 'Patients_Images')
        os.makedirs(patient_img_output_dir, exist_ok=True)
        cam_img_output_dir = os.path.join(model_dir, 'Validation_Images', f'Patients_Grad-CAM_{target_layer_name}')
        os.makedirs(cam_img_output_dir, exist_ok=True)
        
        for i, image_type in enumerate(irm_modalities):
            save_img_as_nifti(x_img[0,i,:,:,:], reference_img_path, output_path=os.path.join(patient_img_output_dir, f"image_{image_type}_patient_{subject_id}.nii.gz"))
        save_img_as_nifti(cam_up, reference_img_path, output_path=os.path.join(cam_img_output_dir, f"gradcam_patient_{subject_id}.nii.gz"))

################################## Rune to cluster #############################
mixt_model_dir_list = ['/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi']

layer_list= ['conv3', 'conv4']

for model_directory in mixt_model_dir_list:
    print(f"Running GradCAM for : {model_directory}")
    for layer in layer_list:
        Pipeline_Compute_SmoothGradCAM3D(model_directory, 
                                 Mixed_CNN_4Conv_AveragePoolOut_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC, target_layer_name=layer)

########################## Grad CAM STATISTICS ###########################
mni_image_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/Other_data/FMRIB58_FA_1mm.nii.gz'
mixt_model_dir_list = ['/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Whole_Cohorte/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/ARAT_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_ARAT_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_imput_M3/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_imput_M3_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Right/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi',
                  '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models/Left/FM_M3_J7_Max_Recovery_ratio/Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_FM_M3_J7_Max_Recovery_ratio_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi']

layer_list= ['conv3', 'conv4']

for model_directory in mixt_model_dir_list:
    diffusion_directory = os.path.join(model_directory, 'Validation_Images', 'Patients_Images')
    registered_images_directory = os.path.join(model_directory, 'Validation_Images', 'Patients_Images_to_MNI')
    os.makedirs(registered_images_directory,  exist_ok=True)
    # 1) Registration to MNI
    # Meaning all validation patient diffusion images
    merged_diffusion_path = os.path.join(diffusion_directory, 'Validation_Patient_Merged_Diffusion.nii.gz')
    fslmerge_cmd = f"fslmerge -t {merged_diffusion_path} {diffusion_directory}/image_IRM_Diffusion_Norm_patient_*.nii.gz"
    run_command(fslmerge_cmd)
    # Computing Mean Diffusion
    mean_diffusion_path = os.path.join(diffusion_directory, 'Validation_Patient_Mean_Diffusion.nii.gz')
    fslmaths_cmd = f"fslmaths {merged_diffusion_path} -Tmean {mean_diffusion_path}"
    run_command(fslmaths_cmd)
    # Registration to MNI 
    # Computing Registration
    affine_path = os.path.join(registered_images_directory, 'Diffusion_to_MNI_Affine.txt')
    affine_out_image_path =  os.path.join(registered_images_directory, 'Diffusion_merged_to_MNI_Affine.nii.gz')
    cmd_reg_aladin = f"reg_aladin -ref {mni_image_path} -flo {mean_diffusion_path}  -aff {affine_path} -res {affine_out_image_path}"
    run_command(cmd_reg_aladin)
    cpp_path = os.path.join(registered_images_directory, 'Diffusion_to_MNI_cpp.nii.gz')
    nlinear_out_image_path =  os.path.join(registered_images_directory, 'Diffusion_merged_to_MNI_nlinear.nii.gz')
    cmd_reg_f3d = f"reg_f3d -ref {mni_image_path} -flo {mean_diffusion_path} -aff {affine_path} -cpp {cpp_path} -res {nlinear_out_image_path}"
    run_command(cmd_reg_f3d)
    # Applying registration to each Grad-CAM image
    gradcam_dir_list = [os.path.join(model_directory, 'Validation_Images', f'Patients_Grad-CAM_{target_layer_name}') for target_layer_name in layer_list]
    for gradcam_dir in gradcam_dir_list :
        registered_gradcam_dir = gradcam_dir + '_to_MNI'
        os.makedirs(registered_gradcam_dir,  exist_ok=True)
        for gradcam_file in os.listdir(gradcam_dir):
            gradcam_file_to_MNI = os.path.join(registered_gradcam_dir, gradcam_file)
            in_gradcam_file = os.path.join(gradcam_dir, gradcam_file)
            reg_resample_cmd = f"reg_resample -ref {mni_image_path} -flo {in_gradcam_file} -trans {affine_path} -cpp {cpp_path} -res {gradcam_file_to_MNI}"
            run_command(reg_resample_cmd)