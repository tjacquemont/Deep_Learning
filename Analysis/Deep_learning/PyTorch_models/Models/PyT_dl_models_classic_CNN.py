#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:25:42 2025

@author: thomas.jacquemont
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from monai.networks.layers import Norm
from monai.transforms import Resize

# Attention ici l'initialisation des modèles prenenet en input la variables 'vectors_featurs_dim' et 'input_image_chape'
# mais ne les utilise pas, de même le forward prend x_img et x_tabular en input mais n'utilise que x_img.
# Celà est scripté ainsi uniquement pour pouvoir lancer les modèle dans le même pipeline et utils que les 
# modèles mixtes


class ClassicCNNModel_5Conv_NoAveragePool_NoDropout(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, regularizers_l2=0, num_classes=1,dropout=None):
        super(ClassicCNNModel_5Conv_NoAveragePool_NoDropout, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        #
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        #
        # self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        #
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        #
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(24576, 128)
#        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Stocking the regularizer in the model
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        # x = self.avgpool(x)
        #
        # x = self.pool3(x)
        #
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
        return self.fc2(x)


class ClassicCNNModel_5Conv_NoAveragePool_withDropout(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, regularizers_l2=0, num_classes=1, dropout=0.1):
        super(ClassicCNNModel_5Conv_NoAveragePool_withDropout, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        #
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        #
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        #
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(24576, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Stocking the regularization_parameters in the model
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ClassicCNNModel_5Conv_NoAveragePool_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape,  regularizers_l2=0, dropout=0.1, num_classes=1):
        super(ClassicCNNModel_5Conv_NoAveragePool_withDropout_3FC, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        #
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        #
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        #
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(24576, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Stocking the regularization_parameters in the model
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class ClassicCNNModel_5Conv_NoAveragePool_withDropout_ReluBeforBatchNorm(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, regularizers_l2=0, num_classes=1, dropout=0.1):
        super(ClassicCNNModel_5Conv_NoAveragePool_withDropout_ReluBeforBatchNorm, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        #
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        #
        # self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        #
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        #
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(24576, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Stocking the regularizer in the model
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(self.bn1(F.relu(self.conv1(x_img))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        x = self.pool5(self.bn5(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ClassicCNNModel_4Conv_NoAveragePool_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, regularizers_l2=0, dropout=0.1, num_classes=1):
        super(ClassicCNNModel_4Conv_NoAveragePool_withDropout_3FC, self).__init__()
        
        # --- 4 Convolutional Blocks ---
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 7 * 9 * 8, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # --- Régularisation ---
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ClassicCNNModel_4Conv_withAveragePool_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, regularizers_l2=0, dropout=0.1, num_classes=1):
        super(ClassicCNNModel_4Conv_withAveragePool_withDropout_3FC, self).__init__()
        
        # --- 4 Convolutional Blocks ---
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # --- Régularisation ---
        self.regularizers_l2 = regularizers_l2

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(F.relu(self.bn4(self.conv4(x))))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ClassicCNNModel_8Conv_withAveragePool_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, input_image_shape, num_classes=1, dropout=0.3):
        super(ClassicCNNModel_8Conv_withAveragePool_withDropout_3FC, self).__init__()

        # 8 convolutions
        self.conv1 = nn.Conv3d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.conv6 = nn.Conv3d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm3d(256)
        self.pool6 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(512)
        self.conv8 = nn.Conv3d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm3d(512)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x_img, x_tabular=None):
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = self.global_pool(x)   # -> (batch, 512, 1, 1, 1)
        x = torch.flatten(x, 1)   # -> (batch, 512)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class ClassicCNNModel_8Conv_withoutAveragePool_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, input_image_shape, num_classes=1, dropout=0.3):
        super(ClassicCNNModel_8Conv_withoutAveragePool_withDropout_3FC, self).__init__()

        # Même backbone que la version 1
        self.conv1 = nn.Conv3d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Conv3d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.conv6 = nn.Conv3d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm3d(256)
        self.pool6 = nn.MaxPool3d(2)

        self.conv7 = nn.Conv3d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(512)
        self.conv8 = nn.Conv3d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm3d(512)

        # ⚠️ calcul dynamique du flatten
        with torch.no_grad():
            dummy = torch.zeros(1, *input_image_shape)
            x = F.relu(self.bn1(self.conv1(dummy)))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool6(F.relu(self.bn6(self.conv6(x))))
            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            flatten_dim = x.numel() // x.shape[0]

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x_img, x_tabular=None):
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = torch.flatten(x, 1)   # pas de global pooling

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class ClassicCNNModel_5Conv_SparsePooling_withDropout_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, num_classes=1, dropout=0.3):
        super(ClassicCNNModel_5Conv_SparsePooling_withDropout_3FC, self).__init__()

        # --- Convolution blocks ---
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(2)

        # --- calcul dynamique du flatten ---
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_image_shape)
            x = F.relu(self.bn1(self.conv1(dummy)))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            flatten_dim = x.numel() // x.shape[0]

        # --- Fully Connected ---
        self.fc1 = nn.Linear(flatten_dim, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x_img, x_tabular=None):  # x_tabular ignoré ici
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# FINE TUNING MODEL USING MONAI

class DenseNet_121_2_first_bloc_3Conv_RMI_only(nn.Module):
    def __init__(self, input_channels,  vector_features_dim, input_image_shape, target_image_shape=(96, 96, 96), dropout=0.5, embed_dim=256, num_classes=1):
        super(DenseNet_121_2_first_bloc_3Conv_RMI_only, self).__init__()

        # Resize transform for image standardization
        self.target_image_shape = target_image_shape
        self.resize_transform = Resize(spatial_size=self.target_image_shape)

        # DenseNet121 Loading
        full_model = DenseNet121(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=1024,
            init_features=64,
            block_config=(6, 12, 24, 16),  # DenseNet-121
            norm=Norm.BATCH,
        )

        # --- Extractor cut at the 2nd dense block ---
        self.feature_extractor = nn.Sequential(
            full_model.features.conv0,
            full_model.features.norm0,
            full_model.features.relu0,
            full_model.features.pool0,
            full_model.features.denseblock1,
            full_model.features.transition1,
            full_model.features.denseblock2
        )
        
        # Freeze pretrained features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # --- Adapter: bloc de convolutions supplémentaires ---
        self.adapter = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, embed_dim, kernel_size=3, padding=1),  
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # sortie (B, embed_dim, 1, 1, 1)
        )

        # --- Classification MLP ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_tabular):
        # Resize si nécessaire
        if x_img.shape[2:] != self.target_image_shape:
            x_img = self.resize_transform(x_img)

        # CNN Feature extractor (DenseNet121 jusqu'à denseblock2)
        x = self.feature_extractor(x_img)  

        # Adapter (conv supplémentaires + pooling global)
        x = self.adapter(x)
        x = x.view(x.size(0), -1)  # (B, embed_dim)

        # Classification finale
        x = self.classifier(x)

        return x

# Multi Level Attention

class AttentionBlock(nn.Module):
    """Attention block pour pondérer les features d'un niveau de CNN."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),  # carte d'attention 1 channel
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : (B, C, D, H, W)
        attn_map = self.attn(x)  # (B,1,D,H,W)
        # pondération et moyenne spatiale
        x_weighted = (x * attn_map).mean(dim=[2,3,4])  # (B,C)
        return x_weighted, attn_map

class CNN3D_Attention_5Conv_NoAdaptivePool_withDropout_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, dropout=0.3, num_classes=1):
        super(CNN3D_Attention_5Conv_NoAdaptivePool_withDropout_4FC, self).__init__()

        # CNN 3D avec 5 convolutions et MaxPool3D
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
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Attention pour les 3 derniers niveaux : conv3, conv4, conv5
        self.attn3 = AttentionBlock(128)
        self.attn4 = AttentionBlock(256)
        self.attn5 = AttentionBlock(512)

        # MLP final
        self.fc1 = nn.Linear(128 + 256 + 512, 256)  # concat des vecteurs attentionnés
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x_img, x_tabular=None):
        # CNN
        x1 = self.conv1(x_img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Attention multi-niveaux sur les 3 derniers niveaux
        x3_attn, _ = self.attn3(x3)
        x4_attn, _ = self.attn4(x4)
        x5_attn, _ = self.attn5(x5)

        # Fusion
        x_fused = torch.cat([x3_attn, x4_attn, x5_attn], dim=1)

        # MLP final
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, local_channels, global_dim):
        super(CrossAttentionBlock, self).__init__()
        self.local_channels = local_channels
        self.global_dim = global_dim

        # Projette le tenseur local vers la même dimension que le global (clé et valeur)
        self.key_proj = nn.Conv3d(local_channels, global_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(local_channels, global_dim, kernel_size=1)

        # Projette le vecteur global vers la dimension de query
        self.query_proj = nn.Linear(global_dim, global_dim)

        # Projection finale pour reconvertir la dimension globale en local_channels si nécessaire
        self.out_proj = nn.Conv3d(global_dim, local_channels, kernel_size=1)

    def forward(self, local_feat, global_feat):
        """
        local_feat: [B, C_local, D, H, W]
        global_feat: [B, global_dim]
        """
        B, C, D, H, W = local_feat.shape
        N = D * H * W

        # Project local features
        keys = self.key_proj(local_feat).view(B, self.global_dim, N).permute(0, 2, 1)    # [B, N, global_dim]
        values = self.value_proj(local_feat).view(B, self.global_dim, N).permute(0, 2, 1) # [B, N, global_dim]

        # Project global feature as query
        query = self.query_proj(global_feat).unsqueeze(1)  # [B, 1, global_dim]

        # Attention
        attn_scores = torch.bmm(query, keys.permute(0, 2, 1)) / (self.global_dim ** 0.5)  # [B,1,N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B,1,N]

        # Weighted sum
        context = torch.bmm(attn_weights, values)  # [B,1,global_dim]
        context = context.squeeze(1)  # [B, global_dim]

        # Broadcast context back to spatial dims
        context = context.view(B, self.global_dim, 1, 1, 1).expand(-1, -1, D, H, W)  # [B, global_dim, D, H, W]

        # Combine context with local features
        out = self.out_proj(context) + local_feat  # [B, C_local, D, H, W]
        out = out.mean(dim=[2,3,4])  # Global average pooling pour retourner un vecteur [B, C_local]
        return out
    
class CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, global_dim=512, dropout=0.3, num_classes=1):
        super(CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_4FC, self).__init__()

        # CNN 3D avec 5 convolutions
        self.conv1 = nn.Sequential(
           nn.Conv3d(input_channels, 32, 3, padding=1),
           nn.BatchNorm3d(32),
           nn.ReLU(),
           nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
           nn.Conv3d(32, 64, 3, padding=1),
           nn.BatchNorm3d(64),
           nn.ReLU(),
           nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
           nn.Conv3d(64, 128, 3, padding=1),
           nn.BatchNorm3d(128),
           nn.ReLU(),
           nn.MaxPool3d(2)
        )
        self.conv4 = nn.Sequential(
           nn.Conv3d(128, 256, 3, padding=1),
           nn.BatchNorm3d(256),
           nn.ReLU(),
           nn.MaxPool3d(2)
        )
       # Conv5 adaptée : pas de pooling pour conserver 7x9x8
        self.conv5 = nn.Sequential(
           nn.Conv3d(256, 512, 3, padding=1),
           nn.BatchNorm3d(512),
           nn.ReLU()
        )

        # Vecteur global
        self.global_proj_conv = nn.Conv3d(512, 256, 1)   # réduit les canaux
        self.global_proj_fc = nn.Linear(256*7*9*8, global_dim)

        # Cross-attention pour les 3 derniers niveaux
        self.attn3 = CrossAttentionBlock(128, global_dim)
        self.attn4 = CrossAttentionBlock(256, global_dim)
        self.attn5 = CrossAttentionBlock(512, global_dim)

        # MLP final
        self.fc1 = nn.Linear(128+256+512, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x_img, x_tabular=None):
       x1 = self.conv1(x_img)
       x2 = self.conv2(x1)
       x3 = self.conv3(x2)
       x4 = self.conv4(x3)
       x5 = self.conv5(x4)

       # vecteur global
       global_feat = self.global_proj_conv(x5)
       B, C, D, H, W = global_feat.shape
       global_feat = global_feat.view(B, -1)
       global_feat = F.relu(self.global_proj_fc(global_feat))

       # cross-attention
       x3_attn = self.attn3(x3, global_feat)
       x4_attn = self.attn4(x4, global_feat)
       x5_attn = self.attn5(x5, global_feat)

       x_fused = torch.cat([x3_attn, x4_attn, x5_attn], dim=1)

       # MLP final
       x = F.relu(self.fc1(x_fused))
       x = self.dropout(x)
       x = F.relu(self.fc2(x))
       x = F.relu(self.fc3(x))
       return self.fc4(x)   


class CNN3D_1024_features_CrossAttention_Optimized(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape,
                 global_dim=512, dropout=0.3, num_classes=1):
        super(CNN3D_1024_features_CrossAttention_Optimized, self).__init__()

        # CNN 3D avec 5 convolutions
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Dropout3d(p=0.5)
        )

        # --- Réduction de la taille avant FC ---
        self.global_proj_conv = nn.Conv3d(1024, 512, 1)
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))   # fixe à 2x2x2
        self.global_proj_fc = nn.Linear(512 * 2 * 2 * 2, global_dim)  # 512*8 = 4096 → 512

        # --- Pooling avant attention pour réduire N ---
        self.pool3 = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.pool4 = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.pool5 = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Cross-attention pour les 3 derniers niveaux
        self.attn3 = CrossAttentionBlock(256, global_dim)
        self.attn4 = CrossAttentionBlock(512, global_dim)
        self.attn5 = CrossAttentionBlock(1024, global_dim)

        # MLP final
        self.fc1 = nn.Linear(256 + 512 + 1024, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x_img, x_tabular=None):
        # CNN backbone
        x1 = self.conv1(x_img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # --- vecteur global ---
        global_feat = self.global_proj_conv(x5)      # [B, 512, D, H, W]
        global_feat = self.global_pool(global_feat)  # [B, 512, 2, 2, 2]
        B, C, D, H, W = global_feat.shape
        global_feat = global_feat.view(B, -1)        # [B, 4096]
        global_feat = F.relu(self.global_proj_fc(global_feat))  # [B, 512]

        # --- pooling pour réduire N avant attention ---
        x3_pooled = self.pool3(x3)  # [B, 256, 4, 4, 4]
        x4_pooled = self.pool4(x4)  # [B, 512, 4, 4, 4]
        x5_pooled = self.pool5(x5)  # [B, 1024, 4, 4, 4]

        # --- cross-attention ---
        x3_attn = self.attn3(x3_pooled, global_feat)
        x4_attn = self.attn4(x4_pooled, global_feat)
        x5_attn = self.attn5(x5_pooled, global_feat)

        # --- fusion ---
        x_fused = torch.cat([x3_attn, x4_attn, x5_attn], dim=1)  # [B, 1792]

        # --- MLP ---
        x = F.relu(self.fc1(x_fused))  # [B, 512]
        x = self.dropout(x)
        return self.fc2(x)   

################################# Working CNN #################################
class CNN3D_CrossAttention_5Conv_AdaptivePool_withDropout_2FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, global_dim=512, dropout=0.3, num_classes=1):
        super(CNN3D_CrossAttention_5Conv_AdaptivePool_withDropout_2FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout3d(p=0.5)
        )

        
        self.global_proj_fc = nn.LazyLinear(global_dim)

        # Attention blocks
        self.attn3 = CrossAttentionBlock(128, global_dim)
        self.attn4 = CrossAttentionBlock(256, global_dim)
        self.attn5 = CrossAttentionBlock(512, global_dim)

        # MLP final (LazyLinear ici aussi pour rendre adaptatif)
        self.fc1 = nn.LazyLinear(256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_img, x_tabular=None):
        x1 = self.conv1(x_img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # (B, 512, 1, 1, 1) -> (B, 512)
        x5_flat = x5.view(x5.size(0), -1)
        global_feat = F.relu(self.global_proj_fc(x5_flat))

        # Attention
        x3_attn = self.attn3(x3, global_feat)
        x4_attn = self.attn4(x4, global_feat)
        x5_attn = self.attn5(x5, global_feat)

        # Fusion
        x_fused = torch.cat([x3_attn, x4_attn, x5_attn], dim=1)

        # Classification
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        return self.fc2(x)
    

class CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_2FC(nn.Module):
    def __init__(self, input_channels, vector_features_dim, input_image_shape, global_dim=512, dropout=0.3, num_classes=1):
        super(CNN3D_CrossAttention_5Conv_NoAdaptivePool_withDropout_2FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=0.5)
        )

        
        self.global_proj_fc = nn.LazyLinear(global_dim)

        # Attention blocks
        self.attn3 = CrossAttentionBlock(128, global_dim)
        self.attn4 = CrossAttentionBlock(256, global_dim)
        self.attn5 = CrossAttentionBlock(512, global_dim)

        # MLP final (LazyLinear ici aussi pour rendre adaptatif)
        self.fc1 = nn.LazyLinear(256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_img, x_tabular=None):
        x1 = self.conv1(x_img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # (B, 512, 1, 1, 1) -> (B, 512)
        x5_flat = x5.view(x5.size(0), -1)
        global_feat = F.relu(self.global_proj_fc(x5_flat))

        # Attention
        x3_attn = self.attn3(x3, global_feat)
        x4_attn = self.attn4(x4, global_feat)
        x5_attn = self.attn5(x5, global_feat)

        # Fusion
        x_fused = torch.cat([x3_attn, x4_attn, x5_attn], dim=1)

        # Classification
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        return self.fc2(x)