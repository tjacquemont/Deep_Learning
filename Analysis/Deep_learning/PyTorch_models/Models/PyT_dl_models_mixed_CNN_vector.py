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

class Mixed_CNN_5Conv_with_Vectors(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.1, regularizers_l2=0.0, num_classes=1):
        """
        Args:
            input_channels (int): nombre de canaux d'entrée des images (ex: 1 pour images médicales)
            vector_features_dim (int): dimension des données tabulaires
            input_image_shape (tuple): forme des images 3D sans les canaux (D, H, W)
        """
        super(Mixed_CNN_5Conv_with_Vectors, self).__init__()
        
        self.vector_features_nb = vector_features_nb
        self.regularizers_l2 = regularizers_l2

        # Blocs convolutionnels 3D
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

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Calcul dynamique de la taille de sortie
        self.image_features_dim = self._get_image_output_dim(input_channels, input_image_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.image_features_dim, 128)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128 + vector_features_nb, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def _get_image_output_dim(self, input_channels, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            return x.view(1, -1).size(1)

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.cat([x, x_tabular], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
class Mixed_CNN_5Conv_with_Vectors_end_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.1, regularizers_l2=0.0, num_classes=1):
        """
        Args:
            input_channels (int): nombre de canaux d'entrée des images (ex: 1 pour images médicales)
            vector_features_dim (int): dimension des données tabulaires
            input_image_shape (tuple): forme des images 3D sans les canaux (D, H, W)
        """
        super(Mixed_CNN_5Conv_with_Vectors_end_Dropout, self).__init__()
        
        self.vector_features_nb = vector_features_nb
        self.regularizers_l2 = regularizers_l2

        # Blocs convolutionnels 3D
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

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Calcul dynamique de la taille de sortie
        self.image_features_dim = self._get_image_output_dim(input_channels, input_image_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.image_features_dim, 128)

        self.fc2 = nn.Linear(128 + vector_features_nb, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_classes)

    def _get_image_output_dim(self, input_channels, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            return x.view(1, -1).size(1)

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, x_tabular], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Mixed_CNN_5Conv_with_Vectors_and_2Dropouts(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.1, regularizers_l2=0.0, num_classes=1):
        """
        Args:
            input_channels (int): nombre de canaux d'entrée des images (ex: 1 pour images médicales)
            vector_features_dim (int): dimension des données tabulaires
            input_image_shape (tuple): forme des images 3D sans les canaux (D, H, W)
        """
        super(Mixed_CNN_5Conv_with_Vectors_and_2Dropouts, self).__init__()
        
        self.vector_features_nb = vector_features_nb
        self.regularizers_l2 = regularizers_l2

        # Blocs convolutionnels 3D
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

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Calcul dynamique de la taille de sortie
        self.image_features_dim = self._get_image_output_dim(input_channels, input_image_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.image_features_dim, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128 + vector_features_nb, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_classes)

    def _get_image_output_dim(self, input_channels, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            return x.view(1, -1).size(1)

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.cat([x, x_tabular], dim=1)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
        

class Mixed_CNN_5Conv_with_Vectors_end_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.1, regularizers_l2=0.0, num_classes=1):
        """
        Args:
            input_channels (int): nombre de canaux d'entrée des images (ex: 1 pour images médicales)
            vector_features_dim (int): dimension des données tabulaires
            input_image_shape (tuple): forme des images 3D sans les canaux (D, H, W)
        """
        super(Mixed_CNN_5Conv_with_Vectors_end_Dropout_and_4FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb
        self.regularizers_l2 = regularizers_l2

        # Blocs convolutionnels 3D
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

        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Calcul dynamique de la taille de sortie
        self.image_features_dim = self._get_image_output_dim(input_channels, input_image_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.image_features_dim, 128)

        self.fc2 = nn.Linear(128 + vector_features_nb, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
       
        self.fc4 = nn.Linear(64, num_classes)

    def _get_image_output_dim(self, input_channels, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_shape)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.pool5(F.relu(self.bn5(self.conv5(x))))
            return x.view(1, -1).size(1)

    def forward(self, x_img, x_tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(x_img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, x_tabular], dim=1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        return self.fc4(x)


class Mixed_CNN_3Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_3Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D réduit
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
            nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP
        )

        # On récupère dynamiquement la taille de sortie
        self.image_features_dim = 128  # À cause du GAP (128x1x1x1)

        # Fusion avec données tabulaires
        self.fc1 = nn.Linear(self.image_features_dim + vector_features_nb, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = torch.cat([x, x_tabular], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Mixed_CNN_4Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_Mid_Dropout_and_3FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D réduit
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
            nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP
        )

        # On récupère dynamiquement la taille de sortie
        self.image_features_dim = 256  # À cause du GAP (128x1x1x1)

        # Fusion avec données tabulaires
        self.fc1 = nn.Linear(self.image_features_dim + vector_features_nb, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = torch.cat([x, x_tabular], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_3FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_3FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D réduit
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
            nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP
        )

        # On récupère dynamiquement la taille de sortie
        self.image_features_dim = 256  # À cause du GAP (128x1x1x1)
        
        # LayerNorm
        self.norm_image = nn.LayerNorm(256)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)

        # Fusion avec données tabulaires
        self.fc1 = nn.Linear(self.image_features_dim + vector_features_nb, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x_img = self.norm_image(x)
        x_tab = self.norm_tabular(x_tabular)

        x = torch.cat([x_img, x_tab], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_Mid_Dropout_and_4FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D réduit
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
            nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP
        )

        # On récupère dynamiquement la taille de sortie
        self.image_features_dim = 256  # À cause du GAP (128x1x1x1)
        
        # LayerNorm
        self.norm_image = nn.LayerNorm(256)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)

        # Fusion avec données tabulaires
        self.fc1 = nn.Linear(self.image_features_dim + vector_features_nb, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x_img = self.norm_image(x)
        x_tab = self.norm_tabular(x_tabular)
        
        
        x = torch.cat([x_img, x_tab], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_GatedFusion(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_GatedFusion, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN 3D réduit
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
            nn.AdaptiveAvgPool3d((1, 1, 1))  # GAP
        )

        # On récupère dynamiquement la taille de sortie
        self.image_features_dim = 256
        
        # Normalisation
        self.norm_image = nn.LayerNorm(self.image_features_dim)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)

        # Gating attention après concaténation
        self.fusion_dim = self.image_features_dim + vector_features_nb
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.fusion_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        # Feature extraction image
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)

        # Normalisation
        x_img = self.norm_image(x)
        x_tab = self.norm_tabular(x_tabular)

        # Fusion + Gating
        x_fused = torch.cat([x_img, x_tab], dim=1)
        gate_weights = self.gate(x_fused)
        x_fused = x_fused * gate_weights  # soft attention

        # FC Layers
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, query, key_value):
        # Assuming [batch_size, embed_dim]
        Q = self.query_proj(query).unsqueeze(1)  # (B, 1, D)
        K = self.key_proj(key_value).unsqueeze(1)  # (B, 1, D)
        V = self.value_proj(key_value).unsqueeze(1)  # (B, 1, D)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # (B, 1, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, 1)
        attn_output = torch.matmul(attn_weights, V)  # (B, 1, D)

        output = self.out_proj(attn_output.squeeze(1))  # (B, D)
        return output

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(BidirectionalCrossAttention, self).__init__()
        self.query_img = nn.Linear(embed_dim, embed_dim)
        self.key_tab = nn.Linear(embed_dim, embed_dim)
        self.value_tab = nn.Linear(embed_dim, embed_dim)

        self.query_tab = nn.Linear(embed_dim, embed_dim)
        self.key_img = nn.Linear(embed_dim, embed_dim)
        self.value_img = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feat, tab_feat):
        # Reshape en (B, 1, D)
        img_feat = img_feat.unsqueeze(1)
        tab_feat = tab_feat.unsqueeze(1)

        # Image attends tabular
        Q1 = self.query_img(img_feat)
        K1 = self.key_tab(tab_feat)
        V1 = self.value_tab(tab_feat)
        attn_weights1 = self.softmax(torch.bmm(Q1, K1.transpose(1, 2)) / (Q1.size(-1) ** 0.5))
        context1 = torch.bmm(attn_weights1, V1).squeeze(1)  # (B, D)

        # Tabular attends image
        Q2 = self.query_tab(tab_feat)
        K2 = self.key_img(img_feat)
        V2 = self.value_img(img_feat)
        attn_weights2 = self.softmax(torch.bmm(Q2, K2.transpose(1, 2)) / (Q2.size(-1) ** 0.5))
        context2 = torch.bmm(attn_weights2, V2).squeeze(1)  # (B, D)

        # Fusion des deux directions
        fused = torch.cat([context1, context2], dim=1)  # (B, 2D)

        return fused


class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC, self).__init__()
        
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
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.image_features_dim = 256
        self.norm_image = nn.LayerNorm(256)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        
        # Projection tabulaire vers même dim
        self.tabular_proj = nn.Linear(vector_features_nb, 256)

        # Cross-Attention module
        self.cross_attention = CrossAttention(embed_dim=256)

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
        x = x.view(x.size(0), -1)  # (B, 256)

        x_img = self.norm_image(x)
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

class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_BidirectionalCrossAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_BidirectionalCrossAttention_Mid_Dropout_and_4FC, self).__init__()
        
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
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.image_features_dim = 256
        self.norm_image = nn.LayerNorm(256)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        
        # Projection tabulaire vers même dim
        self.tabular_proj = nn.Linear(vector_features_nb, 256)

        # Cross-Attention module
        self.cross_attention = BidirectionalCrossAttention(embed_dim=256)

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
        x = x.view(x.size(0), -1)  # (B, 256)

        x_img = self.norm_image(x)
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)

        # Cross-attention
        x_attn = self.cross_attention(x_img, x_tab_proj)

        x = F.relu(self.fc1(x_attn))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, embed_dim=256, num_heads=2, num_classes=1):
        super(Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb
        self.embed_dim = embed_dim

        # --- CNN Blocks ---
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
            nn.Conv3d(128, embed_dim, kernel_size=3, padding=1),  # output 256-dim features
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # --- Normalization and Tabular projection ---
        self.norm_image = nn.LayerNorm(embed_dim)
        
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        self.tabular_proj = nn.Linear(vector_features_nb, embed_dim)

        # --- Multihead Attention ---
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # --- Fusion + MLP ---
        self.fc1 = nn.Linear(embed_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        # ---- CNN Encoder ----
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # shape: (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        x_img = self.norm_image(x)  # (B, 256)
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)  # (B, 256)

        # ---- Multihead Cross Attention ----
        q = x_img.unsqueeze(1)       # (B, 1, 256)
        kv = x_tab_proj.unsqueeze(1) # (B, 1, 256)
        attn_output, _ = self.cross_attention(q, kv, kv)  # (B, 1, 256)
        x_attn = attn_output.squeeze(1)  # (B, 256)

        # ---- Fusion + MLP ----
        x_fused = torch.cat([x_img, x_attn], dim=1)  # (B, 512)
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Mixed_CNN_4Conv_without_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, embed_dim=256, num_heads=2, num_classes=1):
        super(Mixed_CNN_4Conv_without_AveragePool_Vectors_LayerNorm_with_MultiheadAttention_Mid_Dropout_and_4FC, self).__init__()

        self.embed_dim = embed_dim
        self.vector_features_nb = vector_features_nb

        # --- CNN Blocks ---
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)  # -> (59, 79, 67)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)  # -> (29, 39, 33)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)  # -> (14, 19, 16)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.MaxPool3d(2)  # -> (7, 9, 8)
        )

        # Compute flattened size manually
        D, H, W = input_image_shape
        self.spatial_dims = (
            D // 2 // 2 // 2 // 2,
            H // 2 // 2 // 2 // 2,
            W // 2 // 2 // 2 // 2
        )
        flat_dim = embed_dim * self.spatial_dims[0] * self.spatial_dims[1] * self.spatial_dims[2]

        # --- Projection after flatten ---
        self.norm_image = nn.LayerNorm(flat_dim)
        self.reduce_proj = nn.Linear(flat_dim, embed_dim)

        # --- Tabular processing ---
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        self.tabular_proj = nn.Linear(vector_features_nb, embed_dim)

        # --- Multihead Attention ---
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # --- Fusion + MLP ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        # CNN feature extractor
        x = self.conv1(x_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, embed_dim, 7, 9, 8)

        x = x.view(x.size(0), -1)  # Flatten -> (B, 128448)
        x_img = self.norm_image(x)
        x_img = self.reduce_proj(x_img)  # (B, 256)

        # Tabular branch
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)  # (B, 256)

        # Cross-attention
        q = x_img.unsqueeze(1)       # (B, 1, 256)
        kv = x_tab_proj.unsqueeze(1) # (B, 1, 256)
        attn_output, _ = self.cross_attention(q, kv, kv)
        x_attn = attn_output.squeeze(1)  # (B, 256)

        # Fusion + Classification
        x_fused = torch.cat([x_img, x_attn], dim=1)  # (B, 512)
        out = self.classifier(x_fused)
        return out

# TESTING IRM PLUS VALUE

class Mixed_CNN_1Conv_with_AveragePool_Vectors_LayerNorm_CrossAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Mixed_CNN_1Conv_with_AveragePool_Vectors_LayerNorm_CrossAttention_Mid_Dropout_and_4FC, self).__init__()
        
        self.vector_features_nb = vector_features_nb

        # CNN réduit à une seule couche
        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),                 
            nn.AdaptiveAvgPool3d((1, 1, 1)) 
        )

        self.image_features_dim = 32
        self.norm_image = nn.LayerNorm(self.image_features_dim)

        # Tabular branch
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        self.tabular_proj = nn.Linear(vector_features_nb, self.image_features_dim)

        # Cross-attention
        self.cross_attention = CrossAttention(embed_dim=self.image_features_dim)

        # Classifier
        self.fc1 = nn.Linear(self.image_features_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x_img, x_tabular):
        # Image path
        x_img = self.conv(x_img)     # -> (B, 32, 1, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)  # (B, 32)
        x_img = self.norm_image(x_img)

        # Tabular path
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)

        # Cross-attention
        x_attn = self.cross_attention(x_img, x_tab_proj)

        # Fusion
        x_fused = torch.cat([x_img, x_attn], dim=1)

        # Classification
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
# FINE TUNING MODEL USING MONAI

# DenseNet_121_2_first_bloc_3Conv_BatchNorm_with_MultiheadAttention_Mid_Dropout_and_2FC
class DenseNet_121_2_first_bloc_2Conv_with_MultiheadAttention_Mid_Dropout_and_4FC(nn.Module):
    def __init__(self, input_channels, vector_features_nb, target_image_shape=(96, 96, 96), dropout=0.5, embed_dim=256, num_heads=2, num_classes=1):
        super(DenseNet_121_2_first_bloc_2Conv_with_MultiheadAttention_Mid_Dropout_and_4FC, self).__init__()

        # Resize transform for image standardization
        self.target_image_shape = target_image_shape
        self.resize_transform = Resize(spatial_size=self.target_image_shape)
        
        # DenseNet121 Loading
        full_model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1024,
            init_features=64,
            block_config=(6, 12, 24, 16),  # DenseNet-121
            norm=Norm.BATCH,
        )

        # --- Extractor cut at the 2th dense block ---
        self.feature_extractor = nn.Sequential(
            full_model.features.conv0,
            full_model.features.norm0,
            full_model.features.relu0,
            full_model.features.pool0,
            full_model.features.denseblock1,
            full_model.features.transition1,
            full_model.features.denseblock2
        )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # --- Adapter 2 block of 3DConv  ---
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
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # --- Normalization and Tabular projection ---
        self.norm_image = nn.LayerNorm(embed_dim)
        self.norm_tabular = nn.LayerNorm(vector_features_nb)
        self.tabular_proj = nn.Linear(vector_features_nb, embed_dim)

        # --- Multihead Attention ---
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # --- Fusion + MLP ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
            )

    def forward(self, x_img, x_tabular):
        
        if x_img.shape[2:] != self.target_image_shape:
            x_img = self.resize_transform(x_img)
            
        # --- CNN Feature extractor (DenseNet121 jusqu'à denseblock2) ---
        x = self.feature_extractor(x_img)  
    
        # --- Adapter block (2 Conv3D + BN + ReLU + AvgPool3D) ---
        x = self.adapter(x)
        x = x.view(x.size(0), -1)  
    
        # --- Normalize CNN embedding ---
        x_img = self.norm_image(x)
    
        # --- Process tabular vector ---
        x_tab = self.norm_tabular(x_tabular)
        x_tab_proj = self.tabular_proj(x_tab)
    
        # --- Cross Attention (image as query, vector as key & value) ---
        q = x_img.unsqueeze(1)       # (B, 1, embed_dim)
        kv = x_tab_proj.unsqueeze(1) # (B, 1, embed_dim)
        attn_output, _ = self.cross_attention(q, kv, kv)  # (B, 1, embed_dim)
        x_attn = attn_output.squeeze(1)  # (B, embed_dim)
    
        # --- Fusion ---
        x_fused = torch.cat([x_img, x_attn], dim=1)  # (B, embed_dim * 2)
    
        # --- Fully Connected MLP ---
        x = self.classifier(x_fused)
    
        return x