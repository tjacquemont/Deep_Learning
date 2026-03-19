#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:25:42 2025

@author: thomas.jacquemont
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Sequential_3FC_2Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_3FC_2Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)
    
    
class Sequential_3FC_End_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_3FC_End_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)
    

class Sequential_4FC_End_Dropout(nn.Module):
    def __init__(self,  input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_4FC_End_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)


class Sequential_5FC_End_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_5FC_End_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)
    

class Sequential_6FC_End_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_6FC_End_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 258),
            nn.ReLU(),
            nn.Linear(258, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)
    

class Sequential_3FC_with_BatchNorm_Mid_Dropout(nn.Module):
    def __init__(self,  input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_3FC_with_BatchNorm_Mid_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)


class Sequential_4FC_with_BatchNorm_Mid_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_4FC_with_BatchNorm_Mid_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)


class Sequential_5FC_with_BatchNorm_Mid_Dropout(nn.Module):
    def __init__(self, input_channels, vector_features_nb, input_image_shape, dropout=0.3, num_classes=1):
        super(Sequential_5FC_with_BatchNorm_Mid_Dropout, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(vector_features_nb, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x_img, x_tabular):
        return self.model(x_tabular)