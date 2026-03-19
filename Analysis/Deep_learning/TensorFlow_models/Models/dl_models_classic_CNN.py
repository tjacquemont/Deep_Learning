#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:30:43 2025

@author: thomas.jacquemont
"""

from tensorflow.keras import layers, models, regularizers

# modifications:
#    - regularizers_l2 = 0.001 --> 0.01
#    - layers.Dense 128 --> 64 
 
def classic_cnn_model(input_shape, regularizers_l2=0.01, dropout=0.5):
    """
    Définit l'architecture d'un CNN classique.

    Args:
        input_shape (tuple): Forme des données d'entrée.
        regularizers_l2 (float) : force de la regularisation L2
        dropout (float): proportion de dropout

    Returns:
        tf.keras.Model: Modèle CNN.
    """

    input_layer = layers.Input(shape=input_shape)  # Ajout de la couche Input

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularizers_l2))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularizers_l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(regularizers_l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2))(x)
    x = layers.Dropout(dropout)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

def cnn_resnet_model(input_shape, dropout=0.5):
    """
    Définit l'architecture d'un CNN ResNet.

    Args:
        input_shape (tuple): Forme des données d'entrée.

    Returns:
        tf.keras.Model: Modèle CNN ResNet.
    """

    def res_block(x, filters, kernel_size=3):
        res = layers.Conv3D(filters, kernel_size, padding='same', activation='relu')(x)
        res = layers.BatchNormalization()(res)
        res = layers.Conv3D(filters, kernel_size, padding='same')(res)
        res = layers.BatchNormalization()(res)

        shortcut = layers.Conv3D(filters, 1, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([res, shortcut])
        x = layers.Activation('relu')(x)
        return x

    input_layer = layers.Input(shape=input_shape)  # Ajout de la couche Input
    x = layers.Conv3D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)

    x = res_block(x, 32)
    x = layers.MaxPooling3D(2)(x)

    x = res_block(x, 64)
    x = layers.MaxPooling3D(2)(x)

    x = res_block(x, 128)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer) 

    return model
