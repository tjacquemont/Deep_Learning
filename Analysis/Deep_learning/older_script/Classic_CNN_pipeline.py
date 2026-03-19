#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:30:43 2025

@author: thomas.jacquemont
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torchio as tio

def load_dataset(dataset_path):
    """
    Charge les données X_train et y_train à partir d'un fichier pickle (tuple).

    Args:
        dataset_path (str): Chemin vers le fichier pickle.

    Returns:
        tuple: X_train et y_train sous forme de tableaux NumPy.
    """

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data[0]  # Accès au premier élément du tuple
    y_train = data[1]  # Accès au deuxième élément du tuple

    return X_train, y_train

def data_augmentation_function(dataset):
    """
    Applique un flip le long de la ligne médiane (axe Y) aux volumes entiers d'un dataset 3D.

    Args:
        dataset (tio.SubjectsDataset): Dataset TorchIO contenant les volumes 3D.

    Returns:
        tuple: X_train et y_train après l'augmentation de données.
    """

    transform = tio.RandomFlip(axes=(1,)) # Flip le long de l'axe Y (axe médian)

    augmented_dataset = transform(dataset)

    X_train = np.array([subject['image']['data'].numpy() for subject in augmented_dataset])
    y_train = np.array([subject['label'] for subject in augmented_dataset])

    return X_train, y_train

def performance_visualizer(history):
    """
    Visualise les courbes d'apprentissage (accuracy et loss).

    Args:
        history (tf.keras.callbacks.History): Historique de l'entraînement.
    """

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

def build_and_train_model(X_train, y_train, X_test, y_test, model_function, epochs=50):
    """
    Construit et entraîne un modèle de deep learning.

    Args:
        X_train (np.array): Données d'entraînement.
        y_train (np.array): Labels d'entraînement.
        X_test (np.array): Données de test.
        y_test (np.array): Labels de test.
        modele_fonction (function): Fonction qui définit l'architecture du modèle.

    Returns:
        tf.keras.Model: Modèle entraîné.
        tf.keras.callbacks.History: Historique de l'entraînement.
    """

    model = model_function(X_train.shape[1:])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                              epochs=50,
                              validation_data=(X_test, y_test),
                              batch_size=16,
                              verbose=1)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    return model, history


def training_pipeline(pickle_path, model_function, data_augmentation=True, epochs=50):
    """
    Pipeline complet pour charger les données, appliquer l'augmentation de données,
    et entraîner un modèle.

    Args:
        pickle_path (str): Chemin vers le fichier pickle contenant les données.
        model_function (function): Fonction qui définit et entraîne le modèle.
        data_augmentation (bool): Indique si l'augmentation de données doit être appliquée.
        epochs (int): Nombre d'epochs pour l'entrainement.
    """

    X_train, y_train = load_dataset(pickle_path)

    X_train_4d = np.expand_dims(X_train, axis=1)

    subjects = []
    for i in range(len(X_train_4d)):
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=X_train_4d[i].astype(np.float32), affine=np.eye(4)),
            label=y_train[i]
        )
        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)

    if data_augmentation:
        X_train, y_train = data_augmentation_function(dataset)
        print(f"Shape of X_train after augmentation: {X_train.shape}") #verification de la forme.
    else:
        X_train = np.squeeze(np.array([subject['image']['data'].numpy() for subject in dataset]), axis = 1)
        y_train = np.array([subject['label'] for subject in dataset])
        print(f"Shape of X_train without augmentation: {X_train.shape}") #verification de la forme.

    # Ajout de la dimension des canaux
    X_train = np.expand_dims(X_train, axis=-1)

    # Vérification de la forme après l'ajout de la dimension des canaux
    print(f"Shape of X_train after channel dimension: {X_train.shape}")

    # Transposition pour correspondre à la forme attendue par Conv3D
    X_train = np.transpose(X_train, (0, 2, 3, 4, 1, 5))

    # Vérification de la forme avant squeeze
    print(f"Shape of X_train before squeeze: {X_train.shape}")

    # Suppression de la dimension redondante
    X_train = np.squeeze(X_train, axis=4) # changed from 3 to 4.

    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Vérification de la forme de X_train avant le modèle
    print(f"Shape of X_train before model: {X_train.shape}")

    model, history = build_and_train_model(X_train, y_train, X_test, y_test, model_function, epochs=epochs)

    performance_visualizer(history)

    return model
#########################" DEEP LEARNING MODELS ###############################

def classic_cnn_model(input_shape):
    """
    Définit l'architecture d'un CNN classique.

    Args:
        input_shape (tuple): Forme des données d'entrée.

    Returns:
        tf.keras.Model: Modèle CNN.
    """

    input_layer = layers.Input(shape=input_shape)  # Ajout de la couche Input

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)  # Création du modèle Model

    return model

def cnn_resnet_model(input_shape):
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
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)  # Création du modèle Model

    return model

###############################################################################
datasets = {
"ARAT" : "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_nomalisee_arat_binaire.pkl",
"FM" : "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_nomalisee_fm_binaire.pkl"
}
motor_score = "FM"
data_augmentation = False
epochs = 50

classic_cnn_model_FM_data_augmentation_false = training_pipeline(datasets[motor_score], classic_cnn_model, data_augmentation=data_augmentation, epochs=epochs)