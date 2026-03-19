#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:19:34 2025

@author: thomas.jacquemont
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import torchio as tio
from tensorflow.keras import optimizers
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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


def data_augmentation_function_median_flip(dataset, working_dir='working_dir'):
    """
    Applique un flip de 180 degrés le long de la ligne médiane (axe Z), duplique les sujets, enregistre des coupes médianes
    et uniformise le format de X_train.

    Args:
        dataset (torchio.data.dataset.SubjectsDataset): Dataset contenant les sujets.
        working_dir (str): Répertoire où enregistrer les coupes médianes.

    Returns:
        tuple: X_train et y_train après l'augmentation de données.
    """

    transform = tio.transforms.Flip(axes=(0,), include=['image'])  # Modifier l'axe pour le flip le long de l'axe Y

    augmented_subjects = []
    for i, subject in enumerate(dataset):
        augmented_subjects.append(subject)

        flipped_data = transform(subject)  # Appliquer le flip et récupérer les données flippées
        augmented_subjects.append(flipped_data)

        qc_dir = os.path.join(working_dir, "qc_median_slices")
        if not os.path.exists(qc_dir):
            os.makedirs(qc_dir)

        original_median_slice = subject['image']['data'][0, :, :, subject['image']['data'].shape[3] // 2].numpy()
        flipped_median_slice = flipped_data['image']['data'][0, :, :, flipped_data['image']['data'].shape[3] // 2].numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_median_slice, cmap='gray')
        plt.title(f"Original Subject {i} - Median Slice")

        plt.subplot(1, 2, 2)
        plt.imshow(flipped_median_slice, cmap='gray')
        plt.title(f"Flipped Subject {i} - Median Slice")

        plt.savefig(os.path.join(qc_dir, f"subject_{i}_median_slices.png"))
        plt.close()

    augmented_dataset = tio.SubjectsDataset(augmented_subjects)

    X_train = np.array([subject['image']['data'].numpy() for subject in augmented_dataset])
    y_train = np.array([subject['label'] for subject in augmented_dataset])

    # Uniformisation du format
    X_train = np.expand_dims(X_train, axis=-1)
    X_train = np.transpose(X_train, (0, 2, 3, 4, 1, 5))
    X_train = np.squeeze(X_train, axis=5)
    X_train = np.squeeze(X_train, axis=4)

    return X_train, y_train


def performance_visualizer(history, dataset_path, model_function, epochs, y_true, y_pred, working_dir):
    """
    Visualise et enregistre les courbes d'apprentissage et la matrice de confusion.
    """

    print(f"Working directory: {working_dir}")  # Vérifier le working directory

    # Créer le répertoire de travail s'il n'existe pas
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        print(f"Created working directory: {working_dir}") # Verifier la création du working directory

    # Nom de fichier de base
    dataset_name = dataset_path.split("/")[-1].split('.')[0]
    
    base_filename = f"{dataset_name}_{model_function.__name__}_{epochs}"
    print(f'base_filename : {base_filename} ')
    print(os.path.join(working_dir, f"{base_filename}.png"))

    # Courbes d'apprentissage
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        print("Plotting accuracy...") # Verifier si cette partie du code est executée
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        print(os.path.join(working_dir, f"{base_filename}_accuracy.png"))
        plt.savefig(os.path.join(working_dir, f"{base_filename}_accuracy.png"))
        
    if 'loss' in history.history and 'val_loss' in history.history:
        print("Plotting loss...") # Verifier si cette partie du code est executée
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        plt.grid(True)
        print(os.path.join(working_dir, f"{base_filename}_loss.png"))
        plt.savefig(os.path.join(working_dir, f"{base_filename}_loss.png"))


    # Matrice de confusion
    print("Plotting confusion matrix...") # Verifier si cette partie du code est executée
    threshold = 0.5  # Seuil pour la classification binaire
    y_pred_binary = (y_pred > threshold).astype(int).flatten()  # Aplatir y_pred
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    print(os.path.join(working_dir, f"{base_filename}_confusion_matrix.png"))
    plt.savefig(os.path.join(working_dir, f"{base_filename}_confusion_matrix.png"))
  
    
def build_and_train_model(X_train, y_train, X_test, y_test, model_function, epochs=50, loss='binary_crossentropy', metric_list = ['accuracy']):
    """
    Construit et entraîne un modèle de deep learning.

    Args:
        X_train (np.array): Données d'entraînement.
        y_train (np.array): Labels d'entraînement.
        X_test (np.array): Données de test.
        y_test (np.array): Labels de test.
        modele_fonction (function): Fonction qui définit l'architecture du modèle.
        loss (str): Loss function
        metric_list (list of str): List of metrics to use

    Returns:
        tf.keras.Model: Modèle entraîné.
        tf.keras.callbacks.History: Historique de l'entraînement.
    """

    model = model_function(X_train.shape[1:])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=loss,
                  metrics=metric_list)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        batch_size=16,
                        verbose=1)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    return model, history, y_pred

def training_pipeline(pickle_path, model_function, working_dir, data_augmentation=True, epochs=50, loss='binary_crossentropy', metric_list=['accuracy']):
    """
    Pipeline complet pour charger les données, appliquer l'augmentation de données,
    et entraîner un modèle.
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
        X_train, y_train = data_augmentation_function_median_flip(dataset, working_dir=working_dir)
        print(f"Shape of X_train after augmentation: {X_train.shape}")
    else:
        X_train = np.squeeze(np.array([subject['image']['data'].numpy() for subject in dataset]), axis=1)
        y_train = np.array([subject['label'] for subject in dataset])
        print(f"Shape of X_train without augmentation: {X_train.shape}")

    # Uniformisation du format (ajout de la dimension de canal et transposition)
    X_train = np.expand_dims(X_train, axis=-1)  # Ajout de la dimension de canal
    X_train = np.transpose(X_train, (0, 2, 3, 1, 4)) # Transposition correcte

    # Division stratifiée train/test 
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2301)

    for train_index, test_index in splitter.split(X_train, y_train):
        X_train_balanced, X_test_balanced = X_train[train_index], X_train[test_index]
        y_train_balanced, y_test_balanced = y_train[train_index], y_train[test_index]
    
    # Vérification de la forme de X_train avant le modèle
    print(f"Shape of X_train before model: {X_train.shape}")
    print(f"Y test : {y_test_balanced} ")

    model, history, y_pred = build_and_train_model(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, model_function, epochs=epochs, loss=loss, metric_list=metric_list)

    performance_visualizer(history, pickle_path, model_function, epochs, y_pred, y_test_balanced, working_dir)

    return model
