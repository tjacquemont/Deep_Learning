#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:25:17 2025

@author: thomas.jacquemont
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchio
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd

class EarlyStopping: 
    def __init__(self, patience=20, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
                    

class CustomDataset(Dataset):
    """Dataset personnalisé pour charger des IRM 3D + vecteurs tabulaires."""
    def __init__(self, X_img, X_vec, y, transform_img=None, transform_vec=None):
        """
        Args:
            X_img (np.ndarray or torch.Tensor): données d’IRM, forme (N, D, H, W)
            X_vec (np.ndarray or torch.Tensor): données tabulaires, forme (N, V)
            y (np.ndarray or torch.Tensor): labels, forme (N,)
            transform (callable, optional): transformation à appliquer aux images
        """
        self.X_img = torch.tensor(X_img, dtype=torch.float32)
        self.X_vec = torch.tensor(X_vec, dtype=torch.float32)              # (N, V)
        self.y     = torch.tensor(y, dtype=torch.float32).unsqueeze(1)     # (N, 1)
        self.transform_img = transform_img
        self.transform_vec = transform_vec

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        sample = {'image' : self.X_img[idx], 'vector': self.X_vec[idx], 'label' :self.y[idx]}
        if self.transform_img:
            sample['image'] = self.transform_img(sample['image'])
        if self.transform_vec:
            sample['vector'] = self.transform_vec(sample['vector'])

        return sample
    

def stratified_shuffle_split(y, test_size=0.2, random_state=None):
    """
    Réalise un split stratifié et mélangé (shuffle) des données.

    Args:
        y (np.array): Les étiquettes (labels) correspondantes à X.
        test_size (float): La proportion du jeu de données à inclure dans le split de test.
                           Doit être entre 0.0 et 1.0.
        random_state (int, optional): seed pour la reproductibilité du mélange.

    Returns:
        tuple: Un tuple contenant les indices d'entraînement et de test (train_index, test_index).
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    for clas in unique_classes:
        # Obtenez les indices de tous les échantillons appartenant à la classe actuelle
        class_indices = np.where(y == clas)[0]
        np.random.shuffle(class_indices) # Mélangez les indices pour cette classe

        # Calculez le nombre d'échantillons à mettre dans le jeu de test pour cette classe
        n_test_samples_cls = int(len(class_indices) * test_size)

        # Divisez les indices de la classe en test et entraînement
        test_cls_indices = class_indices[:n_test_samples_cls]
        train_cls_indices = class_indices[n_test_samples_cls:]

        test_indices.extend(test_cls_indices)
        train_indices.extend(train_cls_indices)

    # Convertir les listes d'indices en tableaux NumPy et les mélanger globalement
    # C'est important pour que l'ordre des échantillons ne soit pas regroupé par classe
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcule la matrice de confusion manuellement.

    Args:
        y_true (list ou np.array): Les vraies étiquettes.
        y_pred (list ou np.array): Les étiquettes prédites.
        labels (list ou np.array, optional): Une liste des étiquettes à inclure
                                             dans la matrice de confusion. Si None,
                                             toutes les étiquettes uniques de y_true et y_pred
                                             seront utilisées, triées.

    Returns:
        np.array: La matrice de confusion.
    """
    if labels is None:
        # Obtenez toutes les étiquettes uniques et triez-les
        all_labels = np.unique(np.concatenate((y_true, y_pred)))
        labels = np.sort(all_labels)

    num_classes = len(labels)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    # Créez un mapping des étiquettes aux index pour un accès facile
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            confusion_mat[true_idx, pred_idx] += 1
        # Gérer les cas où une étiquette n'est pas dans 'labels' si c'est pertinent
        # Pour une matrice de confusion standard, cela signifie généralement
        # que ces échantillons ne sont pas inclus.

    return confusion_mat
    

def load_dataset(dataset_path):
    """
    Charge les données X_train et y_train à partir d'un fichier pickle (tuple).

    Args:
        dataset_path (str): Chemin vers le fichier pickle.

    Returns:
        tuple: X_img_train, X_vec_train et y_train sous forme de tableaux NumPy.
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    X_img_train = data[0]  # Accès au premier élément du tuple
    X_vec_train = data[1]  # Accès au deuxième élément du tuple
    y_train = data[2]  # Accès au trosieme élément du tuple
    return X_img_train, X_vec_train, y_train


def data_augmentation_function_median_flip(X_img_train, X_vec_train, y_train, working_dir='working_dir'):
    """
    Applique un flip de 180 degrés le long de l'axe Y (équivalent à l'axe 0 après transposition),
    duplique les sujets et enregistre des coupes médianes.

    Args:
        X_train (np.array): Données d'entraînement (N, C, X, Y, Z).
        y_train (np.array): Labels d'entraînement.
        working_dir (str): Répertoire où enregistrer les coupes médianes.

    Returns:
        tuple: X_train et y_train après l'augmentation de données (tenseur PyTorch).
    """
    augmented_X_img = []
    augmented_X_vec = []
    augmented_y = []

    qc_dir = os.path.join(working_dir, "qc_median_slices")
    if not os.path.exists(qc_dir):
        os.makedirs(qc_dir)

    for i in range(len(X_img_train)):
        original_image = X_img_train[i]
        original_vect = X_vec_train[i]
        original_label = y_train[i]

        augmented_X_img.append(original_image)
        augmented_X_vec.append(original_vect)
        augmented_y.append(original_label)

        # Flip le long de l'axe Y (l'axe 1 après la forme actuelle)
        flipped_image = np.flip(original_image, axis=1).copy() 
        
        augmented_X_img.append(flipped_image)
        augmented_X_vec.append(original_vect)
        augmented_y.append(original_label)

        # Visualisation des coupes médianes (axe Z)
        original_median_slice = original_image[0, :, :, original_image.shape[3] // 2]
        flipped_median_slice = flipped_image[0, :, :, flipped_image.shape[3] // 2]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_median_slice, cmap='gray')
        plt.title(f"Original Subject {i} - Median Slice (Z)")

        plt.subplot(1, 2, 2)
        plt.imshow(flipped_median_slice, cmap='gray')
        plt.title(f"Flipped Subject {i} - Median Slice (Z)")

        plt.savefig(os.path.join(qc_dir, f"subject_{i}_median_slices.png"))
        plt.close()

    return np.array(augmented_X_img), np.array(augmented_X_vec), np.array(augmented_y)


def performance_visualizer(history, dataset_path, model_name, epochs, y_true, y_pred, num_classes, working_dir):
    """
    Visualise et enregistre les courbes d'apprentissage et la matrice de confusion.
    """
    print(f"Working directory: {working_dir}")

    # Créer le répertoire de travail s'il n'existe pas
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        print(f"Created working directory: {working_dir}")

    # Nom de fichier de base
    dataset_name = dataset_path.split("/")[-1].split('.')[0]
    base_filename = f"{dataset_name}_{model_name}_{epochs}"
    print(f'base_filename : {base_filename} ')
    print(os.path.join(working_dir, f"{base_filename}.png"))

    # Courbes d'apprentissage
    if history.get('train_accuracy') and history.get('val_accuracy'):
        print("Plotting accuracy...")
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_accuracy'], label='train_accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        print(os.path.join(working_dir, f"{base_filename}_accuracy.png"))
        plt.savefig(os.path.join(working_dir, f"{base_filename}_accuracy.png"))

    if history.get('train_loss') and history.get('val_loss'):
        print("Plotting loss...")
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        print(os.path.join(working_dir, f"{base_filename}_loss.png"))
        plt.savefig(os.path.join(working_dir, f"{base_filename}_loss.png"))

    # Matrice de confusion
    print("Plotting confusion matrix...")
#    threshold = 0.0  # Seuil pour la classification binaire (0.0 ou 0.5)
#    y_pred_binary = (np.array(y_pred) > threshold).astype(int).flatten()
#    cm = confusion_matrix(np.array(y_true).flatten(), y_pred_binary)
    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    print(os.path.join(working_dir, f"{base_filename}_confusion_matrix.png"))
    plt.savefig(os.path.join(working_dir, f"{base_filename}_confusion_matrix.png"))
    plt.savefig(os.path.join(working_dir, f"{base_filename}_confusion_matrix.png"))


def build_and_train_model(train_loader, val_loader, model, epochs=50, learning_rate=0.001, num_classes=1, regularizers_l2=0, metric_list=['accuracy'], early_stop = False, patience=None, device='cpu', work_dir='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Debugs'):
    """
    Construit et entraîne un modèle de deep learning PyTorch.

    Args:
        train_loader (DataLoader): DataLoader pour les données d'entraînement.
        val_loader (DataLoader): DataLoader pour les données de validation.
        model (nn.Module): Modèle PyTorch à entraîner.
        epochs (int): Nombre d'époques d'entraînement.
        learning_rate (float): Taux d'apprentissage de l'optimiseur.
        loss_fn (nn.Module): Fonction de perte.
        metric_list (list of str): Liste des métriques à calculpytorch_dl_models_classic_CNN.er.
        device (str): L'appareil sur lequel entraîner le modèle ('cpu' ou 'cuda').

    Returns:
        nn.Module: Modèle entraîné.
        dict: Historique de l'entraînement (loss et accuracy).
        list: Prédictions sur l'ensemble de validation.
        list: Vraies étiquettes de l'ensemble de validation.
    """
    is_multiclass = num_classes > 2

    # Fonction de perte par défaut
    loss_fn = nn.CrossEntropyLoss() if is_multiclass else nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularizers_l2)
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    if early_stop:
        early_stopper = EarlyStopping(patience=patience, delta=0, verbose=True)

    model.to(device)

    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        train_corrects, val_corrects = 0, 0
        total_train, total_val = 0, 0
        all_preds, all_labels = [], []

        model.train()
        for batch in train_loader:
            inputs_img = batch['image'].to(device)
            inputs_vec = batch['vector'].to(device)
            labels = batch['label'].to(device)

            if is_multiclass:
                labels = labels.view(-1).long() 
            else:
                labels = labels.float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs_img, inputs_vec)            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs_img.size(0)

            # Prédictions et score
            if is_multiclass:
                preds = outputs.argmax(dim=1)
                train_corrects += (preds == labels).sum().item()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).int()
                train_corrects += (preds == labels.int()).sum().item()

            total_train += labels.size(0)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs_img = batch['image'].to(device)
                inputs_vec = batch['vector'].to(device)
                labels = batch['label'].to(device)

                if is_multiclass:
                    labels = labels.view(-1).long()
                else:
                    labels = labels.float().view(-1, 1)

                outputs = model(inputs_img, inputs_vec)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs_img.size(0)

                if is_multiclass:
                    preds = outputs.argmax(dim=1)
                    val_corrects += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    val_corrects += (preds == labels.int()).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                total_val += labels.size(0)

        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        train_accuracy = train_corrects / total_train
        val_accuracy = val_corrects / total_val

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}')

        if early_stop:
            early_stopper.check_early_stop(avg_val_loss)
            if early_stopper.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

    model.train()
    return model, history, all_preds, all_labels

def perfomances_saving(csv_path, history, model_name, data_augmentation, batch_size, learning_rate, dropout, regularizers_l2, epochs, early_stop, patience):

    # Extraire les performances
    val_acc_history = history.get('val_accuracy', [])
    final_val_acc = val_acc_history[-1] if val_acc_history else None
    max_val_acc = max(val_acc_history) if val_acc_history else None

    # Créer une ligne de résumé
    summary_row = {
        'Model': model_name,
        'Data_Augmentation': data_augmentation,
        'Batch_Size': batch_size,
        'Learning_rate': learning_rate,
        'Dropout': dropout,
        'Dropout_fraction': dropout,  # Même valeur ici pour cohérence
        'Regularizers_L2': regularizers_l2,
        'Number_of_Epoch': epochs,
        'Validation_Accuracy': final_val_acc,
        'Max_Validation_Accuracy_during_Epoch': max_val_acc,
        'EarlyStop' : early_stop,
        'Patience' : patience
        }

    # Charger et ajouter la ligne
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, delimiter=';')
        df_new = pd.DataFrame([summary_row])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([summary_row])

    # Sauvegarder
    df_combined.to_csv(csv_path, sep=';', index=False)

def training_pipeline(pickle_path, model_function, model_name, working_dir, num_classes=1,
                      performance_csv_path=None, data_augmentation=True, epochs=50, 
                      metric_list=['accuracy'], 
                      batch_size=16, random_state=42, dropout=0.5, learning_rate=0.0001, 
                      regularizers_l2=0, early_stop=False, patience=None, perf_save=True, 
                      model_save=True, dataset_index_save=True, device='cpu'):
    """
    Pipeline complet pour charger les données, appliquer l'augmentation de données,
    et entraîner un modèle PyTorch.
    """
    print(f'Model : {model_name} ')
    print('PARAMETERS')
    print(f'data_augmentation : {data_augmentation}')
    print(f'Number of Epoch : {epochs}')
    print(f'Early Stop : {early_stop}')
    print(f'Batch Size : {batch_size}')
    print(f'Learning rate : {learning_rate}')
    print(f'Dropout rate : {dropout}')
    
    X_image_train_np, X_vectors_train_np, y_train_np = load_dataset(pickle_path)
    vector_features_nb = X_vectors_train_np.shape[1:][0]
    input_image_shape = X_image_train_np.shape[1:]
    
    # Ajouter la dimension du canal (assumé être 1 pour des images en niveaux de gris)
    if len(X_image_train_np.shape) == 4:
        X_image_train_np = np.expand_dims(X_image_train_np, axis=1)
    # Suppose une forme dy type (N, C, X, Y, Z) pour les Conv3D de PyTorch
    
    if data_augmentation:
        X_image_train_augmented, X_vectors_train_augmented, y_train_augmented = data_augmentation_function_median_flip(X_image_train_np, X_vectors_train_np, y_train_np, working_dir=working_dir)
        print(f"Shape of X_train after augmentation: {X_image_train_augmented.shape}")
    else:
        X_image_train_augmented = X_image_train_np
        X_vectors_train_augmented = X_vectors_train_np
        y_train_augmented = y_train_np
        print(f"Shape of X_train without augmentation: {X_image_train_augmented.shape}")

    # Division stratifiée train/test
    train_index, test_index = stratified_shuffle_split(y_train_augmented, test_size=0.2, random_state=random_state)
    X_image_train_balanced, X_image_test_balanced  = X_image_train_augmented[train_index], X_image_train_augmented[test_index]
    X_vectors_train_balanced, X_vectors_test_balanced  = X_vectors_train_augmented[train_index], X_vectors_train_augmented[test_index]
    y_train_balanced, y_test_balanced = y_train_augmented[train_index], y_train_augmented[test_index]
    print(f"Y train : [0] : {np.bincount(np.array(y_train_balanced, dtype=int))[0]}; [1] : {np.bincount(np.array(y_train_balanced, dtype=int))[1]} ")
    print(f"Y test : [0] : {np.bincount(np.array(y_test_balanced, dtype=int))[0]}; [1] : {np.bincount(np.array(y_test_balanced, dtype=int))[1]} ")
    
    # Normalisation du dataset
    # Images
    mean_im = X_image_train_balanced.mean()
    std_im = X_image_train_balanced.std()

    train_transform_img = transforms.Compose([transforms.Normalize(mean_im, std_im)])
    val_transform_img = transforms.Compose([transforms.Normalize(mean_im, std_im)])
    
    # vecteur
    mean_vec = X_vectors_train_balanced.mean(axis=0)
    std_vec = X_vectors_train_balanced.std(axis=0)

    train_transform_vec = lambda x: (x - torch.tensor(mean_vec)) / torch.tensor(std_vec)
    val_transform_vec = lambda x: (x - torch.tensor(mean_vec)) / torch.tensor(std_vec)

    # Création des Datasets et DataLoaders PyTorch
    train_dataset = CustomDataset(X_image_train_balanced, X_vectors_train_balanced, y_train_balanced, transform_img=train_transform_img, transform_vec=train_transform_vec)
    val_dataset = CustomDataset(X_image_test_balanced, X_vectors_test_balanced, y_test_balanced, transform_img=val_transform_img, transform_vec=val_transform_vec)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Vérification de la forme des données avant le modèle
    print(f"Image shape of batch before model instantiation: {next(iter(train_loader))['image'].shape}")
    print(f"Vector shape of batch before model instantiation: {next(iter(train_loader))['vector'].shape}")

    input_channels = next(iter(train_loader))['image'].shape[1]
    model = model_function(input_channels, vector_features_nb, input_image_shape, dropout_rate=dropout, num_classes=num_classes).to(device)

    model_trained, history, y_pred, y_true = build_and_train_model(train_loader, val_loader, model, epochs=epochs, learning_rate=learning_rate, num_classes=num_classes, regularizers_l2=regularizers_l2, metric_list=metric_list, early_stop=early_stop, patience=patience, device=device)

    performance_visualizer(history, pickle_path, model_name, epochs, y_true, y_pred, num_classes, working_dir)
    
    if perf_save:
        perfomances_saving(performance_csv_path, history, model_name, data_augmentation, batch_size, learning_rate, dropout, regularizers_l2, epochs, early_stop, patience)
    
    if dataset_index_save:
        # Sauvegarde des indices train/test dans un CSV
        dataset_index_save_path = os.path.join(working_dir, f"{model_name}_test_and_val_indices.csv")
    
        train_index_aug = train_index
        test_index_aug = test_index
    
        if data_augmentation:
            train_index_orig = np.unique(train_index // 2)
            test_index_orig = np.unique(test_index // 2)
        else:
            train_index_orig = train_index
            test_index_orig = test_index
    
        df_indices = pd.DataFrame({
            "train_index_aug": pd.Series(train_index_aug),
            "test_index_aug": pd.Series(test_index_aug),
            "train_index_orig": pd.Series(train_index_orig),
            "test_index_orig": pd.Series(test_index_orig)
        })
    
        df_indices.to_csv(dataset_index_save_path, sep=";", index=False)
        print(f"TRan and Test Indices saved : {dataset_index_save_path}")
    
    if model_save:
        model_save_path = os.path.join(working_dir, f"{model_name}_final_model.pth")
        torch.save(model_trained.state_dict(), model_save_path)
        print(f"Modèl saved : {model_save_path}")

    return model_trained
