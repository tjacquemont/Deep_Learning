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
import torchvision.transforms as transforms
import pandas as pd
import csv
import math
import random
import torch.nn.functional as F

# IMPLEMENTED LOSS
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6, max_weight=10.0, normalize=True):
        super(WeightedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.max_weight = max_weight
        self.normalize = normalize

    def forward(self, predictions, targets):
        # Calcul des poids inverses
        weights = 1.0 / (targets + self.epsilon)
        weights = torch.clamp(weights, max=self.max_weight)

        # Option : normalisation des poids (échelle moyenne constante)
        if self.normalize:
            weights = weights / weights.sum() * targets.numel()

        # Perte MSE pondérée
        loss = weights * (predictions - targets) ** 2
        return loss.mean()

class EWMSELoss(nn.Module):
    def __init__(self, lambda_exp=0.1, max_weight=20.0, epsilon=1e-6):
        super(EWMSELoss, self).__init__()
        self.lambda_exp = lambda_exp
        self.max_weight = max_weight
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        diff = predictions - targets
        abs_diff = torch.abs(diff)

        # Poids exponentiels : plus l'erreur est grande, plus elle est pénalisée
        weights = torch.exp(self.lambda_exp * abs_diff)
        weights = torch.clamp(weights, max=self.max_weight)

        loss = weights * diff**2
        return loss.mean()
    
# EARLY STOPPER    
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
                    
# DATASETS CREATION & DATA AUGMENTATION
# - Dataset creation
def Prepare_dataset(label_name, irm_modalities, vector_features,
                         whole_database_path='/home/thomas.jacquemont/Test/PREP_AVC/New_Datasets_Management/Pickle_dataset/PREP_Whole_Database.pkl',
                         working_directory='/home/thomas.jacquemont/Test/PREP_AVC/New_Datasets_Management/Test',
                         filters=None):
    """
    Prépare des références aux images et vecteurs filtrés.
    Les images normalisées sont conservées dans subj['IRM'] mais ne sont pas toutes
    stockées en mémoire à la fois. On renvoie une liste de dictionnaires pour le lazy loading.

    Args:
        label_name (str): nom de la colonne cible (label)
        irm_modalities (list): modalités IRM à inclure
        vector_features (list): features vectoriels à inclure
        whole_database_path (str): chemin du pickle global
        working_directory (str): répertoire pour logs
        filters (dict): ex. {"Type_Ischemic": lambda x: x==1, "Age": lambda x: x>60}

    Returns:
        tuple: (dataset_entries, selected_patients)
            dataset_entries: list of dicts avec keys: 'imgs' (list of IRM arrays), 'vec' (vector), 'label'
            selected_patients: liste des IDs patients retenus
    """
    with open(whole_database_path, "rb") as f:
        dataset = pickle.load(f)

    dataset_entries = []
    selected_patients = []

    # Préparer les fichiers de log
    log_path = os.path.join(working_directory, 'dataset_log.csv')
    patient_list_path = os.path.join(working_directory, 'patient_list.csv')
    with open(log_path, "w", newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["patient_id", "reason"])  # header

        for subj in dataset:
            pid = subj["id"]

            # --- FILTRE
            if filters is not None:
                skip = False
                for f_key, condition in filters.items():
                    subj_val = subj["VECT"].get(f_key, None)
                    if subj_val is None or not condition(subj_val):
                        log_writer.writerow([pid, f"Filtré par {f_key}"])
                        skip = True
                        break
                if skip:
                    continue

            # --- Label
            label_val = subj["VECT"].get(label_name, None)
            if label_val is None:
                log_writer.writerow([pid, f"Label {label_name} manquant"])
                continue

            # --- IRM
            subj_imgs = []
            for irm in irm_modalities:
                path_in_vect = subj["VECT"].get(irm, None)
                img = subj["IRM"].get(irm, None) if path_in_vect is not None else None
                subj_imgs.append(img)

            if any(img is None for img in subj_imgs):
                missing = [irm for irm, img in zip(irm_modalities, subj_imgs) if img is None]
                log_writer.writerow([pid, f"IRM manquante: {missing}"])
                continue

            # --- VECT
            subj_vect = [subj["VECT"].get(v, None) for v in vector_features]
            if any(val is None for val in subj_vect):
                missing = [v for v, val in zip(vector_features, subj_vect) if val is None]
                log_writer.writerow([pid, f"Vecteurs manquants: {missing}"])
                continue

            # Ajouter au dataset final
            selected_patients.append(pid)
            dataset_entries.append({
                "imgs": subj_imgs,   # liste de np.arrays normalisés
                "vec": np.array(subj_vect, dtype=np.float32),
                "label": float(label_val)
            })

    # Sauvegarde de la liste des patients retenus
    with open(patient_list_path, "w", newline="", encoding="utf-8") as patient_csvfile:
        patient_writer = csv.writer(patient_csvfile)
        patient_writer.writerow(["index", "id"])
        for index, patient_id in enumerate(selected_patients):
            patient_writer.writerow([index, patient_id])

    print(f"✅ Dataset préparé. Log des exclusions dans {log_path}")
    return dataset_entries, selected_patients


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
        if isinstance(X_img, list):
            X_img = np.array(X_img)
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
    

def custom_train_test_split(y, test_size=0.2, random_state=2301, stratify_bins=5):
    """
    Sépare les indices de X en train/test (optionnellement stratifié par quantiles).
    
    Args:
        7 (array-like): tableau (n_samples,)
        test_size (float): proportion de test (entre 0 et 1)
        random_state (int): graine aléatoire
        stratify_bins (array-like): nombre de bin souhaité

    Returns:
        tuple: (train_indices, test_indices)
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(y))
    
    if stratify_bins is None:
        np.random.shuffle(indices)
        split = int(len(y) * (1 - test_size))
        return indices[:split], indices[split:]
    else:
        quantiles = np.percentile(y, np.linspace(0, 100, stratify_bins + 1))
        quantiles = np.unique(quantiles)
        y_bins = np.digitize(y, bins=quantiles, right=True)
        
        unique_bins = np.unique(y_bins)
        train_indices, test_indices = [], []

        for bin_value in unique_bins:
            bin_indices = indices[y_bins == bin_value]
            np.random.shuffle(bin_indices)
            split = int(len(bin_indices) * (1 - test_size))
            train_indices.extend(bin_indices[:split])
            test_indices.extend(bin_indices[split:])

        return np.array(train_indices), np.array(test_indices)


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

# - Data Augmentation
def data_augmentation_function_median_flip(X_img_train, X_vec_train, y_train, working_dir='working_dir', train_id=None):
    """
    Applique un flip le long de l'axe Y (axe 0) pour chaque modalité,
    duplique les sujets et affiche côte à côte l'image originale et flipée
    pour contrôle qualité.
    """
    augmented_X_img = []
    augmented_X_vec = []
    augmented_y = []

    qc_dir = os.path.join(working_dir, "qc_median_slices")
    os.makedirs(qc_dir, exist_ok=True)

    for i in range(len(X_img_train)):
        original_images = X_img_train[i]  # liste de np.array pour chaque modalité
        original_vect = X_vec_train[i]
        original_label = y_train[i]

        # --- Ajouter original ---
        augmented_X_img.append(original_images)
        augmented_X_vec.append(original_vect)
        augmented_y.append(original_label)

        # --- Flip chaque modalité le long de l'axe Y (H) ---
        flipped_images = [np.flip(img, axis=0).copy() for img in original_images]
        augmented_X_img.append(flipped_images)
        augmented_X_vec.append(original_vect)
        augmented_y.append(original_label)

        # --- QC : afficher slice médiane de chaque modalité ---
    
        if train_id:
            pid = train_id[i]
            orig_title = f"Original Subject {pid} - Median Slice (Z)"
            flip_title = f"Flipped Subject {pid} - Median Slice (Z)"
        else :
            orig_title = f"Original Subject {i} - Median Slice (Z)"
            flip_title = f"Flipped Subject {i} - Median Slice (Z)"
            
        for m in range(len(original_images)):
            orig_slice = original_images[m][:, :, original_images[0].shape[2] // 2]
            flipped_slice = flipped_images[m][:, :, flipped_images[0].shape[2] // 2]

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_slice, cmap='gray')
            plt.title(f'RMI Modality {m} ' + orig_title)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(flipped_slice, cmap='gray')
            plt.title(f'RMI Modality {m} ' + flip_title)
            plt.axis('off')

            plt.savefig(os.path.join(qc_dir, f"RMI_Modality_{m}_subject_{i}_median_slices.png"))
            plt.close()

    return augmented_X_img, np.array(augmented_X_vec), np.array(augmented_y)

# ------------------ 3D TRANSFORMS ------------------
class Compose3D:
    def __init__(self, transforms_list):
        self.transforms = transforms_list


    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor3D:
    """Ensure input is torch.FloatTensor with shape (C,H,W,D)"""
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x.float()


class Normalize3D:
    """Normalize a 3D volume. mean/std can be scalars or per-channel arrays."""
    def __init__(self, mean, std, eps=1e-8):
        self.mean = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        self.std = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std
        self.eps = eps


    def __call__(self, x):
        # x: (C,H,W,D)
        if self.mean.numel() == 1:
            return (x - float(self.mean)) / (float(self.std) + self.eps)
        else:
            mean = self.mean.view(-1, 1, 1, 1)
        std = self.std.view(-1, 1, 1, 1)
        return (x - mean) / (std + self.eps)


class RandomFlip3D:
    """Random flip along any spatial axis with given probability."""
    def __init__(self, p=0.5, axes=('x','y','z')):
        self.p = p
        self.axes = axes
        

    def __call__(self, x):
        # x: (C,H,W,D)
        if random.random() < self.p:
            for ax in self.axes:
                if random.random() < 0.5:
                    if ax == 'x':
                        x = torch.flip(x, dims=[1])
                    elif ax == 'y':
                        x = torch.flip(x, dims=[2])
                    elif ax == 'z':
                        x = torch.flip(x, dims=[3])
        return x


class RandomIntensityShift:
    """Random multiplicative + additive intensity transform per-channel."""
    def __init__(self, add_range=0.1, mult_range=0.1, p=0.5):
        self.add_range = add_range
        self.mult_range = mult_range
        self.p = p


def __call__(self, x):
    if random.random() < self.p:
        C = x.shape[0]
        adds = torch.randn(C) * self.add_range
        mults = 1.0 + (torch.randn(C) * self.mult_range)
        adds = adds.view(-1,1,1,1)
        mults = mults.view(-1,1,1,1)
        x = x * mults + adds
    return x


class RandomGaussianNoise:
    def __init__(self, std_range=(0.0, 0.05), p=0.5):
        self.std_range = std_range
        self.p = p


    def __call__(self, x):
        if random.random() < self.p:
            std = random.uniform(self.std_range[0], self.std_range[1])
            noise = torch.randn_like(x) * std
            x = x + noise
        return x


class RandomCrop3D:
    def __init__(self, out_size):
        # out_size: (H_new, W_new, D_new)
        self.out_size = out_size


    def __call__(self, x):
        # x: (C,H,W,D)
        C, H, W, D = x.shape
        h_new, w_new, d_new = self.out_size
        if H == h_new and W == w_new and D == d_new:
            return x
        if H < h_new or W < w_new or D < d_new:
            # fallback to center crop + pad
            return center_crop_and_pad(x, self.out_size)
        h1 = random.randint(0, H - h_new)
        w1 = random.randint(0, W - w_new)
        d1 = random.randint(0, D - d_new)
        return x[:, h1:h1+h_new, w1:w1+w_new, d1:d1+d1+d_new]


def center_crop_and_pad(x, out_size):
     C, H, W, D = x.shape
     h_new, w_new, d_new = out_size
     # crop or pad each dim
     result = torch.zeros((C, h_new, w_new, d_new), dtype=x.dtype, device=x.device)
     h_start = max((h_new - H)//2, 0)
     w_start = max((w_new - W)//2, 0)
     d_start = max((d_new - D)//2, 0)


     h_src_start = max((H - h_new)//2, 0)
     w_src_start = max((W - w_new)//2, 0)
     d_src_start = max((D - d_new)//2, 0)
    

     h_copy = min(H, h_new)
     w_copy = min(W, w_new)
     d_copy = min(D, d_new)


     result[:, h_start:h_start+h_copy, w_start:w_start+w_copy, d_start:d_start+d_copy] = \
        x[:, h_src_start:h_src_start+h_copy, w_src_start:w_src_start+w_copy, d_src_start:d_src_start+d_copy]
     return result


class Resize3D:
    def __init__(self, size):
        # size: (H_new, W_new, D_new)
        self.size = size


    def __call__(self, x):
        # x: (C,H,W,D) -> use interpolate: input must be (N,C,D,H,W)
        tensor = x.unsqueeze(0) # (1,C,H,W,D)
        # permute to (1,C,D,H,W)
        tensor = tensor.permute(0,1,4,2,3)
        d_new, h_new, w_new = self.size[2], self.size[0], self.size[1]
        out = F.interpolate(tensor, size=(d_new, h_new, w_new), mode='trilinear', align_corners=False)
        # back to (C,H,W,D)
        out = out.permute(0,1,3,4,2).squeeze(0)
        return out


class RandomAffine3D:
    """Apply a random 3D affine transformation using grid_sample.
    rotation_range: degrees for each axis (rx,ry,rz)
    translation: fraction of volume size
    scale_range: (min, max)
    """
    def __init__(self, rotation_range=(10,10,10), translation_frac=0.05, scale_range=(0.95,1.05), p=0.5):
        self.rotation_range = rotation_range
        self.translation_frac = translation_frac
        self.scale_range = scale_range
        self.p = p


    def __call__(self, x):
        if random.random() >= self.p:
            return x
        # x: (C,H,W,D)
        C,H,W,D = x.shape
        # random rotations in radians
        rx = math.radians(random.uniform(-self.rotation_range[0], self.rotation_range[0]))
        ry = math.radians(random.uniform(-self.rotation_range[1], self.rotation_range[1]))
        rz = math.radians(random.uniform(-self.rotation_range[2], self.rotation_range[2]))


        # rotation matrices
        Rx = torch.tensor([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]], dtype=torch.float32)
        Ry = torch.tensor([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]], dtype=torch.float32)
        Rz = torch.tensor([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]], dtype=torch.float32)
        
        R = Rx @ Ry @ Rz


        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        S = torch.eye(3) * scale

        M = S @ R


        # translation normalized to [-1,1]
        tx = random.uniform(-self.translation_frac, self.translation_frac)
        ty = random.uniform(-self.translation_frac, self.translation_frac)
        tz = random.uniform(-self.translation_frac, self.translation_frac)
        # assemble theta (3x4)
        theta = torch.zeros((1,3,4), dtype=torch.float32)
        theta[0,:,:3] = M
        theta[0,:,3] = torch.tensor([tx, ty, tz], dtype=torch.float32)


        # prepare input for grid_sample: (N,C,D,H,W)
        inp = x.unsqueeze(0).permute(0,1,3,2,4) # from (C,H,W,D) -> (1,C,D,H,W)
        grid = F.affine_grid(theta, size=inp.size(), align_corners=False)
        out = F.grid_sample(inp, grid, padding_mode='border', align_corners=False)
        # back to (C,H,W,D)
        out = out.permute(0,1,3,2,4).squeeze(0)
        return out

# Tranformation function :
def build_image_transforms(mean_im, std_im, target_size=None, augment=True):
    # mean_im/std_im can be scalar or per-channel arrays
    norm = Normalize3D(mean_im, std_im)
    if augment:
        aug = Compose3D([
            ToTensor3D(),
            RandomFlip3D(p=0.8),
            RandomAffine3D(rotation_range=(10,10,10), translation_frac=0.05, scale_range=(0.95,1.05), p=0.6),
            RandomIntensityShift(add_range=0.05, mult_range=0.05, p=0.5),
            RandomGaussianNoise(std_range=(0.0, 0.03), p=0.5),
            ])
        if target_size is not None:
            aug.transforms.insert(3, Resize3D(target_size)) # after affine
        # final normalization
        return Compose3D([aug, norm])
    else:
        seq = [ToTensor3D()]
        if target_size is not None:
            seq.append(Resize3D(target_size))
        seq.append(norm)
        return Compose3D(seq)

# LOSS METRICS
def compute_metrics(y_true, y_pred, metric_list):
    results = {}
    
    if 'mae' in metric_list:
        results['mae'] = np.mean(np.abs(y_true - y_pred))
    if 'rmse' in metric_list:
        results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if 'r2' in metric_list:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        results['r2'] = 1 - ss_res / ss_tot
    return results


# MODEL TRAINING AND EVALUATION


def Prediction_VS_Ground_Truth(y_true, y_pred, working_dir):
    # Plotting Prediction vs Ground Truth
    plt.figure(figsize=(6,6))  
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.title('Prediction vs Ground Truth')
    plt.grid(True)
    print(os.path.join(working_dir, f"Prediction_vs_Ground_Truth.png"))
    plt.savefig(os.path.join(working_dir, f"Prediction_vs_Ground_Truth.png"))
    plt.close()
    

def build_and_train_model(train_loader, val_loader, model, epochs=50, learning_rate=0.001,
                          delay_save=100, loss_fn=nn.MSELoss(), metric_list=['mae', 'rmse', 'r2'],
                          best_metrics_name='r2', regularizers_l2=0, early_stop=False, patience=None, 
                          device='cpu', model_save=True, work_dir='/path/to/save'):
    """
    Entraîne un modèle PyTorch pour la régression et retourne les métriques demandées.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularizers_l2)
    history = {'train_loss': [], 'val_loss': []}
    for metric in metric_list:
        history[f'train_{metric}'] = []
        history[f'val_{metric}'] = []

    if early_stop:
        early_stopper = EarlyStopping(patience=patience, delta=0, verbose=True)

    model.to(device)
    
    model_save_thrs = False
    best_metrics = None

    for epoch in range(epochs):
        if not model_save_thrs and epoch>delay_save:
            model_save_thrs=True
        model.train()
        train_loss = 0.0
        total_train = 0
        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            inputs_img = batch['image'].to(device)
            inputs_vec = batch['vector'].to(device)
            labels = batch['label'].float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs_img, inputs_vec)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            total_train += labels.size(0)
            
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / total_train
        history['train_loss'].append(avg_train_loss)
        
        train_metrics = compute_metrics(np.array(all_train_labels), np.array(all_train_preds), metric_list)
        for metric_name, metric_value in train_metrics.items():
            history[f'train_{metric_name}'].append(metric_value)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total_val = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs_img = batch['image'].to(device)
                inputs_vec = batch['vector'].to(device)
                labels = batch['label'].float().to(device)

                outputs = model(inputs_img, inputs_vec)

                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                total_val += labels.size(0)

                all_val_preds.extend(outputs.detach().cpu().numpy())
                all_val_labels.extend(labels.detach().cpu().numpy())

        avg_val_loss = val_loss / total_val
        history['val_loss'].append(avg_val_loss)

        val_metrics = compute_metrics(np.array(all_val_labels), np.array(all_val_preds), metric_list)
        for metric_name, metric_value in val_metrics.items():
            history[f'val_{metric_name}'].append(metric_value)
        
        # --- Model evauation and save ---
        if model_save_thrs:
            if best_metrics :
                if val_metrics[best_metrics_name]>best_metrics:
                    best_metrics = val_metrics[best_metrics_name]
                    Prediction_VS_Ground_Truth(all_val_labels, all_val_preds, work_dir)
                    if model_save:
                        torch.save(model.state_dict(), os.path.join(work_dir, f"best_model.pth"))
            else :
                best_metrics = val_metrics[best_metrics_name]
                Prediction_VS_Ground_Truth(all_val_labels, all_val_preds, work_dir)
                if model_save:
                    torch.save(model.state_dict(), os.path.join(work_dir, f"best_model.pth"))

        # --- Logging ---
        metrics_train_log = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        metrics_val_log = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        print(f"Epoch [{epoch+1}/{epochs}] ")
        print(f"Train Loss: {avg_train_loss:.4f} - {metrics_train_log}")
        print(f"Val Loss: {avg_val_loss:.4f} - {metrics_val_log}")

        if early_stop:
            early_stopper.check_early_stop(avg_val_loss)
            if early_stopper.stop_training:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if not model_save_thrs:
        Prediction_VS_Ground_Truth(all_val_labels, all_val_preds, work_dir)
        if model_save:
            torch.save(model.state_dict(), os.path.join(work_dir, f"best_model.pth"))
    
    model.train()
    
    return model, history


def training_visualizer(history, dataset_path, model_name, epochs, working_dir):
    """
    Visualise et enregistre les courbes d'apprentissage et les métriques d'évaluation.
    """

    print(f"Working directory: {working_dir}")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        print(f"Created working directory: {working_dir}")

    dataset_name = dataset_path.split("/")[-1].split('.')[0]
    base_filename = f"{dataset_name}_{model_name}_{epochs}"
    print(f'base_filename : {base_filename}')

    # Courbes de toutes les métriques disponibles
    for key in history:
        if key.startswith("train_"):
            metric = key.split('_', 1)[1]  
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"

            if train_key in history and val_key in history:
                plt.figure(figsize=(10, 5))
                plt.plot(history[train_key], label=f'train_{metric}')
                plt.plot(history[val_key], label=f'val_{metric}')
                plt.title(f'{metric.capitalize()} vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel(metric.upper())
                plt.legend(loc='best')
                plt.grid(True)

                save_path = os.path.join(working_dir, f"{base_filename}_{metric}.png")
                print(f"Saving plot: {save_path}")
                plt.savefig(save_path)
                plt.close()


def perfomances_saving(csv_path, history, model_name, data_augmentation, batch_size,
                       learning_rate, dropout, regularizers_l2, epochs, early_stop,
                       patience, loss_fn):
    # Gestion du booléen pour dropout
    dropout_bin = bool(dropout)

    # Résumé des hyperparamètres
    summary_row = {
        'Model': model_name,
        'Data_Augmentation': data_augmentation,
        'Loss' :  str(loss_fn),
        'Batch_Size': batch_size,
        'Learning_rate': learning_rate,
        'Dropout': dropout_bin,
        'Dropout_fraction': dropout,
        'Regularizers_L2': regularizers_l2,
        'Number_of_Epoch': epochs,
        'EarlyStop' : early_stop,
        'Patience' : patience,
        }

    # Métriques où le meilleur est le maximum
    max_metrics = ['r2', 'R2', 'r2_score']

    for key, values in history.items():
        if key.startswith('val_'):
            metric_name = key.replace('val_', '')
            final_val = values[-1] if values else None

            if metric_name.lower() in max_metrics:
                best_val = max(values) if values else None
            else:
                best_val = min(values) if values else None

            summary_row[f'Final_Val_{metric_name}'] = final_val
            summary_row[f'Best_Val_{metric_name}'] = best_val

    # Chargement ou création du DataFrame
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, delimiter=';')
        df_new = pd.DataFrame([summary_row])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([summary_row])

    df_combined.to_csv(csv_path, sep=';', index=False)
    
###############################################################################
def training_pipeline(
    whole_dataset_path, label_name, irm_modalities, vector_features, filters,
    model_function, model_name, working_dir, num_classes=1, epochs=50,
    performance_csv_path=None, flip_augmentation=True, data_augmentation=True, 
    metric_list=['mae', 'rmse', 'r2'], loss_fn=nn.MSELoss(), batch_size=16,
    random_state=2301, dropout=0.5, learning_rate=0.0001, regularizers_l2=0,
    early_stop=False, patience=None, perf_save=True, dataset_index_save=True,
    model_save=True, history_save=True, device='cpu'
):
    print(f"Model : {model_name}")
    print(f"Label : {label_name}")
    print(f"IRM used : {irm_modalities}")
    print(f"Vector features : {vector_features}")
    print(f"Data augmentation : {data_augmentation}")
    print(f"Epochs : {epochs}, Batch size : {batch_size}, LR : {learning_rate}, Dropout : {dropout}")
    print(f"Dataset : {whole_dataset_path} ")

    # --- 1. Chargement des données ---
    dataset_entries, selected_patients = Prepare_dataset(
        label_name, irm_modalities, vector_features,
        whole_database_path=whole_dataset_path,
        working_directory=working_dir, filters=filters
    )
    
    # X_img sera une liste de liste vide si irm_modalities=[]
    X_img = [entry["imgs"] for entry in dataset_entries]
    X_vec = np.array([entry["vec"] for entry in dataset_entries], dtype=np.float32)
    y = np.array([entry["label"] for entry in dataset_entries], dtype=np.float32)

    print(f"Nombre de patients : {len(X_img)}")
    print(f"Taille du vecteur d'un patient : {X_vec.shape[1]}")
    print(f"Exemple de labels : {y[:5]}")

    vector_features_nb = X_vec.shape[1]
    if len(X_img[0]) > 0:
        image_include_in_model = True
        input_image_shape = np.stack(X_img[0], axis=0).shape  # (channels, H, W, D)
    else:
        image_include_in_model = False
        input_image_shape = [np.nan]

    # --- 2. Split train/test ---
    train_index, test_index = custom_train_test_split(y, test_size=0.2, random_state=random_state)
    X_image_train = [X_img[i] for i in train_index]
    X_image_test = [X_img[i] for i in test_index]
    X_vectors_train = X_vec[train_index]
    X_vectors_test = X_vec[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    train_id = [selected_patients[i] for i in train_index]
    
    print(f'Train index : {train_index}')

    # --- 3. Flip augmentation ---
    if flip_augmentation and image_include_in_model:
        X_img_train, X_vec_train, y_train = data_augmentation_function_median_flip(
            X_image_train, X_vectors_train, y_train, working_dir=working_dir, train_id=train_id
            )
    else:
        X_img_train = X_image_train
        X_vec_train = X_vectors_train
        
    print(f"Train size after augmentation: {len(X_image_train)}")

    # --- 4. Normalisation ---
    # Pour les images : calculer mean/std sur toutes les images empilées par batch
    all_train_imgs = np.concatenate([np.stack(imgs, axis=0) for imgs in X_image_train], axis=0)
    mean_im, std_im = all_train_imgs.mean(), all_train_imgs.std()
    normalize_img = transforms.Normalize(mean_im, std_im)

    # Pour les vecteurs
    mean_vec, std_vec = X_vectors_train.mean(axis=0), X_vectors_train.std(axis=0)
    normalize_vec = lambda x: (x - torch.tensor(mean_vec)) / torch.tensor(std_vec)

   # --- 5. 3D Data Augmentation parameters ---   
    if data_augmentation and image_include_in_model:
        train_transform_img = transforms.Compose([
            RandomCrop3D((96, 96, 96)),
            RandomAffine3D(rotation_range=(10,10,10),
                           scale_range=(0.9, 1.1),
                           translation_frac=0.1),
                            RandomFlip3D(p=0.5),
                            Resize3D((96, 96, 96)),
                            normalize_img
                            ])
    else:
        train_transform_img =  transforms.Compose([
                                                Resize3D((96, 96, 96)),
                                                normalize_img
                                                ])

    val_transform_img = transforms.Compose([
                                            Resize3D((96, 96, 96)),
                                            normalize_img
                                            ])

    # --- 6. DataLoaders ---
    train_dataset = CustomDataset(X_img_train, X_vec_train, y_train,
                                  transform_img=train_transform_img, transform_vec=normalize_vec)
    val_dataset = CustomDataset(X_image_test, X_vectors_test, y_test,
                                transform_img=val_transform_img, transform_vec=normalize_vec)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 6. Initialisation du modèle ---
    input_channels = input_image_shape[0]
    model = model_function(input_channels, vector_features_nb, input_image_shape, dropout_rate=dropout).to(device)

    # --- 7. Entraînement ---
    model_trained, history = build_and_train_model(
        train_loader, val_loader, model,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        regularizers_l2=regularizers_l2,
        metric_list=metric_list,
        early_stop=early_stop,
        patience=patience,
        model_save=model_save,
        work_dir=working_dir,
        device=device
    )

    # --- 8. Visualisation ---
    training_visualizer(history, "whole_dataset", model_name, epochs, working_dir)

    # --- 9. Sauvegardes ---
    if performance_csv_path and perf_save:
        perfomances_saving(performance_csv_path, history, model_name, data_augmentation,
                           batch_size, learning_rate, dropout, regularizers_l2,
                           epochs, early_stop, patience, loss_fn)

    if dataset_index_save:
       df_indices = pd.DataFrame({
               "set": ["train"]*len(train_index) + ["test"]*len(test_index),
               "index": np.concatenate([train_index, test_index])                         })
       df_indices.to_csv(os.path.join(working_dir, f"{model_name}_indices.csv"), index=False, sep=";")

    if history_save:
        pd.DataFrame(history).to_csv(os.path.join(working_dir, f"{model_name}_history.csv"), sep=";", index=False)

    return model_trained
