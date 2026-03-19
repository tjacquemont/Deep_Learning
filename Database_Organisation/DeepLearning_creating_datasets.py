#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:16:09 2025

@author: thomas.jacquemont
"""
import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle

def create_dataset_deep_learning_RMI_only(motor_score, database_path, Datasets_path, coupe_path=None):
    """
    Crée un dataset pour le deep learning à partir d'un fichier CSV,
    sauvegarde le dataset et génère des listes d'erreurs.
    Enregistre également une coupe médiane de chaque image pour le contrôle qualité,
    en incluant le numéro du patient dans le nom du fichier.
    """

    df = pd.read_csv(database_path, sep=';')
    df_selection = df.dropna(subset=['IRM_Diffusion_Norm_path', motor_score])

    irm_data = []
    labels_conserves = []
    patients_conserves = []
    patients_erreurs_id = []

    for index, row in df_selection.iterrows(): # Itération sur les lignes de df_selection
        try:
            path = row['IRM_Diffusion_Norm_path']
            label = row[motor_score]
            numero_patient = row['numero']

            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)

            # Vérification si les données sont vides
            if data.size == 0:
                print(f"Erreur : Données IRM vides pour {path}")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification de la forme des données
            if len(data.shape) < 3:
                print(f"Erreur : Forme des données IRM incorrecte pour {path}")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification des valeurs min et max
            if np.isnan(np.nanmin(data)) and np.isnan(np.nanmax(data)):
                print(f"Erreur: l'image {path} contient uniquement des valeurs NaN")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification des valeurs infinies ou non numériques
            if np.any(np.isinf(data)) or np.any(np.isnan(data)):
                print(f"Avertissement : l'image {path} contient des valeurs infinies ou NaN")
                data[np.isinf(data)] = 0
                data[np.isnan(data)] = 0

            # Normalisation
            mean = np.nanmean(data)
            std = np.nanstd(data)
            if std == 0 or np.isnan(std):
                print(f"Avertissement : Écart-type nul ou NaN pour {path}, normalisation ignorée")
            else:
                data = (data - mean) / std

            irm_data.append(data)
            labels_conserves.append(label)
            patients_conserves.append(row)

            # Extraction et sauvegarde de la coupe médiane avec le numéro du patient
            if coupe_path is not None:
                coupe = data[:, :, data.shape[2] // 2]  # Coupe médiane selon l'axe Z
                nom_fichier = f"{numero_patient}_coupe.png"
                chemin_coupe = os.path.join(coupe_path, nom_fichier)
                plt.imshow(coupe, cmap='gray')
                plt.savefig(chemin_coupe)
                plt.close()

        except FileNotFoundError:
            print(f"Fichier IRM introuvable : {path}")
            patients_erreurs_id.append(numero_patient)
        except Exception as e:
            print(f"Erreur lors du chargement de l'IRM {path}: {e}")
            patients_erreurs_id.append(numero_patient)

    irm_data = np.array(irm_data)
    labels_conserves = np.array(labels_conserves)

    with open(Datasets_path, 'wb') as f:
        pickle.dump((irm_data, labels_conserves), f)

    return irm_data, labels_conserves, patients_conserves, patients_erreurs_id


def create_dataset_deep_learning_mixte_RMI_vectors(motor_score, vect_col_list, database_path, Datasets_path, coupe_path=None):
    """
    Crée un dataset pour le deep learning à partir d'un fichier CSV,
    sauvegarde le dataset et génère des listes d'erreurs.
    Enregistre également une coupe médiane de chaque image pour le contrôle qualité,
    en incluant le numéro du patient dans le nom du fichier.
    """

    df = pd.read_csv(database_path, sep=';')
    df_selection = df.dropna(subset=['IRM_Diffusion_Norm_path', motor_score]+vect_col_list)
    
    vect_data = []
    irm_data = []
    labels_conserves = []
    patients_conserves = []
    patients_erreurs_id = []

    for index, row in df_selection.iterrows(): # Itération sur les lignes de df_selection
        try:
            path = row['IRM_Diffusion_Norm_path']
            label = row[motor_score]
            numero_patient = row['numero']
            vecteur_values = row[vect_col_list].values.astype(np.float32)

            img = nib.load(path)
            im_data = img.get_fdata().astype(np.float32)

            # Vérification si les données sont vides
            if im_data.size == 0:
                print(f"Erreur : Données IRM vides pour {path}")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification de la forme des données
            if len(im_data.shape) < 3:
                print(f"Erreur : Forme des données IRM incorrecte pour {path}")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification des valeurs min et max
            if np.isnan(np.nanmin(im_data)) and np.isnan(np.nanmax(im_data)):
                print(f"Erreur: l'image {path} contient uniquement des valeurs NaN")
                patients_erreurs_id.append(numero_patient)
                continue

            # Vérification des valeurs infinies ou non numériques
            if np.any(np.isinf(im_data)) or np.any(np.isnan(im_data)):
                print(f"Avertissement : l'image {path} contient des valeurs infinies ou NaN")
                im_data[np.isinf(im_data)] = 0
                im_data[np.isnan(im_data)] = 0

            # Normalisation
            im_mean = np.nanmean(im_data)
            im_std = np.nanstd(im_data)
            if im_std == 0 or np.isnan(im_std):
                print(f"Avertissement : Écart-type nul ou NaN pour {path}, normalisation ignorée")
            else:
                im_data = (im_data - im_mean) / im_std
                
            # Vérification de la validité du vecteur
            if np.any(pd.isnull(vecteur_values)) or np.any(np.isinf(vecteur_values)):
                print(f"Vecteur clinique invalide pour {numero_patient}")   
                patients_erreurs_id.append(numero_patient)
                continue
            
            vect_data.append(vecteur_values)
            irm_data.append(im_data)
            labels_conserves.append(label)
            patients_conserves.append(row)

            # Extraction et sauvegarde de la coupe médiane avec le numéro du patient
            if coupe_path is not None:
                coupe = im_data[:, :, im_data.shape[2] // 2]  # Coupe médiane selon l'axe Z
                nom_fichier = f"{numero_patient}_coupe.png"
                chemin_coupe = os.path.join(coupe_path, nom_fichier)
                plt.imshow(coupe, cmap='gray')
                plt.savefig(chemin_coupe)
                plt.close()

        except FileNotFoundError:
            print(f"Fichier IRM introuvable : {path}")
            patients_erreurs_id.append(numero_patient)
        except Exception as e:
            print(f"Erreur lors du chargement de l'IRM {path}: {e}")
            patients_erreurs_id.append(numero_patient)

    irm_data = np.array(irm_data)
    vect_data = np.array(vect_data)
    labels_conserves = np.array(labels_conserves)

    with open(Datasets_path, 'wb') as f:
        pickle.dump((irm_data, vect_data, labels_conserves), f)

    return irm_data, vect_data, labels_conserves, patients_conserves, patients_erreurs_id

######################################################################################################################################################################
# Creating Datasets RMI Only
PREP_database_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
Datasets_path_arat_bin = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_arat_binaire.pkl'
Datasets_path_FM_bin = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fm_binaire.pkl'
Datasets_path_FMprop_bin = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fmprop_binaire.pkl'
Datasets_path_arat = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_arat.pkl'
Datasets_path_FM = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fm.pkl'
Datasets_path_FMprop = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Dataset_diffusion_normalisee_fmprop.pkl'
Liste_score_moteur = ['Categorie_M3_ARAT_binaire', 'Categorie_M3_FM_binaire', 'FM_M3_J7_Max_Recovery_ratio_binaire', 'ARAT_imput_M3', 'FM_imput_M3', 'FM_M3_J7_Max_Recovery_ratio']
Normalisation_quality_checking_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/RMI_Only/Quality_checking'
######################################################################################################################################################################

# ARAT 
# binaire

X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('Categorie_M3_ARAT_binaire', PREP_database_path, Datasets_path_arat_bin, coupe_path=Normalisation_quality_checking_directory)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_arat_bin, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)

# Regression
X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('ARAT_imput_M3', PREP_database_path, Datasets_path_arat, coupe_path=Normalisation_quality_checking_directory)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_arat, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)
    
# FM 
# binaire

X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('Categorie_M3_FM_binaire', PREP_database_path, Datasets_path_FM_bin)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_FM_bin, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)

# Regression
X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('FM_imput_M3', PREP_database_path, Datasets_path_FM)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_FM, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)
    
    
# FM proportionnel 
# binaire

X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('FM_M3_J7_Max_Recovery_ratio_binaire', PREP_database_path, Datasets_path_FMprop_bin)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_FMprop_bin, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)

# Regression
X, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_RMI_only('FM_M3_J7_Max_Recovery_ratio', PREP_database_path, Datasets_path_FMprop)

if X is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_path_FMprop, 'rb') as f:
        X_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X chargé shape:", X_charge.shape)
    print("y chargé shape:", y_charge.shape)


######################################################################################################################################################################
# Creating Datasets Mixt RMI_Vectors
PREP_database_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
Normalisation_quality_checking_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Quality_checking'
# J3
vect_col_list_J3 = ['Age', 'SAFE_J3', 'NIHSS_J3']
Datasets_mixt_path_arat_binaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_arat_binaire.pkl'
Datasets_mixt_path_FM_binaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fm_binaire.pkl'
Datasets_mixt_path_FMprop_binaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fmprop_binaire.pkl'
#------------
Datasets_mixt_path_arat_quaternaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_arat_quaternaire.pkl'
Datasets_mixt_path_FM_quaternaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fm_quaternaire.pkl'
Datasets_mixt_path_FMprop_quaternaire_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fmprop_quaternaire.pkl'
#------------
Datasets_mixt_path_arat_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_arat.pkl'
Datasets_mixt_path_FM_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fm.pkl'
Datasets_mixt_path_FMprop_J3 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J3_fmprop.pkl'
# J7
vect_col_list_J7 = ['Age', 'SAFE_J7', 'NIHSS_J7', 'PEM_plus1_ipsi', 'FM_J7']
Datasets_mixt_path_arat_binaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_arat_binaire.pkl'
Datasets_mixt_path_FM_binaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fm_binaire.pkl'
Datasets_mixte_path_FMprop_binaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fmprop_binaire.pkl'
#------------
Datasets_mixt_path_arat_quaternaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_arat_quaternaire.pkl'
Datasets_mixt_path_FM_quaternaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fm_quaternaire.pkl'
Datasets_mixt_path_FMprop_quaternaire_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fmprop_quaternaire.pkl'
#------------
Datasets_mixt_path_arat_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_arat.pkl'
Datasets_mixt_path_FM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fm.pkl'
Datasets_mixte_path_FMprop_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_fmprop.pkl'

# J7 without PEM
vect_col_list_wo_PEM_J7 = ['Age', 'SAFE_J7', 'NIHSS_J7', 'FM_J7']
Datasets_mixt_path_arat_binaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_arat_binaire.pkl'
Datasets_mixt_path_FM_binaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fm_binaire.pkl'
Datasets_mixte_path_FMprop_binaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fmprop_binaire.pkl'
#------------
Datasets_mixt_path_arat_quaternaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_arat_quaternaire.pkl'
Datasets_mixt_path_FM_quaternaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fm_quaternaire.pkl'
Datasets_mixt_path_FMprop_quaternaire_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fmprop_quaternaire.pkl'
#------------
Datasets_mixt_path_arat_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_arat.pkl'
Datasets_mixt_path_FM_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fm.pkl'
Datasets_mixte_path_FMprop_wo_PEM_J7 = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Datasets/Mixte_RMI_Vectors/Dataset_diffusion_normalisee_vect_J7_wo_PEM_fmprop.pkl'

######################################################################################################################################################################

# J3
# ARAT binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_binaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_arat_binaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_binaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_equal_quaternaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_arat_quaternaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_quaternaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_FM_binaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FM_binaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_binaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_imput_M3_equal_quaternaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FM_quaternaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_quaternaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop binaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio_binaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FMprop_binaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_binaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FMprop quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FMprop_quaternaire_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_quaternaire_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# ARAT

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('ARAT_imput_M3', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_arat_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FM

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_imput_M3', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FM_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio', vect_col_list_J3, PREP_database_path, Datasets_mixt_path_FMprop_J3, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_J3, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# J7 
# ARAT binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_binaire', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_arat_binaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_binaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_equal_quaternaire', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_arat_quaternaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_quaternaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_FM_binaire', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_FM_binaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_binaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_imput_M3_equal_quaternaire', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_FM_quaternaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_quaternaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop binaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio_binaire', vect_col_list_J7, PREP_database_path, Datasets_mixte_path_FMprop_binaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_binaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FMprop quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_FMprop_quaternaire_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_quaternaire_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('ARAT_imput_M3', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_arat_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FM

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_imput_M3', vect_col_list_J7, PREP_database_path, Datasets_mixt_path_FM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio', vect_col_list_J7, PREP_database_path, Datasets_mixte_path_FMprop_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)
    
    
# J7 Without PEM 
# ARAT binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_FM_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_imput_M3_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop binaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixte_path_FMprop_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FMprop quaternaire# J7 Without PEM 
# ARAT binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_ARAT_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM binaire

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_M3_FM_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FM quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_imput_M3_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop binaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio_binaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixte_path_FMprop_binaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_binaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FMprop quaternaire
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FMprop_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('ARAT_imput_M3', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FM

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_imput_M3', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixte_path_FMprop_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FMprop_quaternaire_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FMprop_quaternaire_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# ARAT

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('ARAT_imput_M3', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_arat_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_arat_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)


# FM

X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_imput_M3', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixt_path_FM_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixt_path_FM_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_vector chargé shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)

# FMprop
X_rmi, X_vector, y, patients_conserves, patients_erreurs_id = create_dataset_deep_learning_mixte_RMI_vectors('FM_M3_J7_Max_Recovery_ratio', vect_col_list_wo_PEM_J7, PREP_database_path, Datasets_mixte_path_FMprop_wo_PEM_J7, coupe_path=Normalisation_quality_checking_directory)

if X_rmi is not None and X_vector is not None:
    print("Dataset complet créé et sauvegardé avec succès !")
    print("Nombre de patients conservés :", len(patients_conserves))
    print("Nombre de patients avec erreurs de chargement :", len(patients_erreurs_id))

    if patients_erreurs_id:
        print("\nIdentifiants des patients avec erreurs de chargement d'IRM :")
        print(patients_erreurs_id)
    else:
        print("\nAucune erreur de chargement d'IRM.")

    # Chargement du dataset complet sauvegardé (exemple)
    with open(Datasets_mixte_path_FMprop_wo_PEM_J7, 'rb') as f:
        X_rmi_charge, X_vector_charge, y_charge = pickle.load(f)

    print("\nDataset complet chargé avec succès depuis le fichier sauvegardé.")
    print("X_rmi chargé shape:", X_rmi_charge.shape)
    print("X_rmi vector shape:", X_vector_charge.shape)
    print("y chargé shape:", y_charge.shape)
    
#############################################################################################################################################################################################################"
# Creating Whole Datasets
PREP_database_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
Vect_col_list = ['Age','Sexe','Taille_cm','IRM_Diffusion_Raw','IRM_Diffusion_Norm_to_MNI', 'IRM_B0_Raw', 'IRM_B0_Norm_to_MNI', 'Lesion_Mask_Norm_to_MNI', 'ADC_Raw', 'ADC_Norm_to_MNI', 'Lateralite_D','Type_Ischemic','COTE_D','TIV','TM','Delai_AVC_NIHSS_J3','NIHSS_J3','NIHSS_MS_J3','NIHSS_MI_J3','NIHSS_neg_J3','NIHSS_apha_J3','delai_AVC_SAFE_J3','SAFE_J3','SA_J3','FE_J3','Delai_AVC_NIHSS_J7','NIHSS_J7','NIHSS_MS_J7','NIHSS_MI_J7','NIHSS_neg_J7','NIHSS_apha_J7','delai_AVC_SAFE_J7','SAFE_J7','SA_J7','FE_J7','MOCA', 'delai_AVC_IRM','loca_1SC_2C_3CSC','Volume','Overlap_CST_cross','delai_AVC_TMS','rMT_IPSI','rMT_CONTRO','PEM_plus1_ipsi','PEM_plus1_contro','PEMmax_ipsi','PEMmax_contro','FM_J7','delai_AVC_prelevement','Leucocytes','Lymphocytes','RNL','Neutrophiles','Plaquettes','Delai_follow_up_en_mois','ARAT_saisir','ARAT_agriper','ARAT_pincer','ARAT_motricite_globale','FM_3mois','ARAT_total','FM_imput_M3','ARAT_imput_M3','Probleme_Epaule','deces_follow_up','Categorie_M3_FM_binaire','Categorie_M3_ARAT','Categorie_M3_ARAT_binaire','FM_M3_J7_Max_Recovery_ratio','FM_M3_J7_Max_Recovery_ratio_binaire','Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire','FM_M3_J7_Recovery','Categorie_FM_imput_M3_equal_quaternaire','Categorie_M3_ARAT_equal_quaternaire']
Image_col_list = ['IRM_Diffusion_Raw', 'IRM_Diffusion_Norm_to_MNI', 'IRM_B0_Raw', 'IRM_B0_Norm_to_MNI', 'Lesion_Mask_Norm_to_MNI', 'ADC_Raw', 'ADC_Norm_to_MNI']
# Vect_col_list = ['Age','Sexe','Taille_cm','IRM_Diffusion_Raw','IRM_Diffusion_Norm', 'IRM_Diffusion_Norm_to_MNI', 'IRM_B0_Raw','IRM_B0_Norm', 'IRM_B0_Norm_to_MNI', 'IRM_FLAIR_Raw','Lesion_Mask_Norm', 'Lesion_Mask_Norm_to_MNI', 'ADC_Raw', 'ADC_Norm', 'ADC_Norm_to_MNI', 'Lateralite_D','Type_Ischemic','COTE_D','TIV','TM','Delai_AVC_NIHSS_J3','NIHSS_J3','NIHSS_MS_J3','NIHSS_MI_J3','NIHSS_neg_J3','NIHSS_apha_J3','delai_AVC_SAFE_J3','SAFE_J3','SA_J3','FE_J3','Delai_AVC_NIHSS_J7','NIHSS_J7','NIHSS_MS_J7','NIHSS_MI_J7','NIHSS_neg_J7','NIHSS_apha_J7','delai_AVC_SAFE_J7','SAFE_J7','SA_J7','FE_J7','MOCA', 'delai_AVC_IRM','loca_1SC_2C_3CSC','Volume','Overlap_CST_cross','delai_AVC_TMS','rMT_IPSI','rMT_CONTRO','PEM_plus1_ipsi','PEM_plus1_contro','PEMmax_ipsi','PEMmax_contro','FM_J7','delai_AVC_prelevement','Leucocytes','Lymphocytes','RNL','Neutrophiles','Plaquettes','Delai_follow_up_en_mois','ARAT_saisir','ARAT_agriper','ARAT_pincer','ARAT_motricite_globale','FM_3mois','ARAT_total','FM_imput_M3','ARAT_imput_M3','Probleme_Epaule','deces_follow_up','Categorie_M3_FM_binaire','Categorie_M3_ARAT','Categorie_M3_ARAT_binaire','FM_M3_J7_Max_Recovery_ratio','FM_M3_J7_Max_Recovery_ratio_binaire','Categorie_FM_M3_J7_Max_Recovery_ratio_equal_quaternaire','FM_M3_J7_Recovery','Categorie_FM_imput_M3_equal_quaternaire','Categorie_M3_ARAT_equal_quaternaire']
# Image_col_list = ['IRM_Diffusion_Raw', 'IRM_Diffusion_Norm', 'IRM_Diffusion_Norm_to_MNI', 'IRM_B0_Raw', 'IRM_B0_Norm', 'IRM_B0_Norm_to_MNI', 'IRM_FLAIR_Raw', 'Lesion_Mask_Norm', 'Lesion_Mask_Norm_to_MNI', 'ADC_Raw', 'ADC_Norm', 'ADC_Norm_to_MNI']
Out_Datasets_path = '/home/thomas.jacquemont/Test/PREP_AVC/New_Datasets_Management/Pickle_dataset/PREP_Whole_Database.pkl'


def create_whole_dataset_deep_learning(Vect_col_list, Image_col_list, Out_Datasets_path, PREP_database_path='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'):
    # Lecture du CSV
    data = pd.read_csv(PREP_database_path, sep=';')
    
    # Sélection des colonnes
    df_selection = data[['numero'] + Vect_col_list + Image_col_list]
    dataset = []
    
    for idx, row in df_selection.iterrows():
        numero_patient = row["numero"]
        
        subject_entry = {
            "id": numero_patient,
            "IRM": {},
            "VECT": {}
        }
        
        # -------------------------
        # 1. IRM
        # -------------------------
        for irm_col in Image_col_list:
            try:
                path = row[irm_col]
                
                # Gestion des valeurs manquantes
                if isinstance(path, pd.Series):
                    path = path.iloc[0]
                if pd.isna(path) or str(path).strip() == "":
                    raise FileNotFoundError("Chemin manquant")
                
                path = str(path)
                
                # Chargement de l'image
                img = nib.load(path)
                img_data = img.get_fdata().astype(np.float32)
                
                # Nettoyage
                img_data[np.isinf(img_data)] = 0
                img_data[np.isnan(img_data)] = 0
                
                # Normalisation
                mean, std = np.mean(img_data), np.std(img_data)
                if std > 0:
                    img_data = (img_data - mean) / std
                
                subject_entry["IRM"][irm_col] = img_data
                
            except Exception as e:
                print(f"⚠️ Patient {numero_patient} : {irm_col} manquante ({e})")
                subject_entry["IRM"][irm_col] = None
        
        # -------------------------
        # 2. VECT
        # -------------------------
        for col in Vect_col_list:
            try:
                val = row[col]
                
                # Si c'est une Series, prendre le premier élément
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                
                if pd.isna(val):
                    subject_entry["VECT"][col] = None
                else:
                    # Garder les colonnes IRM brutes telles quelles, sinon float32
                    subject_entry["VECT"][col] = val if col in Image_col_list else np.float32(val)
                    
            except Exception as e:
                print(f"⚠️ Patient {numero_patient} : erreur vecteur {col} ({e})")
                subject_entry["VECT"][col] = None
        
        dataset.append(subject_entry)
    
    # -------------------------
    # Sauvegarde Pickle
    # -------------------------
    os.makedirs(os.path.dirname(Out_Datasets_path), exist_ok=True)
    with open(Out_Datasets_path, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"✅ Dataset sauvegardé dans {Out_Datasets_path}")

create_whole_dataset_deep_learning(Vect_col_list, Image_col_list, Out_Datasets_path)