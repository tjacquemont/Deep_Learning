#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:14:39 2025

@author: thomas.jacquemont
"""

import os
import pandas as pd

root_dir = "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/IRM_Diffusion_ADC_Lesion_Mask_norm"

def get_model_name(path):
    return os.path.basename(os.path.dirname(path))

def get_vector_features(model_name):
    if "vect_" in model_name:
        idx = model_name.find("vect_") + len("vect_")
        return model_name[idx:]
    return "Unknown"

def get_lesion_side(path_parts):
    for part in path_parts:
        if part in ["Left", "Right"]:
            return part
    return "Unknown"

def get_vector_set(path_parts):
    for part in path_parts:
        if part in ["J7", "PREP2", "IRM_only"]:
            return part
    return "Unknown"

def get_motor_score(path_parts):
    for part in path_parts:
        if part in ["FM_imput_M3", "ARAT_imput_M3", "FM_M3_J7_Max_Recovery_ratio"]:
            return part
    return "Unknown"

def get_lesion_side(path_parts):
    for part in path_parts:
        if part in ["Left", "Right"]:
            return part
    return "Unknown"

def get_augmentation(augmentation_dir):
    if 'without_Flip' in augmentation_dir:
        return 'without_Flip'
    elif 'with_Flip' in augmentation_dir:
        return 'with_Flip'
    else:
        return "Unknown"

###############################################################################
def get_deeplearning_results_summarized(root_dir):
    
    summary_data = {}
    
    for rmi_data_augmentation_dir in os.listdir(root_dir):
        rmi_data_augmentation = get_augmentation(rmi_data_augmentation_dir)
        rmi_data_augmentation_path = os.path.join(root_dir, rmi_data_augmentation_dir)
        if not os.path.isdir(rmi_data_augmentation_path):
            continue
        
        # Parcours des sets de vecteurs (J7, PREP2)
        for vector_set_dir in os.listdir(rmi_data_augmentation_path):
            vector_set_path = os.path.join(rmi_data_augmentation_path, vector_set_dir)
            if not os.path.isdir(vector_set_path):
                continue
    
            # Parcours des scores moteurs
            for score_dir in os.listdir(vector_set_path):
                score_path = os.path.join(vector_set_path, score_dir)
                if not os.path.isdir(score_path):
                    continue
    
                if score_dir not in summary_data:
                    summary_data[score_dir] = []
    
                # Parcours des modèles
                for model_dir in os.listdir(score_path):
                    model_path_dir = os.path.join(score_path, model_dir)
                    if not os.path.isdir(model_path_dir):
                        continue
    
                    for file in os.listdir(model_path_dir):
                        if file.endswith("_history.csv"):
                            file_path = os.path.join(model_path_dir, file)
                            model_name = get_model_name(file_path)
                            vector_features = get_vector_features(model_name)
    
                            # Chemins supplémentaires
                            best_model_path = os.path.join(model_path_dir, "best_model.pth")
                            patient_list_path = os.path.join(model_path_dir, "patient_list.csv")
                            indices_path = os.path.join(model_path_dir, file.replace("_history.csv", "_indices.csv"))
    
                            try:
                                df = pd.read_csv(file_path, sep=';')
                            except Exception as e:
                                print(f"Impossible de lire {file_path}: {e}")
                                continue
    
                            required_cols = ["val_r2", "val_rmse", "val_mae"]
                            if not all(col in df.columns for col in required_cols):
                                print(f"Colonnes manquantes dans {file_path}, skipping")
                                continue
    
                            best_idx = df["val_r2"].idxmax()
    
                            summary_data[score_dir].append({
                                "model_name": model_name,
                                "flip_augmentation" : rmi_data_augmentation,
                                "vector_set": vector_set_dir,
                                "vector_features": vector_features,
                                "best_val_r2": round(df.loc[best_idx, "val_r2"], 3),
                                "val_mae_at_best_r2": round(df.loc[best_idx, "val_mae"], 3),
                                "val_rmse_at_best_r2": round(df.loc[best_idx, "val_rmse"], 3),
                                "model_path": best_model_path if os.path.exists(best_model_path) else "MISSING",
                                "patient_list_path": patient_list_path if os.path.exists(patient_list_path) else "MISSING",
                                "indices_path": indices_path if os.path.exists(indices_path) else "MISSING"
                            })
    
        # Sauvegarde des fichiers résumés pour chaque score moteur
    for score, data in summary_data.items():
        if data:
            df_summary = pd.DataFrame(data)
            summary_file = os.path.join(root_dir, f"DeepLearning_Performance_Summary_{score}.csv")
            df_summary.to_csv(summary_file, index=False)
            print(f"Résumé sauvegardé pour {score} dans {root_dir} -> {summary_file}")

###############################################################################"
def get_deeplearning_lesion_side_results_summarized(root_dir):
    summary_data = {
        "FM_imput_M3": [],
        "ARAT_imput_M3": [],
        "FM_M3_J7_Max_Recovery_ratio": []
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith("_history.csv"):
                file_path = os.path.join(dirpath, file)
    
                try:
                    df = pd.read_csv(file_path, sep=';')
                except Exception as e:
                    print(f"Impossible de lire {file_path}: {e}")
                    continue
    
                required_cols = ["val_r2", "val_rmse", "val_mae"]
                if not all(col in df.columns for col in required_cols):
                    print(f"Colonnes manquantes dans {file_path}, skipping")
                    continue
    
                best_idx = df["val_r2"].idxmax()
                best_val_r2 = round(df.loc[best_idx, "val_r2"], 3)
                val_rmse_at_best = round(df.loc[best_idx, "val_rmse"], 3)
                val_mae_at_best = round(df.loc[best_idx, "val_mae"], 3)
    
                path_parts = dirpath.split(os.sep)
                model_name = get_model_name(file_path)
                vector_set = get_vector_set(path_parts)
                motor_score = get_motor_score(path_parts)
                lesion_side = get_lesion_side(path_parts)
                flip_augmentation = get_augmentation(file_path)
    
                vector_features = "Unknown"
                if "vect_" in model_name:
                    idx = model_name.find("vect_") + len("vect_")
                    vector_features = model_name[idx:]
    
                model_path = os.path.join(os.path.dirname(file_path), "best_model.pth")
                patient_list_path = os.path.join(os.path.dirname(file_path), "patient_list.csv")
                indices_path = os.path.join(os.path.dirname(file_path), file.replace("history", "indices"))
    
                if motor_score in summary_data:
                    summary_data[motor_score].append({
                        "model_name": model_name,
                        "flip_augmentation" : flip_augmentation,
                        "vector_set": vector_set,
                        "lesion_side": lesion_side,
                        "vector_features": vector_features,
                        "best_val_r2": best_val_r2,
                        "val_rmse_at_best_r2": val_rmse_at_best,
                        "val_mae_at_best_r2": val_mae_at_best,
                        "model_path": model_path,
                        "patient_list_path": patient_list_path,
                        "indices_path": indices_path
                    })
    
    # Sauvegarder les résumés
    for score, data in summary_data.items():
        if data:
            df_summary = pd.DataFrame(data)
            summary_file = os.path.join(root_dir, f"DeepLearning_Performance_Summary_{score}.csv")
            df_summary.to_csv(summary_file, index=False)
            print(f"Résumé sauvegardé pour {score} -> {summary_file}")
