#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:44:39 2025

@author: thomas.jacquemont
"""

import pandas as pd
import numpy as np

def formater_donnees_excel(chemin_fichier, chemin_fichier_irm_csv="/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv", anonymisation=False, chemin_sortie_csv=None):
    """
    Cette fonction prend un chemin vers un fichier Excel, effectue diverses
    opérations de nettoyage et de formatage des données, puis renvoie
    le DataFrame modifié.

    Args:
        chemin_fichier (str): Le chemin complet vers le fichier Excel.
        anonymisation (bool, optional): Si True, supprime les colonnes
                                        'IPP', 'Nom', 'Prenom'. Par défaut à False.

    Returns:
        pandas.DataFrame: Le DataFrame formaté.
    """
    try:
        df = pd.read_excel(chemin_fichier)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{chemin_fichier}' est introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier Excel : {e}")
        return None
    
    if "IPP" in df.columns:
        # Convertir en numérique d'abord, en forçant les erreurs en NaN
        df["IPP"] = pd.to_numeric(df["IPP"], errors='coerce')
        # Convertir en Int64 avec support de NaN
        df["IPP"] = df["IPP"].astype(pd.Int64Dtype())
        print("La colonne 'IPP' a été formatée en entiers (Int64) avec gestion des NaN.")
    else:
        print("Avertissement : La colonne 'IPP' n'a pas été trouvée pour le formatage.")

    # Renommer les colonnes
    df = df.rename(columns={
        "Taille\n(cm)": "Taille_cm",
        "Lateralite\n1G_2D": "Lateralite_D",
        "Date AVC": "Date_AVC",
        "Type AVC\n1I_2H": "Type_Ischemic",
        "COTE\n1G_2D": "COTE_D",
        "Date _IRM" : "Date_IRM",
        "TTT_0rien_\n1TIV_2TM_3 comb": "TTT_0rien_1TIV_2TM_3_comb", # Renommer temporairement pour la création des colonnes
        "Delai\nAVC_NIHSS J3": "Delai_AVC_NIHSS_J3",
        "NIHSS_MS\nJ3": "NIHSS_MS_J3",
        "NIHSS_MI\nJ3": "NIHSS_MI_J3",
        "NIHSS_neg\nJ3": "NIHSS_neg_J3",
        "NIHSS_apha\nJ3": "NIHSS_apha_J3",
        "Date_SAFE\nJ3": "Date_SAFE_J3",
        "delai\nAVC_SAFE J3": "delai_AVC_SAFE_J3",
        "date_NIHSS\nJ7": "date_NIHSS_J7",
        "delai \nAVC_NIHSS J7": "Delai_AVC_NIHSS_J7",
        "NIHSS_MS\nJ7": "NIHSS_MS_J7",
        "NIHSS_MI\nJ7": "NIHSS_MI_J7",
        "NIHSS_neg\nJ7": "NIHSS_neg_J7",
        "NIHSS_aphaJ7": "NIHSS_apha_J7",
        "Date_SAFE\nJ7": "Date_SAFE_J7",
        "delai AVC_SAFE J7": "delai_AVC_SAFE_J7",
        "loca 1SC 2C 3CSC": "loca_1SC_2C_3CSC",
        "Date TMS": "Date_TMS",
        "Date\nPrelevement": "Date_Prelevement",
        "ARAT_motricite globale": "ARAT_motricite_globale",
        "Epaule (1: Probleme 0: No) ": "Probleme_Epaule",
        "FM_imputé": "FM_imput_M3",
        "ARAT_imputé" : "ARAT_imput_M3",
        "Remarques suivi Lina" : "Remarques_suivi"
        }, errors="raise")
    
   # Find the index of "Remarques_suivi" and drop columns to its right
    if "Remarques_suivi" in df.columns:
       idx_remarques_suivi = df.columns.get_loc("Remarques_suivi")
       # Select all columns up to and including "Remarques_suivi"
       df = df.iloc[:, :idx_remarques_suivi + 1]
       print(f"Columns to the right of 'Remarques_suivi' have been removed.")
    else:
       print("Warning: 'Remarques_suivi' column not found. No columns were removed based on this criterion.")
       
    # Remove all semicolons from "Remarques_suivi" column
    if "Remarques_suivi" in df.columns:
        # Convert to string type first to ensure .str.replace works
        df["Remarques_suivi"] = df["Remarques_suivi"].astype(str).str.replace(';', '', regex=False)
        print("Semicolons have been removed from the 'Remarques_suivi' column.")
    else:
        print("Warning: 'Remarques_suivi' column not found, could not remove semicolons.")


    # Traiter la colonne "Lateralite_D"
    df["Lateralite_D"] = df["Lateralite_D"].replace({1: 0, 2: 1})

    # Traiter la colonne "Type_Ischemic"
    df["Type_Ischemic"] = df["Type_Ischemic"].replace({2: 0})

    # Traiter la colonne "COTE_D"
    df["COTE_D"] = df["COTE_D"].replace({1: 0, 2: 1})

    # Créer les colonnes "TIV" et "TM"
    if "TTT_0rien_1TIV_2TM_3_comb" in df.columns:
        # Assurez-vous que la colonne est numérique avant de faire des comparaisons
        df["TTT_0rien_1TIV_2TM_3_comb"] = pd.to_numeric(df["TTT_0rien_1TIV_2TM_3_comb"], errors='coerce')

        # Initialiser TIV et TM avec des zéros
        df["TIV"] = np.nan
        df["TM"] = np.nan

        # Remplir les colonnes TIV et TM selon les conditions
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 0, "TIV"] = 0
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 0, "TM"] = 0
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 1, "TIV"] = 1
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 1, "TM"] = 0
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 2, "TM"] = 1
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 2, "TIV"] = 0
        df.loc[df["TTT_0rien_1TIV_2TM_3_comb"] == 3, ["TIV", "TM"]] = 1

        # Trouver l'index de la colonne source pour insérer les nouvelles colonnes
        idx_ttt = df.columns.get_loc("TTT_0rien_1TIV_2TM_3_comb")

        # Insérer "TIV" et "TM" après "TTT_0rien_1TIV_2TM_3_comb"
        # Insérer TM en premier si on veut TIV puis TM à droite de TTT
        df.insert(idx_ttt + 1, "TM", df.pop("TM"))
        df.insert(idx_ttt + 1, "TIV", df.pop("TIV"))

        # Supprimer la colonne originale après avoir créé et déplacé les nouvelles
#        df = df.drop(columns=["TTT_0rien_1TIV_2TM_3_comb"])
        print("'TIV' et 'TM' créées et placées à côté de l'ancienne colonne 'TTT_0rien_1TIV_2TM_3_comb', qui a été supprimée.")
    else:
        print("Avertissement : La colonne 'TTT_0rien_1TIV_2TM_3_comb' n'a pas été trouvée. 'TIV' et 'TM' n'ont pas été créées à l'emplacement souhaité.")
    
    # --- Gestion des colonnes IRM (nouvelle section) ---
    if "Date_AVC" in df.columns:
        idx_date_avc = df.columns.get_loc("Date_AVC")
        new_irm_cols = [
            "Raw_IRM_Diffusion", "Date_Raw_Diffusion", "Raw_IRM_Diffusion_path",
            "Raw_IRM_FLAIR", "Raw_IRM_FLAIR_path", "IRM_Diffusion_Norm",
            "IRM_Diffusion_Norm_path", "Lesion_Mask_Norm", "Lesion_Mask_Norm_path"
            ]
        # Si un chemin CSV pour les IRM est fourni, on fusionne
        if chemin_fichier_irm_csv:
                try:
                    df_irm = pd.read_csv(chemin_fichier_irm_csv, sep=';')

                    # Fusionner les données IRM. 'how='left'' garantit que toutes les lignes de df sont conservées.
                    df = pd.merge(df, df_irm[['numero'] + new_irm_cols], on='numero', how='left')
                    print(f"Données IRM fusionnées depuis '{chemin_fichier_irm_csv}'.")

                except FileNotFoundError:
                    print(f"Erreur : Le fichier CSV IRM '{chemin_fichier_irm_csv}' est introuvable. Les colonnes IRM seront vides.")
                    # Si le fichier n'est pas trouvé, on insère des colonnes vides
                    for col in reversed(new_irm_cols):
                        df.insert(idx_date_avc + 1, col, np.nan)
                except Exception as e:
                    print(f"Une erreur est survenue lors de la lecture ou la fusion du fichier CSV IRM : {e}. Les colonnes IRM seront vides.")
                    # Si une autre erreur survient, on insère des colonnes vides
                    for col in reversed(new_irm_cols):
                        df.insert(idx_date_avc + 1, col, np.nan)
        else:
            # Si aucun CSV IRM n'est fourni, mais Date_AVC est présente, on insère des colonnes vides
            for col in reversed(new_irm_cols):
                df.insert(idx_date_avc + 1, col, np.nan)
                print("Aucun fichier CSV IRM fourni. Les colonnes IRM ont été ajoutées vides.")
        
        # Reordonner les colonnes pour placer les IRM à côté de "Date_AVC"
        current_cols = df.columns.tolist()
        idx_date_avc = current_cols.index("Date_AVC")

        # Supprimer les colonnes IRM de leur position actuelle (probablement la fin)
        # et les ajouter à la liste dans l'ordre souhaité
        cols_to_move = [col for col in new_irm_cols if col in current_cols]
        for col in cols_to_move:
            current_cols.remove(col) # Supprime la colonne de sa position actuelle

        # Insérer les colonnes IRM après "Date_AVC"
        for i, col in enumerate(cols_to_move):
            current_cols.insert(idx_date_avc + 1 + i, col)

        df = df[current_cols] # Appliquer le nouvel ordre
        print("Les colonnes IRM ont été déplacées à côté de 'Date_AVC'.")

    else:
        print("La colonne 'Date_AVC' n'a pas été trouvée, les colonnes IRM n'ont pas été ajoutées à l'emplacement souhaité.")


    # Nettoyer la colonne "MOCA"
    # Convertir en numérique, les erreurs deviendront NaN, puis les supprimer
    df["MOCA"] = pd.to_numeric(df["MOCA"], errors='coerce')

    # Créer la colonne "deces_follow_up"
    df["deces_follow_up"] = df["Date_suivi"].astype(str).apply(
        lambda x: 1 if any(word in x for word in ["Décedé", "Décedée", "DCD", "décedée", "décedée", "Decedée", "Décédée", "Décédé"]) else 0
    )
    
    # Convertir les colonnes FM_imput_M3 et ARAT_imput_M3 en numérique AVANT de les utiliser pour les catégories
    df["FM_imput_M3"] = pd.to_numeric(df["FM_imput_M3"], errors='coerce')
    df["ARAT_imput_M3"] = pd.to_numeric(df["ARAT_imput_M3"], errors='coerce')
    df["FM_J7"] = pd.to_numeric(df["FM_J7"], errors='coerce')

    # Création des colonnes de catégories
    # Categorie_M3_FM_binaire
    df["Categorie_M3_FM_binaire"] = np.nan # Initialiser avec NaN
    df.loc[(df["FM_imput_M3"] < 40), "Categorie_M3_FM_binaire"] = 0
    df.loc[(df["FM_imput_M3"] >= 40) & (df["FM_imput_M3"] <= 60), "Categorie_M3_FM_binaire"] = 1

    # Categorie_M3_ARAT
    df["Categorie_M3_ARAT"] = np.nan # Initialiser avec NaN
    df.loc[(df["ARAT_imput_M3"] >= 0) & (df["ARAT_imput_M3"] <= 12), "Categorie_M3_ARAT"] = 0
    df.loc[(df["ARAT_imput_M3"] >= 13) & (df["ARAT_imput_M3"] <= 33), "Categorie_M3_ARAT"] = 1
    df.loc[(df["ARAT_imput_M3"] >= 34) & (df["ARAT_imput_M3"] <= 48), "Categorie_M3_ARAT"] = 2
    df.loc[(df["ARAT_imput_M3"] > 48) & (df["ARAT_imput_M3"] <= 57), "Categorie_M3_ARAT"] = 3 

    # Categorie_M3_ARAT_binaire
    df["Categorie_M3_ARAT_binaire"] = np.nan
    df.loc[(df["ARAT_imput_M3"] < 34), "Categorie_M3_ARAT_binaire"] = 0
    df.loc[(df["ARAT_imput_M3"] >= 34) & (df["ARAT_imput_M3"] <= 57), "Categorie_M3_ARAT_binaire"] = 1

    # Création des colonnes de ratios et différences
    # FM_M3_J7_Max_Recovery_ratio
    df["FM_M3_J7_Max_Recovery_ratio"] = (df["FM_imput_M3"] - df["FM_J7"]) / (60 - df["FM_J7"])

    # FM_M3_J7_Max_Recovery_ratio_binaire
    df["FM_M3_J7_Max_Recovery_ratio_binaire"] = np.nan
    df.loc[df["FM_M3_J7_Max_Recovery_ratio"] < 0.7, "FM_M3_J7_Max_Recovery_ratio_binaire"] = 0
    df.loc[df["FM_M3_J7_Max_Recovery_ratio"] >= 0.7, "FM_M3_J7_Max_Recovery_ratio_binaire"] = 1

    # FM_M3_J7_Recovery
    df["FM_M3_J7_Recovery"] = df["FM_imput_M3"] - df["FM_J7"]

    # Traitement des dates pour les délais
    date_cols = ["Date_Prelevement", "Date_AVC", "Date_IRM", "Date_TMS"]
    for col in date_cols:
        if col in df.columns:
            # Convertir en datetime, les erreurs (formats non valides) deviennent NaT
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # delai_AVC_prelevement
    df["delai_AVC_prelevement"] = (df["Date_Prelevement"] - df["Date_AVC"]).dt.days

    # delai_AVC_IRM
    df["delai_AVC_IRM"] = (df["Date_IRM"] - df["Date_AVC"]).dt.days

    df["delai_AVC_TMS"] = (df["Date_TMS"] - df["Date_AVC"]).dt.days

    # Supprimer les valeurs "DM" (données manquantes) dans toutes les colonnes sauf "Remarques_suivi"
    for col in df.columns:
        if col != "Remarques_suivi":
            df[col] = df[col].replace("DM", np.nan)
            df[col] = df[col].replace("DM ", np.nan)
            df[col] = df[col].replace("DM  ", np.nan)
            

    # Anonymisation
    if anonymisation:
        cols_to_drop = ["IPP", "Nom", "Prenom"]
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
            else:
                print(f"La colonne '{col}' à supprimer pour l'anonymisation n'a pas été trouvée.")

    if "Remarques_suivi" in df.columns:
        remarques_col = df.pop("Remarques_suivi") # Removes the column and returns it
        df["Remarques_suivi"] = remarques_col     # Adds it back at the end
        print("'Remarques_suivi' column moved to the end.")
    else:
        print("Warning: 'Remarques_suivi' column not found, could not move it to the end.")
    
    if chemin_sortie_csv:
        try:
            df.to_csv(chemin_sortie_csv, index=False, encoding='utf-8')
            print(f"DataFrame successfully saved to: {chemin_sortie_csv}")
        except Exception as e:
            print(f"An error occurred while saving the CSV file: {e}")

    return df