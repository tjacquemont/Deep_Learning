#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:36:09 2025

@author: thomas
"""
import pandas as pd
import numpy as np

study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'

def generer_tableau_recapitulatif(df, variables):
    """
    Génère un tableau récapitulatif des variables spécifiées.
    """

    recap = []
    for var in variables:
        if var in df.columns: # Verification de l'existance de la colonne.
            infos = {}
            infos['Variable'] = var
            infos['N_manquant'] = df[var].isnull().sum()
            infos['N_present'] = df[var].notnull().sum()

            if pd.api.types.is_numeric_dtype(df[var]):
                infos['Moyenne +/- Ecart-type'] = f"{df[var].mean():.2f} +/- {df[var].std():.2f}"
            else:
                try:
                    counts = df[var].value_counts(normalize=True) * 100
                    for value, percentage in counts.items():
                        infos[f'Pourcentage_{value}'] = f"{percentage:.2f}%"
                except TypeError:
                     infos['Modalités non calculables'] = 'Variable non numérique non catégorisable'
            recap.append(infos)
        else :
            print(f"Attention: La colonne '{var}' n'existe pas dans le DataFrame.")
    return pd.DataFrame(recap)

##############################################################################
# For the wholde database
# database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'
# recap_path = study_directory + '/Results/Demographics_Analysis/Table_1_-_Demographics_whole_cohorte.csv'


# After filtration
working_directory_filtration = study_directory + '/Results/Patients_selection'
database_path = working_directory_filtration + '/' + 'Filtred_cohorte.csv'
recap_path = study_directory + '/Results/Demographics_Analysis/Table_1_-_Demographics_filtered_cohorte.csv'

data = pd.read_csv(database_path, sep=';')

save = True


variables_of_interest =  ['Age', 'Sexe',
       'Taille_cm', 'Lateralite', 'Raw_IRM_Diffusion','Raw_IRM_FLAIR',
       'IRM_Diffusion_Norm', 'Lesion_Mask_Norm',
       'Type_I_ischemic_H_hemorrhagic', 'COTE_1G_2D', 'TIV','TM', 'Delai_AVC_NIHSS_J3', 'NIHSS_J3', 'NIHSS_MS_J3', 'NIHSS_neg_J3', 'NIHSS_apha_J3',
       'SAFE_J3', 'SAFE_J7', 'NIHSS_J7', 'NIHSS_MS_J7', 'NIHSS_neg_J7', 'NIHSS_aphaJ7', 
       'MOCA', 'Aphasie', 'Volume', 'Overlap_CST_cross', 'rMT_IPSI', 'rMT_CONTRO',
       'PEM_plus1_ipsi', 'PEM_plus1_contro', 'PEMmax_ipsi', 'PEMmax_contro',
       'FM_J7', 'Leucocytes', 'Lymphocytes','RNL', 'Neutrophiles', 'Plaquettes', 
       'Delai_follow_up_en_mois', 'deces_follow_up', 'FM_imput_M3', 'ARAT_imput_M3',
       'delai_AVC_TMS', 'delai_AVC_IRM', 'delai_AVC_prelevement']

recap = generer_tableau_recapitulatif(data, variables_of_interest)

recap.to_csv(recap_path, sep=';', index=False)