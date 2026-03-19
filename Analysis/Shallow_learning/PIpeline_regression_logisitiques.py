#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12:36:09 2025

@author: thomas
"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression  # Pour la séparation des données
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import statsmodels.api as sm
import os
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Vérification de la disponibilité de CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PYTORCH : Utilisation de l'appareil : {device}")


##########################################################################################################################
# Function_definition
##########################################################################################################################

def patient_filtration(cohorte_data, dependante_variable, independante_variables_to_conserve, working_directory):

    variables_to_conserve = independante_variables_to_conserve + dependante_variable 
    variables_to_conserve += ['numero']

    not_missable_variables_cohorte = cohorte_data[variables_to_conserve].dropna()
    nb_patient_selected = not_missable_variables_cohorte.shape[0]

    print(f"Number of selected patients : {nb_patient_selected}")

    filtered_cohorte = cohorte_data.loc[cohorte_data['numero'].isin(not_missable_variables_cohorte['numero'])]

    filtered_cohorte.to_csv(working_directory + '/' + 'Filtred_cohorte.csv', sep=';', index=False)

    return filtered_cohorte
    

def generer_tableau_recapitulatif(df, list_variables):
    """
    Génère un tableau récapitulatif des variables spécifiées.
    """

    recap = []
    for var in list_variables:
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

##########################################################################################################################
# Multivariate analysis
##########################################################################################################################
 
def regression_multivariée(cohorte_data, selected_variables, target_variable, output_dir):
    """
    Effectue une régression multivariée et enregistre le résumé dans un fichier texte.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        variables_indépendantes (list): Liste des variables indépendantes (prédicteurs).
        variable_dépendante (str): Nom de la variable dépendante (cible).
        output_dir (str): Le dossier où les fichiers seront enregistrés.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = sm.add_constant(cohorte_data[selected_variables])
    y = cohorte_data[target_variable]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    output_file = os.path.join(output_dir, f"regression_of_{target_variable}_with_{'_'.join(selected_variables)}.txt")
    with open(output_file, 'w') as f:
        f.write(str(summary))

def logits_regression_pipeline(cohorte_data, dependante_variable, independante_variables_to_conserve, working_directory, list_demographics_variables):
    """
    Will performe a multivariate logistic regression of the extracted cohorte will writting the extraced CSV and the Table-1 - Dmogrpahics
    
    Parameters
    ----------
    database_path : TYPE (str)
        DESCRIPTION : PAth to the CSV database
    dependante_variable : TYPE (list)
        DESCRIPTION : List of dependant variable, varibale to explain
    independante_variables_to_conserve : TYPE (list)
        DESCRIPTION : List of independant variable, used to explain the dependante varibale
    working_directory : TYPE (str)
        DESCRIPTION. : directory where the study sub_directory_will_be_created
    list_demogaphics_variables : TYPE (list)
        DESCRIPTION : List of variable presente in the Table 1 - Demoraphics

    Returns
    -------
    None.

    """
    ###########################################################################
    # Overall patient before filtering
    nb_overall_patient = cohorte_data.shape[0]
    print(f"Initial number of patient in database : {nb_overall_patient} ")
    
    ###########################################################################
    # creation du dossier de travail propre à la combinaison de variables
    variables_to_conserve = independante_variables_to_conserve + dependante_variable
    new_directory = '_'.join(variables_to_conserve)
    working_directory = working_directory + '/' + new_directory
    print(f"Creating the directoy : {working_directory} ")
    if not os.path.exists(working_directory):
           os.makedirs(working_directory)
    
    ###########################################################################
    # Filtration des données selon les variables étudiée (variables dépendante et indépendantes)
    print(f"Selection of the patient with all the necessay data")
    print(f"Dependant variable : {dependante_variable}")
    print(f"Independant variables : {independante_variables_to_conserve}")
    filtered_cohorte = patient_filtration(cohorte_data, dependante_variable, independante_variables_to_conserve, working_directory)
    
    ###########################################################################
    # Demographic Table 1
    print(f"Creating the Table 1 - Demographics of the selected cohorte")
    recap_path = working_directory + '/Table_1_-_Demographics_filtered_cohorte.csv'
    recap = generer_tableau_recapitulatif(filtered_cohorte, list_demographics_variables)

    recap.to_csv(recap_path, sep=';', index=False)
    
    ###########################################################################
    # Performing multiple regression
    print(f"Peforming the Multivariate Logistic Regression")
    regression_multivariée(filtered_cohorte, independante_variables_to_conserve, dependante_variable, working_directory)


##########################################################################################################################
study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'
database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'
working_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Standard_Machine_Learning_Methods'


list_demographics_variables =  ['Age', 'Sexe',
       'Taille_cm', 'Lateralite', 'Raw_IRM_Diffusion','Raw_IRM_FLAIR',
       'IRM_Diffusion_Norm', 'Lesion_Mask_Norm',
       'Type_I_ischemic_H_hemorrhagic', 'COTE_1G_2D', 'TIV','TM', 'Delai_AVC_NIHSS_J3', 'NIHSS_J3', 'NIHSS_MS_J3', 'NIHSS_neg_J3', 'NIHSS_apha_J3',
       'SAFE_J3', 'SAFE_J7', 'NIHSS_J7', 'NIHSS_MS_J7', 'NIHSS_neg_J7', 'NIHSS_aphaJ7', 
       'MOCA', 'Aphasie', 'Volume', 'Overlap_CST_cross', 'rMT_IPSI', 'rMT_CONTRO',
       'PEM_plus1_ipsi', 'PEM_plus1_contro', 'PEMmax_ipsi', 'PEMmax_contro',
       'FM_J7', 'Leucocytes', 'Lymphocytes','RNL', 'Neutrophiles', 'Plaquettes', 
       'Delai_follow_up_en_mois', 'deces_follow_up', 'FM_imput_M3', 'ARAT_imput_M3',
       'delai_AVC_TMS', 'delai_AVC_IRM', 'delai_AVC_prelevement']


machine_larning_features = ['Age', 'Sexe', 'NIHSS_J3', 'SAFE_J3', 'SAFE_J7', 'NIHSS_J7','Volume', 'Overlap_CST_cross', 'PEM_plus1_ipsi']

##########################################################################################################################
# Model variables définitions
##########################################################################################################################
liste_independante_variables_to_conserve = [['NIHSS_J3', 'SAFE_J3','Overlap_CST_cross'], ['NIHSS_J3', 'SAFE_J3','Overlap_CST_cross', 'PEM_plus1_ipsi'], ['NIHSS_J7', 'SAFE_J7','Overlap_CST_cross', 'PEM_plus1_ipsi'] ]
liste_dependante_variable = [['Categorie_M3_FM_binaire'], ['Categorie_M3_ARAT_binaire']]
list_cohorte_to_analyse = ['Whole', 'SAFE' or 'PEM']
cohorte_to_analyse = 'Whole' # define the cohorte population, should be 'Whole', 'SAFE' or 'PEM'

##########################################################################################################################
# Selection des sous groupes (PEM+/PEM-) ou SAFE J3 >=5 ou <5

for dependante_variable in liste_dependante_variable:
    for independante_variables_to_conserve in liste_independante_variables_to_conserve:
        ########################################################################
        # Récupération des données
        print(f"Loading data in the {database_path} CSV.")
        print(f"Cohorte to analyse : {cohorte_to_analyse}")
        cohorte_data = pd.read_csv(database_path, sep=';')
        if cohorte_to_analyse == 'Whole':
            working_directory_cohorte = working_directory + '/Whole_cohorte'
            cohorte_data_to_analyse = cohorte_data
            logits_regression_pipeline(cohorte_data_to_analyse, dependante_variable, independante_variables_to_conserve, working_directory_cohorte, list_demographics_variables)
        elif cohorte_to_analyse == 'SAFE':
            working_directory_safe = working_directory + '/SAFE_J3'
            cohorte_data_SAFE_J3_cleaned = cohorte_data.dropna(subset=['SAFE_J3']).copy()
            independante_variables_to_conserve_safe = independante_variables_to_conserve.copy()
            if 'SAFE_J3' in independante_variables_to_conserve_safe:
                independante_variables_to_conserve_safe.remove('SAFE_J3')
            cohorte_data_SAFE_J3_cleaned_plus_5 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] >= 5].copy()
            cohorte_data_SAFE_J3_cleaned_moins_5 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] < 5].copy()
            ###################################################################
            # Analysing the SAFE J3 >= 5 subcohort
            print("Analysing the SAFE J3 > 5 sub-cohort")
            working_directory_cohorte = working_directory_safe + '/More_than_5'
            cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_plus_5
            logits_regression_pipeline(cohorte_data_to_analyse, dependante_variable, independante_variables_to_conserve_safe, working_directory_cohorte, list_demographics_variables)
            ###################################################################
            # Analysing the SAFE J3 < 5 subcohort
            print("Analysing the SAFE J3 < 5 sub-cohort")
            working_directory_cohorte = working_directory_safe + '/Less_than_5'
            cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_moins_5
            logits_regression_pipeline(cohorte_data_to_analyse, dependante_variable, independante_variables_to_conserve_safe, working_directory_cohorte, list_demographics_variables)
        elif cohorte_to_analyse == 'PEM':
            working_directory_pem = working_directory + '/PEM'
            cohorte_data_PEM_cleaned = cohorte_data.dropna(subset=['PEM_plus1_ipsi']).copy()
            independante_variables_to_conserve_pem = independante_variables_to_conserve.copy()
            if 'PEM_plus1_ipsi' in independante_variables_to_conserve_pem:
                independante_variables_to_conserve_pem.remove('PEM_plus1_ipsi')
            cohorte_data_PEM_pos = cohorte_data_PEM_cleaned[cohorte_data_PEM_cleaned['PEM_plus1_ipsi']==1].copy()
            cohorte_data_PEM_neg = cohorte_data_PEM_cleaned[cohorte_data_PEM_cleaned['PEM_plus1_ipsi']==0].copy()
            ###################################################################
            # Analysing the PEM positive subcohort
            print("Analysing the PEM positive subcohort")
            working_directory_cohorte = working_directory_pem + '/Present'
            cohorte_data_to_analyse = cohorte_data_PEM_pos
            independante_variables_to_conserve_PEM_pos = independante_variables_to_conserve_pem + ['PEMmax_ipsi']
            logits_regression_pipeline(cohorte_data_to_analyse, dependante_variable, independante_variables_to_conserve_PEM_pos, working_directory_cohorte, list_demographics_variables)
            ###################################################################
            # Analysing the PEM negative subcohort
            print("Analysing the PEM negative subcohort")
            working_directory_cohorte = working_directory_pem + '/Absent'
            cohorte_data_to_analyse = cohorte_data_PEM_neg
            logits_regression_pipeline(cohorte_data_to_analyse, dependante_variable, independante_variables_to_conserve_pem, working_directory_cohorte, list_demographics_variables)
        else:
            print("ERROR IN COHORTE DEFINITION")
     

