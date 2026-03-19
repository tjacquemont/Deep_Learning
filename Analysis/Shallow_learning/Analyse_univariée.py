#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:00:44 2024

@author: thomas
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import mne
import os
from statsmodels.stats.multitest import fdrcorrection

study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'
           
###################### Definition Parametres Script #######################            
save = True
working_directory = study_directory + '/Results/Univariate_Analysis'
cohorte_data_path = study_directory + '/Results/Patients_selection/Filtred_cohorte.csv'
cohorte_data = pd.read_csv(cohorte_data_path, sep=';')

# Variable selection
alpha = 0.05
correction = False

################# DEFINITION DES VRIABLES D'INTERETS #####################
variables_not_missable = ['Age', 'Sexe', 'Lateralite', 'Type_I_ischemic_H_hemorrhagic', 
                          'TIV', 'TM', 'NIHSS_J3', 'SAFE_J3', 'SAFE_J7', 'NIHSS_J7', 
                          'Volume', 'Overlap_CST_cross', 'rMT_IPSI', 'rMT_CONTRO', 
                          'PEM_plus1_ipsi', 'PEM_plus1_contro', 'PEMmax_ipsi', 
                          'PEMmax_contro', 'FM_J7']

target_variables = ['FM_imput_M3', 'ARAT_imput_M3', 'FM_M3_J7_Recovery']

########################################       STATISTIQUES      ####################################
# Binarisation des variables 'Sexe', 'Lateralité' et 'Type_I_ischemic_H_hemorrhagic'
cohorte_data['Sexe'] = cohorte_data['Sexe'].map({'H': 1, 'F': 0})
cohorte_data['Lateralite'] = cohorte_data['Lateralite'].map({'D': 1, 'G': 0})
cohorte_data['Type_I_ischemic_H_hemorrhagic'] = cohorte_data['Type_I_ischemic_H_hemorrhagic'].map({'I': 1, 'H': 0})
    
def analyser_correlation_student(df, variables_not_missable, target_variables):
    binary_vars = ['Sexe', 'Lateralite', 'Type_I_ischemic_H_hemorrhagic', 'TIV', 'TM', 'PEM_plus1_ipsi', 'PEM_plus1_contro']
    results = {} # Utiliser un dictionnaire pour stocker les résultats par variable cible

    for target in target_variables:
        target_results = []
        for var in variables_not_missable:
            if var in binary_vars:
                group1 = df[df[var] == 0][target].dropna()
                group2 = df[df[var] == 1][target].dropna()
                if len(group1) > 5 and len(group2) > 5:
                    stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                else:
                    p_value = np.nan
            else:
                temp_df = df[[var, target]].dropna(how='any')
                if len(temp_df) > 2:
                    stat, p_value = stats.pearsonr(temp_df[var], temp_df[target])
                else:
                    p_value = np.nan
            target_results.append({'Variable': var, 'Target': target, 'P-value': p_value})

        target_results_df = pd.DataFrame(target_results)
        target_results_df['P-value_corrected'] = fdrcorrection(target_results_df['P-value'].fillna(1))[1]
        results[target] = target_results_df # Stocker le DataFrame corrigé dans le dictionnaire

    return results

def significant_variables(resultats, output_dir, alpha=0.05, correction_multiple=True):
    """
    Trouve les variables significatives et les enregistre dans des fichiers texte individuels.

    Args:
        resultats (dict): Un dictionnaire de DataFrames contenant les résultats des tests.
        alpha (float): Le seuil alpha pour la significativité.
        correction_multiple (bool): Indique si la correction pour les comparaisons multiples a été appliquée.
        output_dir (str): Le dossier où les fichiers seront enregistrés.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Créer le dossier s'il n'existe pas

    all_conserved_variables = {}  # Stocker les listes de variables pour chaque target

    for target, result_df in resultats.items():
        if correction_multiple:
            conserved_variables = result_df[result_df['P-value_corrected'] < alpha]['Variable'].tolist()
        else:
            conserved_variables = result_df[result_df['P-value'] < alpha]['Variable'].tolist()

        all_conserved_variables[target] = conserved_variables #Stockage des variables par target dans le dictionnaire.

        output_file = os.path.join(output_dir, f"variables_significatives_{target}_Threshold_{str(alpha).replace('.','_')}_Correction_{str(correction_multiple)}.txt")
        with open(output_file, 'w') as f:
            for var in conserved_variables:
                f.write(var + '\n')

    return all_conserved_variables
###########################################################################################

resultats = analyser_correlation_student(cohorte_data, variables_not_missable, target_variables)
for target, result_df in resultats.items():
    print(f"Résultats pour {target}:\n{result_df}\n")
    result_df.to_csv(working_directory + f'/Univariate_analysis_{target}.csv', sep=';', index=False)


variables_significatives = significant_variables(resultats, working_directory, alpha=alpha, correction_multiple=correction)
print(variables_significatives)


