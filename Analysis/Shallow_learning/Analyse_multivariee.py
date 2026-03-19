#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:01:19 2025

@author: thomas
"""
import pandas as pd
import statsmodels.api as sm
import os

study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'

###################### Definition Parametres Script #######################            
save = True
working_directory = study_directory + '/Results/Multivariate_Analysis'
cohorte_data_path = study_directory + '/Results/Patients_selection/Filtred_cohorte.csv'
cohorte_data = pd.read_csv(cohorte_data_path, sep=';')

# Variable selection
alpha = 0.05
correction_multiple = False
selected_variables_directory = study_directory + '/Results/Univariate_Analysis'


############################## Model fitting ################################

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

def charger_variables_selectionnées(dossier_variables, target_variable, alpha, correction_multiple):
    """
    Charge les variables sélectionnées à partir d'un fichier texte.

    Args:
        dossier_variables (str): Le dossier contenant les fichiers de variables sélectionnées.
        variable_cible (str): Le nom de la variable cible.

    Returns:
        list: La liste des variables sélectionnées.
    """

    fichier_variables = os.path.join(dossier_variables, f"variables_significatives_{target_variable}_Threshold_{str(alpha).replace('.','_')}_Correction_{str(correction_multiple)}.txt")
    with open(fichier_variables, 'r') as f:
        variables = [line.strip() for line in f]
    return variables

################################################################################
# Using all covariables (but with colinearity)
# Variables cibles
variables_cibles = ['FM_imput_M3', 'ARAT_imput_M3'] # FM_M3_J7_Recovery non étudié car aucune variable retenue

for target_variable in variables_cibles:
    selected_variables = charger_variables_selectionnées(selected_variables_directory, target_variable, alpha, correction_multiple)
    regression_multivariée(cohorte_data, selected_variables, target_variable, working_directory)

# Choosing onlys 1 variable for clinique, electrophysiologique and CST
variables_cibles = ['FM_imput_M3', 'ARAT_imput_M3'] 
selected_variables_clinic = ['NIHSS_J3', 'SAFE_J3', 'SAFE_J7', 'NIHSS_J7', 'FM_J7']
selected_variables_tms = [ 'PEM_plus1_ipsi', 'PEMmax_ipsi']
selected_variables_image = ['Overlap_CST_cross']

for target_variable in variables_cibles:
    for clinic in selected_variables_clinic:
        for tms in selected_variables_tms:
            for img in selected_variables_image:
                selected_variables = [clinic] + [tms] + [img]
                regression_multivariée(cohorte_data, selected_variables, target_variable, working_directory)
                
