#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:08:05 2025

@author: thomas
"""
import pandas as pd

study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'

##########################################################################################################################"

working_directory_filtration = study_directory + '/Results/Patients_selection'
database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'
cohorte_data = pd.read_csv(database_path, sep=';')
nb_overall_patient = cohorte_data.shape[0]
save=True

# Variables pour lesquelles l'absence rend le patient non interpretable
variables_not_missable = ['Age', 'Sexe',  'Lateralite', 'Type_I_ischemic_H_hemorrhagic', 
                          'COTE_1G_2D','TIV', 'TM', 'Delai_AVC_NIHSS_J3', 'NIHSS_J3',
                          'SAFE_J3', 'SAFE_J7','NIHSS_J7', 'Volume','Overlap_CST_cross',
                          'rMT_IPSI', 'rMT_CONTRO', 'PEM_plus1_ipsi', 'PEM_plus1_contro',
                          'PEMmax_ipsi', 'PEMmax_contro', 'FM_J7','Delai_follow_up_en_mois',
                          'FM_imput_M3', 'ARAT_imput_M3','delai_AVC_TMS', 'delai_AVC_IRM' ]
variables_not_missable += ['numero']

not_missable_variables_cohorte = cohorte_data[variables_not_missable].dropna()
nb_patient_selected = not_missable_variables_cohorte.shape[0]

print(f"Initial number of patient in database : {nb_overall_patient} ")
print(f"Non missable variables : {variables_not_missable}")
print(f"Number of selected patients : {nb_patient_selected}")

filtered_cohorte = cohorte_data.loc[cohorte_data['numero'].isin(not_missable_variables_cohorte['numero'])]

if save:
    filtered_cohorte.to_csv(working_directory_filtration + '/' + 'Filtred_cohorte.csv', sep=';', index=False)