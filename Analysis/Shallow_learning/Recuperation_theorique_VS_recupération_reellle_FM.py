#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:08:05 2025

@author: thomas
"""
import pandas as pd
import matplotlib.pyplot as plt


study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'

###############################################################################

working_directory_filtration = study_directory + '/Results/Patients_selection'
database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'

def plot_recuperation_relle_vs_theorique(database_path):
    """
    Crée un plot de 0.7*(60-FM_J7) en fonction de FM_imput_M3 - FM_J7,
    en colorant les points selon la variable binaire 'PEM_plus1_ipsi'.

    Args:
        csv_file (str): Chemin vers le fichier CSV contenant les données.
    """

    # Charger les données depuis le fichier CSV
    try:
        data = pd.read_csv(database_path, sep=';')
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{database_path}' n'a pas été trouvé.")
        return

    # Calculer les nouvelles colonnes
    data['y_axis'] = 0.7 * (60 - data['FM_J7'])
    data['x_axis'] = data['FM_imput_M3'] - data['FM_J7']

    # Séparer les données en fonction de 'PEM_plus1_ipsi'
    pem_plus = data[data['PEM_plus1_ipsi'] == 1.0]
    pem_moins = data[data['PEM_plus1_ipsi'] == 0.0]

    # Créer le plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pem_plus['x_axis'], pem_plus['y_axis'], color='red', label='PEM + (PEM_plus1_ipsi = 1)')
    plt.scatter(pem_moins['x_axis'], pem_moins['y_axis'], color='blue', label='PEM - (PEM_plus1_ipsi = 0)')

    # Ajouter des labels et un titre
    plt.xlabel('FM_imput_M3 - FM_J7')
    plt.ylabel('0.7 * (60 - FM_J7)')
    plt.title('Recupération FM theorique vs relle en fonction du PEM')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_recuperation_relle_vs_theorique(database_path)

###############################################################################