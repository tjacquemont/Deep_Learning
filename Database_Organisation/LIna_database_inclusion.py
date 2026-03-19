#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:30:21 2025

@author: thomas.jacquemont
"""

import os
import shutil
import pandas as pd

def integrer_données_neuroimagerie(source_dir, dest_dir, correspondance_file, base_prep_file):
    """
    Intègre les données de neuroimagerie dans la base de données finale et met à jour le fichier Base_PREP_Thomas.csv.

    Args:
        source_dir (str): Le chemin du dossier source contenant les données à intégrer.
        dest_dir (str): Le chemin du dossier de destination (base de données finale).
        correspondance_file (str): Le chemin du fichier .ods contenant la correspondance nom/numéro_anonymisation.
        base_prep_file (str): Le chemin du fichier CSV Base_PREP_Thomas.csv.
    """

    # Charger le fichier de correspondance
    df_correspondance = pd.read_excel(correspondance_file, engine='odf')

    # Charger le fichier Base_PREP_Thomas.csv
    df_base_prep = pd.read_csv(base_prep_file, delimiter=';')

    # Parcourir les dossiers dans le dossier source
    for patient_dir in os.listdir(source_dir):
        patient_path = os.path.join(source_dir, patient_dir)
        if os.path.isdir(patient_path):
            # Obtenir le nom du patient (en majuscules)
            patient_name_upper = patient_dir.upper()

            # Trouver le numéro d'anonymisation correspondant
            numero_anonymisation = df_correspondance.loc[df_correspondance['Nom'] == patient_name_upper, 'numero'].values[0]

            # Créer le dossier de destination pour le patient
            dest_patient_dir = os.path.join(dest_dir, numero_anonymisation, '00000000000000', 'Ax_Diff_3mm_HB_b1000')
            os.makedirs(dest_patient_dir, exist_ok=True)

            # Initialiser les valeurs pour le fichier Base_PREP_Thomas.csv
            raw_diffusion = 0.0
            raw_diffusion_path = ''
            diffusion_norm = 0.0
            diffusion_norm_path = ''
            lesion_mask_norm = 0.0
            lesion_mask_norm_path = ''

            # Copier et renommer les fichiers
            for filename in os.listdir(patient_path):
                source_file = os.path.join(patient_path, filename)
                if os.path.isfile(source_file):
                    if '.nii' in filename and '.gz' and '_mask' not in filename :
                        dest_file = os.path.join(dest_patient_dir, 'v_00000000000000_S3_Ax_Diff_b1000_3mm.nii')
                        raw_diffusion = 1.0
                        raw_diffusion_path = dest_file
                    elif '.bval' in filename:
                        dest_file = os.path.join(dest_patient_dir, 'v_00000000000000_S3_Ax_Diff_b1000_3mm.bval')
                    elif '.bvec' in filename:
                        dest_file = os.path.join(dest_patient_dir, 'v_00000000000000_S3_Ax_Diff_b1000_3mm.bvecs')
                    elif '.json' in filename:
                        dest_file = os.path.join(dest_patient_dir, 'v_00000000000000_S3_Ax_Diff_b1000_3mm.json')
                    elif '_to_FA.nii.gz' in filename:
                        dest_file = os.path.join(dest_patient_dir, 'aff_nr_nodif_brain_to_template_B0_THROMBO.nii.gz')
                        diffusion_norm = 1.0
                        diffusion_norm_path = dest_file
                    elif '_mask.nii' in filename:
                        dest_file = os.path.join(dest_patient_dir, 'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask.nii')
                        lesion_mask_norm = 1.0
                        lesion_mask_norm_path = dest_file
                    else:
                        continue
                    shutil.copy2(source_file, dest_file)

            # Mettre à jour le fichier Base_PREP_Thomas.csv
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'Raw_IRM_Diffusion'] = raw_diffusion
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'Raw_IRM_Diffusion_path'] = raw_diffusion_path
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'IRM_Diffusion_Norm'] = diffusion_norm
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'IRM_Diffusion_Norm_path'] = diffusion_norm_path
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'Lesion_Mask_Norm'] = lesion_mask_norm
            df_base_prep.loc[df_base_prep['numero'] == numero_anonymisation, 'Lesion_Mask_Norm_path'] = lesion_mask_norm_path

    # Sauvegarder le fichier Base_PREP_Thomas.csv modifié
    df_base_prep.to_csv(base_prep_file, index=False, sep=';')

# Exemple d'utilisation
source_dir = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/Database_Charlotte_Lina_a_integrer_dans_la_database'
dest_dir = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/Test_integration_database'
correspondance_file = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/Other_data/database_charlotte_lina.ods'
base_prep_file = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'

integrer_données_neuroimagerie(source_dir, dest_dir, correspondance_file, base_prep_file)