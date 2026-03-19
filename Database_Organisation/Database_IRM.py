#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:27:26 2025

@author: thomas.jacquemont
"""

import pandas as pd
import numpy as np
import os
import glob
import shutil


def formater_date_irm(date_irm):
    """Formate une date d'IRM de AAAAMMJJ en AAAA/MM/JJ."""
    annee = date_irm[:4]
    mois = date_irm[4:6]
    jour = date_irm[6:]
    return f"{jour}/{mois}/{annee}"

database_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
image_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/2_Final_database'
save= False

database = pd.read_csv(database_path, delimiter=';')
list_patient_image = os.listdir(image_directory)

list_patients_with_image_but_not_in_database = []
list_patients_with_image_but_multiple_in_database = []
list_patients_no_diffusion = []
list_patients_multiple_diffusion = []
list_patients_multiple_flair = []

##################### PROCESSING DE LA BASE DE DONNEE INITIALE #####################

for patient in list_patient_image:
    #################### Réinitialisétion des variables ########################
    patient_id = np.nan
    patient_index = np.nan
    diffusion = []
    flair = []
    
    #################### IMAGES and DATABASE CORRESPONDANCE #####################
    # Trying to find the occurence in the database from the directory name
    patient_lastname = patient.split("^")[0]
    patient_firstname = patient.split("^")[1]
    count_lastname = np.count_nonzero(database["Nom"].values == patient_lastname)
    if count_lastname == 1: # Ideal situation, the lastname is unique within the database
        patient_id = database.loc[(database["Nom"]==patient_lastname) & (database["Prenom"]==patient_firstname)]["Patient"].values[0]
    if count_lastname == 0:  # The Patient is not found, within the database, need to be check by hand
        print(f"Patient {patient} not found in database") 
        list_patients_with_image_but_not_in_database += [patient]
        continue
    elif count_lastname > 1: # The last name is not unique within the database, use the first name
        patient_id = database.loc[(database["Nom"]==patient_lastname) & (database["Prenom"]==patient_firstname)]["Patient"].values
        if len(patient_id)==0 :
            list_patients_with_image_but_not_in_database += [patient]
            print(f"Patient {patient_firstname} {patient_lastname} not found in database") 
            continue
        elif len(patient_id)>1:
            list_patients_with_image_but_multiple_in_database += patient
            print(f"Patient {patient_firstname} {patient_lastname} multiple found in database") 
            continue
        elif len(patient_id)==0 :
            patient_id = patient_id[0]
    patient_id = int(patient_id)
    
    ################### GRABBING PATIENT DIFFUSION +/- FLAIR FILES ############
    patient_image_directory = image_directory + '/' + patient
    os.chdir(patient_image_directory)
    diffusion = glob.glob(os.path.join(patient_image_directory, '*/*/*Diff*.nii'))
    flair =  glob.glob(os.path.join(patient_image_directory, '*/*/*FLAIR*.nii'))
    if len(diffusion)>1:
        print(f"Multiple diffusion found for {patient}") 
        list_patients_multiple_diffusion += [patient]
        diffusion.sort()
        diffusion_path = diffusion[0]        
        date_diffusion = formater_date_irm(diffusion_path.split('/')[-3][:8])
    elif len(diffusion)==0:
        print(f"No diffusion find for patient {patient}")
        list_patients_no_diffusion += [patient]
        pass
    elif len(diffusion)==1:
        diffusion_path = diffusion[0]
        date_diffusion = formater_date_irm(diffusion_path.split('/')[-3][:8])
    
    ################ FILLIN THE DATABASE ACCORDING TO IMAGE DATA ##############
    patient_index = database.index[database['Patient']==patient_id]
    database.loc[patient_index,'IRM_Diffusion'] = 1
    database.loc[patient_index,'Date_Diffusion'] = date_diffusion
    database.loc[patient_index,'IRM_Diffusion_path'] = diffusion_path
    
    if len(flair)==1:
        database.loc[patient_index, 'IRM_FLAIR'] = 1
        database.loc[patient_index,'IRM_FLAIR_path'] = flair[0]
    elif len(flair)==0:
        database.loc[patient_index,'IRM_FLAIR'] = 0
        database.loc[patient_index,'IRM_FLAIR_path'] = np.nan
    elif len(flair)>1:
        print(f"Multiple FLAIR find for patient {patient}")
        list_patients_multiple_flair += [patient]
        flair.sort()
        database.loc[patient_index,'IRM_FLAIR_path'] = flair[0]
        database.loc[patient_index,'IRM_FLAIR'] = 1

if save:
    database.to_csv(database_path, sep=';', index=False)

########## Intégration des IRM nifti et mask normalisées vers le template FA ############
# Le but de cette partie du code est d'integrer dans la base de donnée les IRM normalisée
# IRM normalisées
normalised_image_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/AFF_NR_diffusion'
list_patient_normalised_images = os.listdir(normalised_image_directory)

list_patients_with_norm_image_but_not_in_image_database = []
list_patients_with_norm_image_and_multiple_directory_in_image_database = []
list_patients_with_norm_image_and_directory_but_no_diffusion_in_directory = []
list_patients_with_norm_image_and_directory_but_multiple_diffusion_in_directory = []

copy_rmi = False

for patient_norm in list_patient_normalised_images:
    raw_diffusion_path = ''
    patient_image_directory_path = ''
    patient_image_directory = ''
    patient_lastname = patient_norm.split('_')[-3].upper()
    patient_image_directory_path = image_directory + '/' + patient_lastname +'^*'
    patient_image_directory = glob.glob(patient_image_directory_path)
    if len(patient_image_directory)==0:
        list_patients_with_norm_image_but_not_in_image_database += [patient_lastname]
        print(f'{patient_lastname} Have a normalized file but no raw data')
    elif len(patient_image_directory)>1:
        list_patients_with_norm_image_and_multiple_directory_in_image_database += [patient_lastname]
        print(f'{patient_lastname} Have a normalized file and multiple directory in raw data')
    elif len(patient_image_directory)==1:
        patient_image_directory = patient_image_directory[0]
        diffusion_files_list = glob.glob(os.path.join(patient_image_directory, '*/*/*Diff*.nii'))
        if len(diffusion_files_list)==0:
            list_patients_with_norm_image_and_directory_but_no_diffusion_in_directory += [patient_lastname]
            print(f'{patient_lastname} Have a normalized file and a directory in raw data but no raw diffusion')
        elif len(diffusion_files_list)>1:
            list_patients_with_norm_image_and_directory_but_multiple_diffusion_in_directory += [patient_lastname]
            print(f'{patient_lastname} Have a normalized file and a directory in raw data but but multiple raw diffusion')
        elif len(diffusion_files_list)==1:
            raw_diffusion_path = diffusion_files_list[0]
            directory = os.path.dirname(raw_diffusion_path)
            if copy_rmi:               
                shutil.copy(normalised_image_directory + '/' + patient_norm, directory +'/' + patient_norm) 
    
# Mak normalisés
# Le but de cette partie du code est d'integrer dans la base de donnée les IRM normalisée

normalised_mask_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/lesions/Mask'
list_patient_normalised_mask = os.listdir(normalised_mask_directory)

list_patients_with_norm_mask_but_not_in_image_database = []
list_patients_with_norm_mask_and_multiple_directory_in_image_database = []
list_patients_with_norm_mask_and_directory_but_no_diffusion_in_directory = []
list_patients_with_norm_mask_and_directory_but_multiple_diffusion_in_directory = []

copy_mask = False

for patient_norm in list_patient_normalised_mask:
    raw_diffusion_path = ''
    patient_mask_directory_path = ''
    patient_mask_directory = ''
    patient_lastname = patient_norm.split('_')[-4].upper()
    patient_image_directory_path = image_directory + '/' + patient_lastname +'^*'
    patient_image_directory = glob.glob(patient_image_directory_path)
    if len(patient_image_directory)==0:
        list_patients_with_norm_mask_but_not_in_image_database += [patient_lastname]
        print(f'{patient_lastname} Have a normalized file but no raw data')
    elif len(patient_image_directory)>1:
        list_patients_with_norm_mask_and_multiple_directory_in_image_database += [patient_lastname]
        print(f'{patient_lastname} Have a normalized file and multiple directory in raw data')
    elif len(patient_image_directory)==1:
        patient_image_directory = patient_image_directory[0]
        diffusion_files_list = glob.glob(os.path.join(patient_image_directory, '*/*/*Diff*.nii'))
        if len(diffusion_files_list)==0:
            list_patients_with_norm_mask_and_directory_but_no_diffusion_in_directory += [patient_lastname]
            print(f'{patient_lastname} Have a normalized file and a directory in raw data but no raw diffusion')
        elif len(diffusion_files_list)>1:
            list_patients_with_norm_mask_and_directory_but_multiple_diffusion_in_directory += [patient_lastname]
            print(f'{patient_lastname} Have a normalized file and a directory in raw data but but multiple raw diffusion')
        elif len(diffusion_files_list)==1:
            raw_diffusion_path = diffusion_files_list[0]
            directory = os.path.dirname(raw_diffusion_path)
            if copy_mask :
                shutil.copy(normalised_mask_directory + '/' + patient_norm, directory +'/' + patient_norm)

################ AJOUT DE LA BASE DE THOMAS CHECKOURI #########################

database_checkouri = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/Thomas_Checkouri_database'
diffusion_checkouri = database_checkouri + '/Diffusion_database'
mask_checkouri = database_checkouri + '/Mask_database'
list_patient_checkouri = os.listdir(diffusion_checkouri)
database_path = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
database = pd.read_csv(database_path, delimiter=';')

move_file = False #should be done only once
list_patient_in_checkouri_but_not_in_image_database = []
list_patient_in_checkouri_and_in_image_database = []
list_patient_in_checkouri_and_multiple_in_image_database = []
list_patient_in_checkouri_and_not_in_database = []
list_patient_in_checkouri_and_multiple_in_database = []

for patient in list_patient_checkouri:
    patient_lastname = patient.split('final')[1].split('.')[0]
    diffusion_path =  glob.glob(diffusion_checkouri + f'/*{patient_lastname}*')[0]
    mask_path =  glob.glob(mask_checkouri + f'/*{patient_lastname}*')[0]
    patient_lastname_upper = patient_lastname.upper()
    patient_image_directory_path = image_directory + '/' + patient_lastname_upper  +'^*'
    patient_image_directory = glob.glob(patient_image_directory_path)
    if len(patient_image_directory)==0 :
        patient_index = database.index[database['Nom']==patient_lastname_upper]
        if len(patient_index)==1:
            patient_firstname = database['Prenom'].iloc[patient_index].values[0]
            patient_firstname_upper = patient_firstname.upper()
            patient_image_directory = image_directory + f'/{patient_lastname_upper}^{patient_firstname_upper}/00000000000000/Ax_Diff_3mm_HB_b1000'
            if move_file:
                os.makedirs(patient_image_directory, exist_ok=True)
                shutil.copy(diffusion_path, patient_image_directory + '/' +  diffusion_path.split('/')[-1])
                shutil.copy(mask_path, patient_image_directory + '/' +  mask_path.split('/')[-1])
            database.loc[patient_index,'Lesion_mask'] = 1
            database.loc[patient_index,'Lesion_mask_path'] = patient_image_directory + '/' +  diffusion_path.split('/')[-1]
            database.loc[patient_index,'IRM_Diffusion_Norm'] = 1
            database.loc[patient_index,'IRM_Diffusion_Norm_path'] = patient_image_directory + '/' +  mask_path.split('/')[-1]
        if len(patient_index)==0:
            print(f'{patient_lastname_upper} in Checkouri database and not in PREP database')
            list_patient_in_checkouri_and_not_in_database += [patient_lastname_upper]
        if len(patient_index)>1:
            print(f'{patient_lastname_upper} in Checkouri database and multiple file in PREP database')
            list_patient_in_checkouri_and_multiple_in_database += [patient_lastname_upper]
    elif len(patient_image_directory)>0:
        print(f'{patient_lastname_upper} in Checkouri database and multiple file in image database')
        list_patient_in_checkouri_and_multiple_in_image_database += [patient_lastname_upper]
    elif len(patient_image_directory)==1:
        print(f'{patient_lastname_upper} in Checkouri database and in image database')
        list_patient_in_checkouri_and_in_image_database += [patient_lastname_upper]
        

if save:
    database.to_csv(database_path, sep=';', index=False)
    
################### DATABASE ANNONYMISATION ###################################

save_anonyme = False # réaliser qu'une seule fois

for patient in list_patient_image:
    #################### Réinitialisétion des variables ########################
    patient_id = np.nan
    patient_index = np.nan
    patient_number = ''
    diffusion = []
    flair = []
    new_raw_irm_path = ''
    new_raw_flair_path = ''
    new_norm_irm_path = ''
    new_norm_mask_path = ''

    #################### IMAGES and DATABASE CORRESPONDANCE #####################
    # Trying to find the occurence in the database from the directory name
    patient_image_directory = image_directory + '/' + patient
    patient_lastname = patient.split("^")[0]
    patient_firstname = patient.split("^")[1]
    patient_index = database.index[(database["Nom"]==patient_lastname) & (database["Prenom"]==patient_firstname)]
    if len(patient_index)==1:
        patient_number = database.loc[patient_index,"numero"]
        patient_number = patient_number.values[0].replace(' ','')
        raw_irm_path = database.loc[patient_index, "Raw_IRM_Diffusion_path"].values[0]
        raw_flair_path = database.loc[patient_index, "Raw_IRM_FLAIR_path"].values[0]
        norm_irm_path = database.loc[patient_index, "IRM_Diffusion_Norm_path"].values[0]
        norm_mask_path = database.loc[patient_index, "Lesion_Mask_Norm_path"].values[0]
        old_directory_name = patient_image_directory.split('/')[-1]
        os.rename(patient_image_directory, image_directory + '/' + str(patient_number))
        if pd.isna(raw_irm_path)==False:
        	new_raw_irm_path = raw_irm_path.replace(patient, patient_number)
        	database.loc[patient_index, "Raw_IRM_Diffusion_path"] = new_raw_irm_path
        if pd.isna(raw_flair_path)==False:
        	new_raw_flair_path = raw_flair_path.replace(patient, patient_number)
        	database.loc[patient_index, "Raw_IRM_FLAIR_path"] = new_raw_flair_path
        if pd.isna(norm_irm_path)==False:
        	new_norm_irm_path = norm_irm_path.replace(patient, patient_number)
        	rename_norm_irm_path = new_norm_irm_path.replace(norm_irm_path.split('/')[-1], 'aff_nr_nodif_brain_to_template_B0_THROMBO.nii')
        	os.rename(new_norm_irm_path, rename_norm_irm_path)
        	database.loc[patient_index, "IRM_Diffusion_Norm_path"] = rename_norm_irm_path
        if pd.isna(norm_mask_path)==False:
        	new_norm_mask_path = norm_mask_path.replace(patient, patient_number)
        	rename_norm_mask_path = new_norm_mask_path.replace(norm_mask_path.split('/')[-1], 'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask.nii')
        	os.rename(new_norm_mask_path, rename_norm_mask_path)
        	database.loc[patient_index, "Lesion_Mask_Norm_path"] = rename_norm_mask_path
    else:
        print(f"ERROR on the patient {patient}")

if save_anonyme:
    database.to_csv(database_path, sep=';', index=False)

#################### CALCULS DES DELAIS D'INTERET ##########################"

def calculer_delais(df):
    """
    Calcule les délais en jours entre la date de l'AVC et d'autres événements.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame avec les nouvelles colonnes de délai.
    """

    df['Date_AVC'] = pd.to_datetime(df['Date_AVC'])
    df['Date_TMS'] = pd.to_datetime(df['Date_TMS'])
    df['Date_IRM'] = pd.to_datetime(df['Date_IRM'])
    df['Date_Prelevement'] = pd.to_datetime(df['Date_Prelevement'])

    df['delai_AVC_TMS'] = (df['Date_TMS'] - df['Date_AVC']).dt.days
    df['delai_AVC_IRM'] = (df['Date_IRM'] - df['Date_AVC']).dt.days
    df['delai_AVC_prelevement'] = (df['Date_Prelevement'] - df['Date_AVC']).dt.days

    return df