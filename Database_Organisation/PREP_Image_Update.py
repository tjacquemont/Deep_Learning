#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:21:23 2025

@author: thomas.jacquemont
"""

import pandas as pd
import numpy as np
import os
import shutil

database_csv_path = '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/DeepLearning_Motor_Prediction/PREP_Database/Base_PREP_Thomas.csv'
database_csv = pd.read_csv(database_csv_path)

database_folder = '/network/iss/cenir/analyse/irm/users/thomas.jacquemont/PREP_AVC/DeepLearning_Motor_Prediction/Images_Database/2_Final_database'
new_patients_folder = '/home/thomas.jacquemont/Test/PREP_AVC/1_Database_Update/Image_Database/20251217/New_patient_full_image_data/NIFTI'

patient_list = os.listdir(new_patients_folder)

concordance_image_database = {
    'mask_lesion_to_MNI_1mm.nii' : 'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask_to_mni_1mm.nii',
     'aff_nr_nodif_brain_to_template_B0_THROMBO.nii.gz' : 'nw_native_b1000_with_brain_mask_to_MNI.nii.gz', 
     'aff_nr_nodif_brain_to_template_B0_THROMBO.nii' : 'nw_native_b1000_with_brain_mask_to_MNI.nii',
     'native_b1000.nii.gz' : 'native_b1000.nii.gz',
     'native_B1000_B0.nii.gz' : 'native_B1000_B0.nii.gz',
     'native_B0_B1000.bvec' : 'native_B0_B1000.bvec',
     'native_ADC_map.nii': 'native_ADC_map.nii',
     'mask_lesion_to_MNI_2mm.nii': 'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask_to_mni.nii',
     'native_B0_B1000.bval' : 'native_B0_B1000.bval' ,
     'native_ADC_map_to_MNI_B0_Warped.nii.gz': 'nnw_native_ADC_map_to_MNI.nii.gz',
     'nw_native_ADC_map_with_brain_mask_to_MNI.nii.gz' : 'nw_native_ADC_map_with_brain_mask_to_MNI.nii.gz',
     'native_B0_B1000.json':'native_B0_B1000.json',
     'native_b0_brain_to_MNI_B0_1Warp.nii.gz' : 'native_b0_brain_to_MNI_B0_1Warp.nii.gz',
     'nw_native_b0_to_MNI_1mm.nii.gz' : 'nw_native_b0_to_MNI_1mm.nii.gz',
     'native_b0.nii.gz':'native_b0.nii.gz',
     'native_b0_brain_to_MNI_B0_InverseWarped.nii.gz':'native_b0_brain_to_MNI_B0_InverseWarped.nii.gz',
     'native_B0_B1000.nii' : 'native_B0_B1000.nii',
     'native_ADC_masked_1mm.nii.gz' : 'native_ADC_masked_1mm.nii.gz',
     'nodif_brain.nii.gz' : 'nodif_brain.nii.gz',
     'native_b0_brain_to_MNI_B0_0GenericAffine.mat' : 'native_b0_brain_to_MNI_B0_0GenericAffine.mat',
     'nw_native_b0_to_MNI.nii.gz' : 'nw_native_b0_to_MNI.nii.gz',
     'native_b0_brain_to_MNI_B0_1InverseWarp.nii.gz' : 'native_b0_brain_to_MNI_B0_InverseWarped.nii.gz',
     'native_b1000_masked_1mm.nii.gz' : 'native_b1000_masked_1mm.nii.gz',
     'native_b1000_to_MNI_B0_Warped.nii.gz': 'native_b1000_to_MNI_B0_Warped.nii.gz' 
    }

concordance_image_CSV = {
   'native_b1000.nii.gz' : 'IRM_Diffusion_Raw',
   'nw_native_b1000_with_brain_mask.nii.gz' : 'IRM_Diffusion_Norm',
   'nw_native_b1000_with_brain_mask_to_MNI.nii.gz' : 'IRM_Diffusion_Norm_to_MNI',
   'native_b0.nii.gz' : 'IRM_B0_Raw',
   'nw_native_b0.nii.gz' : 'IRM_B0_Norm',
   'nw_native_b0_to_MNI.nii.gz' : 'IRM_B0_Norm_to_MNI',
   'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask.nii' : 'Lesion_Mask_Norm',
   'aff_nr_nodif_brain_to_template_B0_THROMBO_Mask_to_mni.nii' : 'Lesion_Mask_Norm_to_MNI',
   'native_ADC_map.nii' : 'ADC_Raw',
   'nw_native_ADC_map_with_brain_mask.nii.gz' : 'ADC_Norm',
   'nw_native_ADC_map_with_brain_mask_to_MNI.nii.gz' : 'ADC_Norm_to_MNI'
    }

Patient_sub_directory_already_exist = []

for patient_subdir in patient_list:
    pib = patient_subdir.split('_')[1]
    patient_source_directory = os.path.join(new_patients_folder, patient_subdir)
    patient_image_database_subfolder = os.path.join(database_folder, pib)
    if not os.path.exists(patient_image_database_subfolder):
        image_patient_subfolder = os.path.join(patient_image_database_subfolder, '00000000000000', 'Ax_Diff_3mm_HB_b1000')
        os.makedirs(image_patient_subfolder)
    image_patient_subfolder = os.path.join(patient_image_database_subfolder, '00000000000000', 'Ax_Diff_3mm_HB_b1000')
    for image_file in os.listdir(patient_source_directory):
        image_file_source_path = os.path.join(patient_source_directory, image_file)
        image_file_destination_path = os.path.join(image_patient_subfolder, concordance_image_database[image_file])
        if concordance_image_database[image_file] in concordance_image_CSV.keys():
            database_csv.loc[database_csv['numero']==pib, concordance_image_CSV[concordance_image_database[image_file]]] = image_file_destination_path
        #shutil.copy(image_file_source_path, image_file_destination_path)
        print('----------------------------------------')
        print(f'File {image_file_source_path} copy to {image_file_destination_path}')
    else :
        Patient_sub_directory_already_exist += [patient_subdir]

database_csv.to_csv(database_csv_path, index=False)