#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 09:24:13 2025

@author: thomas.jacquemont
"""

import os
import subprocess
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_command(command):
    """Exécute une commande shell et lève une exception si elle échoue."""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: {command}")
        raise


def show_middle_slice(image_path, patient, cmap='gray', save_dir=None, overlay_path=None, alpha=0.5):
    """Sauvegarde une coupe médiane axiale pour le contrôle qualité."""
    img = nib.load(image_path)
    data = img.get_fdata()

    overlay_data = None
    if overlay_path and os.path.exists(overlay_path):
        overlay_data = nib.load(overlay_path).get_fdata()

    z_mid = data.shape[2] // 2
    slice_data = np.rot90(data[:, :, z_mid])
    if overlay_data is not None:
        overlay_slice = np.rot90(overlay_data[:, :, min(z_mid, overlay_data.shape[2]-1)])

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, cmap=cmap, origin='lower')
    if overlay_data is not None:
        plt.imshow(overlay_slice, cmap='autumn', alpha=alpha, origin='lower')
    plt.axis('off')
    plt.title(f'{patient} : {os.path.basename(image_path)}')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{patient}_{os.path.basename(image_path).replace('.nii.gz','.png')}")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ Figure sauvegardée : {save_path}")
    else:
        plt.show()


def process_patient(patient, native_b0, MNI_B0_brain_template, MNI_B0_brain_mask, QC_directory):
    """Traite un patient (registration, resampling, QC)."""
    try:
        image_patient_directory = os.path.dirname(native_b0)
        native_b0_bet = os.path.join(image_patient_directory, 'nodif_brain.nii.gz')

        # B0 registration to MNI
        B0_to_mni_matrix = os.path.join(image_patient_directory, 'native_b0_brain_to_MNI_B0_0GenericAffine.mat')
        B0_to_mni_warp = os.path.join(image_patient_directory, 'native_b0_brain_to_MNI_B0_1Warp.nii.gz')
        B0_to_mni = os.path.join(image_patient_directory, 'native_b0_brain_to_MNI_B0_Warped.nii.gz')
        B0_to_mni_1mm = os.path.join(image_patient_directory, 'nw_native_b0_to_MNI_1mm.nii.gz')
        B0_to_mni_2mm = os.path.join(image_patient_directory, 'nw_native_b0_to_MNI.nii.gz')

        cmd_reg = f"antsRegistrationSyN.sh -d 3 -f {MNI_B0_brain_template} -m {native_b0_bet} -o {os.path.join(image_patient_directory, 'native_b0_brain_to_MNI_B0_')}"
        run_command(cmd_reg)
        os.rename(B0_to_mni, B0_to_mni_1mm)
        run_command(f"ResampleImage 3 {B0_to_mni_1mm} {B0_to_mni_2mm} 2x2x2 0")
        show_middle_slice(B0_to_mni_1mm, patient, save_dir=QC_directory, overlay_path=MNI_B0_brain_template)

        # B1000
        native_b1000 = os.path.join(image_patient_directory, 'native_b1000.nii.gz')
        b1000_to_mni = os.path.join(image_patient_directory, 'native_b1000_to_MNI_B0_Warped.nii.gz')
        b1000_masked_1mm = os.path.join(image_patient_directory, 'native_b1000_masked_1mm.nii.gz')
        b1000_masked_2mm = os.path.join(image_patient_directory, 'aff_nr_nodif_brain_to_template_B0_THROMBO.nii.gz')

        run_command(f"antsApplyTransforms -d 3 -i {native_b1000} -r {MNI_B0_brain_template} "
                    f"-t {B0_to_mni_warp} -t {B0_to_mni_matrix} -o {b1000_to_mni} -n Linear")
        run_command(f"fslmaths {b1000_to_mni} -mas {MNI_B0_brain_mask} {b1000_masked_1mm}")
        run_command(f"ResampleImage 3 {b1000_masked_1mm} {b1000_masked_2mm} 2x2x2 0")
        show_middle_slice(b1000_masked_1mm, patient, save_dir=QC_directory, overlay_path=MNI_B0_brain_template)

        # ADC
        native_adc = os.path.join(image_patient_directory, 'native_ADC_map.nii')
        adc_to_mni = os.path.join(image_patient_directory, 'native_ADC_map_to_MNI_B0_Warped.nii.gz')
        adc_masked_1mm = os.path.join(image_patient_directory, 'native_ADC_masked_1mm.nii.gz')
        adc_masked_2mm = os.path.join(image_patient_directory, 'nw_native_ADC_map_with_brain_mask_to_MNI.nii.gz')

        run_command(f"antsApplyTransforms -d 3 -i {native_adc} -r {MNI_B0_brain_template} "
                    f"-t {B0_to_mni_warp} -t {B0_to_mni_matrix} -o {adc_to_mni} -n Linear")
        run_command(f"fslmaths {adc_to_mni} -mas {MNI_B0_brain_mask} {adc_masked_1mm}")
        run_command(f"ResampleImage 3 {adc_masked_1mm} {adc_masked_2mm} 2x2x2 0")
        show_middle_slice(adc_masked_1mm, patient, save_dir=QC_directory, overlay_path=MNI_B0_brain_template)

        print(f"✅ Patient {patient} traité avec succès.")
    except Exception as e:
        print(f"❌ Erreur pour le patient {patient}: {e}")


# --------------------------------------------------------------------------
MNI_B0_brain_template = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/DeepLearning_Motor_Prediction/Images_Database/Other_data/MNI152_B0_1mm.nii.gz'
MNI_B0_brain_mask = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/DeepLearning_Motor_Prediction/Images_Database/Other_data/MNI152_T1_1mm_brain_mask_dil.nii.gz'
databas_csv = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/DeepLearning_Motor_Prediction/PREP_Database/Base_PREP_Thomas.csv'
QC_directory = '/home/thomas.jacquemont/Test/PREP_AVC/Native_to_MNI/QC'

df = pd.read_csv(databas_csv)[['numero', 'IRM_B0_Raw']].dropna()
patients = df['numero'].values
paths = df['IRM_B0_Raw'].values

# Nombre de processus en parallèle
N_JOBS = 20

with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
    futures = [
        executor.submit(process_patient, patients[i], paths[i], MNI_B0_brain_template, MNI_B0_brain_mask, QC_directory)
        for i in range(len(patients))
    ]

    for future in as_completed(futures):
        future.result()  # pour propager les erreurs éventuelles
