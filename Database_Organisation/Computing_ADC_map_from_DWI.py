#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 14:32:32 2025

@author: thomas.jacquemont
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import subprocess

def run_command(command):
#    """Executes a shell command and raises an exception if it fails."""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        raise

# Racine de la base de données
database_dir = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/2_Final_database'
path_database = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'
data = pd.read_csv(path_database, sep=';')
subject_issue = []

for subject_path in glob.glob(os.path.join(database_dir, "*")):
    # Réinitialisation des variables
    subject_id = None
    diff_folders = None
    diff_folder = None
    nii_files = None
    bval_files = None
    bvec_files= None
    nii_path = None
    nii_filename= None
    bvec_path = None
    bvec_path = None
    mif_path = None
    mask_path = None
    mask_nii_path = None
    img = None
    data_img = None
    mask_img = None
    brain_mask = None
    bvals = None
    idx_b0 = None
    idx_b1000 = None
    data_b0 = None
    data_b1000 = None
    valid_mask = None
    b0_value = None
    b1000_value = None
    adc_out_path = None
    adc_img = None 
    
    subject_id = os.path.basename(os.path.normpath(subject_path))
    
    try :
        # Cherche les dossiers contenant "Ax_Diff"
        diff_folders = [
            f for f in glob.glob(os.path.join(subject_path, "**/*Ax_Diff*"), recursive=True)
            if os.path.isdir(f)
        ]
    
        if not diff_folders:
            subject_issue += [subject_id]
            continue
    
        diff_folder = diff_folders[0]
        nii_files = glob.glob(os.path.join(diff_folder, "*Ax_Diff*.nii*"))
        bval_files = glob.glob(os.path.join(diff_folder, "*Ax_Diff*.bval"))
        bvec_files = glob.glob(os.path.join(diff_folder, "*Ax_Diff*.bvec"))
    
        if not nii_files or not bval_files or not bvec_files:
            subject_issue += [subject_id]
            continue
    
        nii_path = nii_files[0]
        nii_filename = os.path.basename(nii_path).split(".")[0]
        bval_path = bval_files[0]
        bvec_path = bvec_files[0]
    
        print(f"Traitement : {nii_path}")
    
        mif_path = os.path.join(diff_folder, nii_filename + ".mif")
        mask_path = os.path.join(diff_folder, nii_filename + "_mask.mif")
        mask_nii_path = os.path.join(diff_folder, nii_filename + "_mask.nii.gz")
    
        # Convertir en .mif avec gradient info
        mr_convert_cmd = f"mrconvert {nii_path} {mif_path} --fslgrad {bvec_path} {bval_path} --force"
        run_command(mr_convert_cmd)
    
        # Calcul du brain mask
        dw2mask_cmd = f"dwi2mask {mif_path} {mask_path} --force"
        run_command(dw2mask_cmd)
    
        # Convertir le mask en NIfTI
        mr_convert_bis_cmd = f"mrconvert {mask_path} {mask_nii_path} --force"
        run_command(mr_convert_bis_cmd)
    
        # Charger imagerie diffusion et mask
        img = nib.load(nii_path)
        data_img = img.get_fdata()
    
        mask_img = nib.load(mask_nii_path)
        brain_mask = mask_img.get_fdata().astype(bool)
    
        # Charger bvals
        with open(bval_path, "r") as f:
            bvals = np.array([float(x) for x in f.read().split()])
    
        idx_b0 = np.where(bvals == 0)[0]
        idx_b1000 = np.where(bvals != 0)[0]
    
        if len(idx_b0) == 0 or len(idx_b1000) == 0:
            print(f"⚠ Pas de b0 ou b1000 trouvé pour {nii_path}")
            continue
    
        data_b0 = np.mean(data_img[..., idx_b0], axis=3)
        data_b1000 = np.mean(data_img[..., idx_b1000], axis=3)
    
        valid_mask = brain_mask & (data_b0 > 0)
    
        b0_value = np.mean(bvals[idx_b0])
        b1000_value = np.mean(bvals[idx_b1000])
    
        adc_map = np.zeros_like(data_b0)
        adc_map[valid_mask] = -np.log(data_b1000[valid_mask] / data_b0[valid_mask]) / (b1000_value - b0_value)
    
        # Sauvegarde ADC
        adc_img = nib.Nifti1Image(adc_map, img.affine, img.header)
        adc_out_path = os.path.join(diff_folder, nii_filename + "_ADC_map_with_brain_mask.nii")
        nib.save(adc_img, adc_out_path)
    
        print(f"✅ ADC sauvegardée : {adc_out_path}")
    
        # Mettre à jour le DataFrame
        data.loc[data['numero'] == subject_id, 'Raw_IRM_Diffusion_path'] = nii_path
        data.loc[data['numero'] == subject_id, 'Raw_ADC_path'] = adc_out_path
    
    except Exception as e:
        print(f"❌ Erreur pour {subject_id} : {e}")
        subject_issue.append(subject_id)
        continue


def compute_ADC_map(subject_id, nii_files,
                    database_dir='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/2_Final_database', 
                    path_database='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv'):
    """
    Function used to compute ADC map on patient for which the for loop didn't work' .
    The CSV database should be 

    Parameters
    ----------
    subject_id : TYPE  STR : subject ID
        DESCRIPTION.
    database_dir : TYPE, optional
        DESCRIPTION. The default is '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/2_Final_database'.

    Returns
    -------
    None.

    """
    
    data = pd.read_csv(path_database, sep=';')
    nii_filename = os.path.basename(nii_files).split(".")[0]
    diff_folder = os.path.dirname(nii_files)    
    bval_files = os.path.join(diff_folder, nii_filename + ".bval")
    bvec_files = os.path.join(diff_folder, nii_filename + ".bvec")
      
    print(f"Traitement : {nii_files}")
        
    mif_path = os.path.join(diff_folder, nii_filename + ".mif")
    mask_path = os.path.join(diff_folder, nii_filename + "_mask.mif")
    mask_nii_path = os.path.join(diff_folder, nii_filename + "_mask.nii.gz")
        
    # Convertir en .mif avec gradient info
    mr_convert_cmd = f"mrconvert {nii_files} {mif_path} --fslgrad {bvec_files} {bval_files} --force"
    run_command(mr_convert_cmd)
        
    # Calcul du brain mask
    dw2mask_cmd = f"dwi2mask {mif_path} {mask_path} --force"
    run_command(dw2mask_cmd)
        
    # Convertir le mask en NIfTI
    mr_convert_bis_cmd = f"mrconvert {mask_path} {mask_nii_path} --force"
    run_command(mr_convert_bis_cmd)
        
    # Charger imagerie diffusion et mask
    img = nib.load(nii_files)
    data_img = img.get_fdata()
        
    mask_img = nib.load(mask_nii_path)
    brain_mask = mask_img.get_fdata().astype(bool)
        
    # Charger bvals
    with open(bval_files, "r") as f:
         bvals = np.array([float(x) for x in f.read().split()])
        
    idx_b0 = np.where(bvals == 0)[0]
    idx_b1000 = np.where(bvals != 0)[0]
        
    if len(idx_b0) == 0 or len(idx_b1000) == 0:
         print(f"⚠ Pas de b0 ou b1000 trouvé pour {nii_files}")
               
    data_b0 = np.mean(data_img[..., idx_b0], axis=3)
    data_b1000 = np.mean(data_img[..., idx_b1000], axis=3)
        
    valid_mask = brain_mask & (data_b0 > 0)
        
    b0_value = np.mean(bvals[idx_b0])
    b1000_value = np.mean(bvals[idx_b1000])
        
    adc_map = np.zeros_like(data_b0)
    adc_map[valid_mask] = -np.log(data_b1000[valid_mask] / data_b0[valid_mask]) / (b1000_value - b0_value)
        
    # Sauvegarde ADC
    adc_img = nib.Nifti1Image(adc_map, img.affine, img.header)
    adc_out_path = os.path.join(diff_folder, nii_filename + "_ADC_map_with_brain_mask.nii")
    nib.save(adc_img, adc_out_path)
        
    print(f"✅ ADC sauvegardée : {adc_out_path}")
        
    # Mettre à jour le DataFrame
    data.loc[data['numero'] == subject_id, 'Raw_IRM_Diffusion_path'] = nii_files
    data.loc[data['numero'] == subject_id, 'Raw_ADC_path'] = adc_out_path
    data.to_csv(path_database, sep=';', index=False)