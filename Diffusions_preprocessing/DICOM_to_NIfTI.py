#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:53:39 2025

@author: thomas.jacquemont
"""

import os
import subprocess

# Convert all the DICOM find in the folders using dcmt2niix
# This script assums that the DICOMS are in unitary subfolder in dicom_data  

def run_command(command):
#    """Executes a shell command and raises an exception if it fails."""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        raise
    
dicom_data = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/RAW_DICOM/RAW_DICOM'
nifti_converted_folder = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Images_Database/RAW_DICOM/nifti'

list_of_dicom_folder = os.listdir(dicom_data)

for folder in list_of_dicom_folder:
    os.chdir(dicom_data + '/' + folder)
    conversion_cmd = f'dcm2niix -ba y -f %n/%t/S_%s_%d/v_%t_S%s_%d -o {nifti_converted_folder} *'
    run_command(conversion_cmd)
    print("################################################################")
    os.chdir(dicom_data)