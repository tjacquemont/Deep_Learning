#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:35:39 2025

@author: thomas.jacquemont
"""
# HarvardOxford lateralisation
import nibabel as nib
import numpy as np
import pandas as pd
import os

# Charger l’atlas
atlas_path = "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
label_path = "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/HarvardOxford_labels.tsv"
img = nib.load(atlas_path)
atlas = img.get_fdata().astype(int)

# Trouver le plan sagittal
x_mid = atlas.shape[0] // 2

# Copier
atlas_lat = atlas.copy()

# Côté droit : on ajoute un offset de 48
atlas_lat[x_mid:, ...] = np.where(atlas_lat[x_mid:, ...] > 0, atlas_lat[x_mid:, ...] + 48, 0)

# Créer la table des labels
labels = pd.read_csv(label_path, sep="\t")
labels_left = labels.copy()
labels_right = labels.copy()
labels_left["hemi"] = "L"
labels_right["hemi"] = "R"
labels_right["label"] = labels_right["label"] + 48
labels_lr = pd.concat([labels_left, labels_right], ignore_index=True)

# Sauvegarde
out_image_path = os.path.join('/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas','HarvardOxford_cortical_lat.nii.gz' )
out_label_path = os.path.join('/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas',"HarvardOxford_labels_lat.tsv" )

nib.save(nib.Nifti1Image(atlas_lat, img.affine, img.header),
         out_image_path)
labels_lr.to_csv(out_label_path, sep="\t", index=False)

print("✅ Atlas latéralisé : HarvardOxford_cortical_lat.nii.gz")
print("✅ Labels TSV : HarvardOxford_labels_lat.tsv")

# Cerebellum adding
import os
import numpy as np
import nibabel as nib
import pandas as pd

# --- Chemins ---
base_dir = "/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/HarvarOxford_Lat_with_Cerebellum_computation"

# Atlas Harvard–Oxford déjà latéralisé
cort_path = os.path.join(base_dir, "HarvardOxford-cort-maxprob-thr25-1mm_lat.nii.gz")
label_path = os.path.join(base_dir, "HarvardOxford_labels_lat.tsv")

# Atlas du cervelet (FSL)
cereb_path = os.path.join(base_dir, "Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz")


# Fichiers de sortie
out_image_path = os.path.join(base_dir, "HarvardOxford_Lat_Cerebellum_combined-maxprob-thr25-1mm.nii.gz")
out_label_path = os.path.join(base_dir, "HarvardOxford_Lat_Cerebellum_combined.tsv")

# --- Charger les atlas ---
cort_img = nib.load(cort_path)
cort = cort_img.get_fdata().astype(int)

cereb_img = nib.load(cereb_path)
cereb = cereb_img.get_fdata().astype(int)

# --- Vérifier dimensions ---
if cort.shape != cereb.shape:
    raise ValueError("Dimensions mismatch! Les deux atlas doivent être en MNI 1mm (182x218x182).")

# --- Identifier les deux hémisphères du cervelet ---
x_mid = cereb.shape[0] // 2
cereb_L = np.where((cereb > 0) & (np.arange(cereb.shape[0])[:, None, None] < x_mid), 1, 0)
cereb_R = np.where((cereb > 0) & (np.arange(cereb.shape[0])[:, None, None] >= x_mid), 1, 0)

# --- Ajouter les labels 97 & 98 ---
combined = cort.copy()
combined[cereb_L > 0] = 97
combined[cereb_R > 0] = 98

# --- Mettre à jour les labels TSV ---
labels = pd.read_csv(label_path, sep="\t")

# Ajouter les 2 nouvelles lignes
labels = pd.concat([
    labels,
    pd.DataFrame([
        {"label": 97, "name": "L_Cerebellum", "hemi": "L"},
        {"label": 98, "name": "R_Cerebellum", "hemi": "R"}
    ])
], ignore_index=True)

# --- Sauvegarde ---
nib.save(nib.Nifti1Image(combined, cort_img.affine, cort_img.header), out_image_path)
labels.to_csv(out_label_path, sep="\t", index=False)

print("✅ Atlas combiné cortex + cervelet :", out_image_path)
print("✅ Labels TSV :", out_label_path)
print("Labels uniques :", np.unique(combined)[:15], "... max =", int(combined.max()))
print("Nombre total de labels :", len(labels))