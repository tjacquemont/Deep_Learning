#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Grad-CAM x Atlas Overlap & Stats (MNI space)

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import math

# -------------------- Utility functions --------------------

def load_nii(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def qvalues_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty_like(p)
    q[order] = (p[order] * n / ranks)
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return np.clip(q, 0, 1)

def pvalues_bonferroni(pvals):
    """
    Bonferroni correction for multiple comparisons.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    p_corr = np.minimum(p * n, 1.0)
    return p_corr

def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    delta = (2*u) / (nx*ny) - 1
    return float(delta)

def quantile_threshold(arr, q, mask=None):
    if mask is not None:
        vals = arr[mask > 0]
    else:
        vals = arr.ravel()
    thr = float(np.quantile(vals, q))
    return thr, (arr >= thr).astype(np.uint8)

def flip_lr_if_needed(arr):
    flip_array = np.flip(arr, axis=0)
    return flip_array

def resample_like(src_img, tgt_img, order=0):
    try:
        from nilearn.image import resample_to_img
        res = resample_to_img(src_img, tgt_img, interpolation='nearest' if order == 0 else 'linear')
        return res.get_fdata(dtype=np.float32)
    except Exception:
        raise RuntimeError("Please install nilearn for resampling: pip install nilearn")

def load_labels_tsv(path: Path):
    df = pd.read_csv(path, sep="\t")
    if "label" not in df.columns or "name" not in df.columns:
        raise ValueError("labels TSV must contain at least 'label' and 'name' columns")
    if "hemi" not in df.columns:
        df["hemi"] = "B"
    return df

def roi_iter_from_label_atlas(atlas_arr, labels_df):
    for _, row in labels_df.iterrows():
        lab = int(row["label"])
        mask = (atlas_arr == lab).astype(np.uint8)
        if mask.sum() == 0:
            continue
        hemi = row["hemi"] if "hemi" in row and isinstance(row["hemi"], str) else "B"
        yield lab, row["name"], hemi, mask

def roi_iter_from_prob_atlas(atlas_arr, labels_df=None):
    if atlas_arr.ndim == 4:
        n_roi = atlas_arr.shape[-1]
        for i in range(n_roi):
            w = atlas_arr[..., i].astype(np.float32)
            if np.sum(w) <= 0:
                continue
            name = f"ROI_{i}" if labels_df is None else labels_df.iloc[i]["name"]
            hemi = "B" if labels_df is None or "hemi" not in labels_df.columns else labels_df.iloc[i].get("hemi", "B")
            yield i, name, hemi, w
    else:
        yield 1, "ROI_all", "B", atlas_arr.astype(np.float32)

def overlap_metrics(grad, roi, mask=None, thr_quant=0.60, metrics=("mean","coverage","dice"), atlas_is_prob=False):
    if mask is not None:
        grad = np.where(mask > 0, grad, 0)

    thr, grad_bin = quantile_threshold(grad, thr_quant, mask=mask)

    if atlas_is_prob:
        roi_w = roi.astype(np.float32)
        roi_bin = (roi_w > 0).astype(np.uint8)
        num = np.sum(grad * roi_w)
        den = np.sum(roi_w) + 1e-8
        mean_in = float(num / den) if np.sum(roi_w)>0 else float("nan")
    else:
        roi_bin = roi.astype(np.uint8)
        mean_in = float(np.mean(grad[roi_bin > 0])) if np.sum(roi_bin) > 0 else float("nan")

    inter = int(np.sum((grad_bin > 0) & (roi_bin > 0)))
    dice = (2 * inter) / (np.sum(grad_bin) + np.sum(roi_bin)) if np.sum(roi_bin)>0 else float("nan")
    coverage = inter / np.sum(grad_bin) if np.sum(grad_bin)>0 else float("nan")

    out = {}
    if "mean" in metrics:
        out["mean_in_roi"] = mean_in
    if "coverage" in metrics:
        out["coverage_roi"] = float(coverage)
    if "dice" in metrics:
        out["dice_roi"] = float(dice)
    out["threshold"] = float(thr)
    out["n_gradcam_vox"] = int(np.sum(grad_bin))
    out["n_roi_vox"] = int(np.sum(roi_bin))
    return out


def plot_roi_stats(stats_csv, overlap_csv, out_dir, p_value_col='p_fdr', alpha=0.05, max_cols=5, flip_right_to_left=True):
    """
    Visualise les stats ROI par un barplot pour chaque région.
    - stats_csv : chemin vers roi_stats_left_vs_right.csv
    - overlap_csv : chemin vers roi_overlap_per_patient.csv
    - out_dir : dossier de sortie pour les figures
    """

    stats_df = pd.read_csv(stats_csv)
    overlap_df = pd.read_csv(overlap_csv)

    # On limite aux ROI pour lesquels il y a bien des stats
    rois = stats_df["roi_name"].unique()
    n_rois = len(rois)
    ncols = min(max_cols, n_rois)
    nrows = math.ceil(n_rois / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    axes = axes.flatten()

    for i, roi_name in enumerate(rois):
        ax = axes[i]
        sub = stats_df[stats_df["roi_name"] == roi_name]
        metric = sub.iloc[0]["metric"] if "metric" in sub.columns else "mean"
        if flip_right_to_left:
            fig_title=roi_name.replace('L_','Ipsi_').replace('R_', 'Contra_')
        else:
            fig_title=roi_name

        # Récupère les valeurs par patient pour calculer les barres d’erreur
        dsub = overlap_df[overlap_df["roi_name"] == roi_name]
        left_vals = dsub.loc[dsub["group"] == "left", f"{metric}_in_roi" if metric == "mean" else f"{metric}_roi"].dropna()
        right_vals = dsub.loc[dsub["group"] == "right", f"{metric}_in_roi" if metric == "mean" else f"{metric}_roi"].dropna()

        mean_left, std_left = left_vals.mean(), left_vals.std()
        mean_right, std_right = right_vals.mean(), right_vals.std()

        ax.bar(["Left", "Right"], [mean_left, mean_right],
               yerr=[std_left, std_right], capsize=5,
               color=["#4C72B0", "#DD8452"], alpha=0.8)
        ax.set_title(fig_title, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)

        # Vérifie la significativité
        pval = sub[p_value_col].values[0]
        if pval < alpha:
            sig_marker = "*" if pval >= 0.01 else "**" if pval >= 0.001 else "***"
            ymax = max(mean_left + std_left, mean_right + std_right)
            ax.text(0.5, ymax*0.95, sig_marker, ha="center", va="bottom", color="red", fontsize=14)

    # Supprime les sous-graphiques vides
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.suptitle("ROI-wise GradCAM overlap (mean ± SD)", y=1.02, fontsize=14, fontweight="bold")

    out_path = Path(out_dir) / "roi_stats_barcharts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out_path}")
    

def plot_metric_vs_motor(overlap_csv, motor_score_name, out_dir,
                         clinical_csv='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/PREP_Database/Base_PREP_Thomas.csv',
                         metric="mean_in_roi", alpha=0.05, max_cols=5):
    """
    Analyse la corrélation entre la métrique GradCAM et le score moteur pour chaque ROI,
    séparément pour les AVC gauches et droits (colonne 'group' du fichier overlap_per_patient).

    Parameters
    ----------
    overlap_csv : str | Path
        CSV contenant les moyennes par ROI et patient (roi_overlap_per_patient.csv)
    clinical_csv : str | Path
        CSV contenant les scores moteurs par patient (avec 'patient_id' et le score)
    motor_score_name : str
        Nom du score moteur à corréler (ex: 'FM_imput_M3')
    out_dir : str | Path
        Dossier de sortie
    metric : str
        Nom de la métrique à utiliser ('mean_in_roi', 'coverage_roi', etc.)
    alpha : float
        Seuil de significativité (Bonferroni)
    max_cols : int
        Nombre max de colonnes à afficher
    """

    import seaborn as sns
    from scipy.stats import spearmanr

    overlap_df = pd.read_csv(overlap_csv)
    clinical_df = pd.read_csv(clinical_csv)

    if "group" not in overlap_df.columns:
        raise ValueError("Le fichier overlap_per_patient doit contenir une colonne 'group' (left/right).")

    # Merge avec le CSV clinique pour récupérer les scores moteurs
    df = overlap_df.merge(clinical_df[["patient_id", motor_score_name]], on="patient_id", how="inner")
    if df.empty:
        print(f"[WARN] Aucun patient commun entre {overlap_csv} et {clinical_csv}")
        return

    # Itération sur les deux groupes (AVC gauche et droit)
    for side in ["left", "right"]:
        df_side = df[df["group"] == side]
        if df_side.empty:
            print(f"[INFO] Aucun patient pour le groupe {side}.")
            continue

        print(f"\n[INFO] Corrélations pour les AVC {side.upper()} ({len(df_side['patient_id'].unique())} patients)")

        results = []
        rois = df_side["roi_name"].unique()
        for roi in rois:
            sub = df_side[df_side["roi_name"] == roi]
            if len(sub) < 5:
                continue

            x = sub[motor_score_name].values
            y = sub[metric].values

            if np.all(np.isnan(y)) or np.all(np.isnan(x)):
                continue

            rho, pval = spearmanr(x, y, nan_policy='omit')
            results.append({"roi_name": roi, "rho": rho, "p_value": pval})

        corr_df = pd.DataFrame(results)
        if corr_df.empty:
            print(f"[WARN] Pas de corrélation calculée pour les {side}.")
            continue

        # Correction Bonferroni
        corr_df["p_bonf"] = pvalues_bonferroni(corr_df["p_value"].values)
        corr_df["significant"] = corr_df["p_bonf"] < alpha

        # Sauvegarde CSV
        out_csv = Path(out_dir) / f"roi_correlation_{motor_score_name}_AVC-{side}.csv"
        save_csv(corr_df, out_csv)
        print(f"✅ Corrélations sauvegardées ({side}) : {out_csv}")

        # Sélection des ROI les plus significatives
        top_rois = corr_df.sort_values("p_bonf").head(max_plots)["roi_name"]
        n = len(top_rois)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
        axes = axes.flatten()

        for i, roi in enumerate(top_rois):
            ax = axes[i]
            sub = df_side[df_side["roi_name"] == roi]
            sns.regplot(
                x=motor_score_name, y=metric, data=sub,
                scatter_kws={"s": 40, "alpha": 0.7},
                line_kws={"color": "red", "lw": 1.5}, ax=ax
            )
            rho = corr_df.loc[corr_df["roi_name"] == roi, "rho"].values[0]
            p_bonf = corr_df.loc[corr_df["roi_name"] == roi, "p_bonf"].values[0]
            ax.set_title(f"{roi}\nρ={rho:.2f}, p={p_bonf:.3g}")
            ax.grid(alpha=0.3)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        out_fig = Path(out_dir) / f"roi_corr_scatter_{motor_score_name}_AVC-{side}.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Figure sauvegardée ({side}) : {out_fig}")



# -------------------- Main pipeline --------------------

def Gradcam_Atlas_Overlap(left_dir, right_dir, atlas, atlas_labels, atlas_is_prob, out_dir,
                          mask='/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/MNI152_T1_1mm_brain_mask.nii.gz', 
                          flip_right_to_left=True, metric="mean", threshold=0.60, vizualisation=True):
    import numpy as np
    import pandas as pd

    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    left_files = sorted([p for p in left_dir.glob("*.nii*")])
    right_files = sorted([p for p in right_dir.glob("*.nii*")])
    if len(left_files) == 0 or len(right_files) == 0:
        raise RuntimeError("No NIfTI found in left_dir/right_dir")

    # Load one Grad-CAM to set target grid
    tgt_img, _ = load_nii(left_files[0])

    atlas_img, _ = load_nii(Path(atlas))
    atlas_arr = resample_like(atlas_img, tgt_img, order=0 if not atlas_is_prob else 1)

    labels_df = load_labels_tsv(Path(atlas_labels)) if atlas_labels else None
    if flip_right_to_left and labels_df is not None:
        print("[INFO] flip_right_to_left=True → Now : Left = Ipsilateral, Right = Contralateral")

    mask_arr = None
    if mask:
        _, mask_arr = load_nii(Path(mask))
        mask_arr = (mask_arr > 0).astype(np.uint8)

    # Prepare ROI iterator
    if atlas_is_prob:
        roi_iter = list(roi_iter_from_prob_atlas(atlas_arr, labels_df))
    else:
        if labels_df is None:
            raise ValueError("For label atlases, --atlas_labels is required")
        roi_iter = list(roi_iter_from_label_atlas(atlas_arr, labels_df))

    metrics = [m.strip() for m in metric.split(",") if m.strip()]
    rows = []

    def process_group(files: List[Path], group_name: str):
        group_rows = []
        for f in files:
            pid = f.stem
            img, grad = load_nii(f)
            if flip_right_to_left and group_name == "right":
                grad = flip_lr_if_needed(grad)

            # Resample grad to target if shape mismatch
            if grad.shape != tgt_img.shape:
                from nilearn.image import resample_to_img
                res = resample_to_img(img, tgt_img, interpolation='linear')
                grad = res.get_fdata(dtype=np.float32)

            for (lab, name, hemi, roi) in roi_iter:
                stats_row = overlap_metrics(
                    grad=grad,
                    roi=roi,
                    mask=mask_arr,
                    thr_quant=threshold,
                    metrics=metrics,
                    atlas_is_prob=atlas_is_prob
                )
                row = {
                    "numero": pid.replace('gradcam_patient_','').replace('.nii',''),
                    "group": group_name,
                    "roi_label": lab,
                    "roi_name": name,
                    "roi_hemi": hemi,
                }
                row.update(stats_row)
                group_rows.append(row)
        return group_rows

    rows.extend(process_group(left_files, "left"))
    rows.extend(process_group(right_files, "right"))

    df = pd.DataFrame(rows)
    overlap_csv = out_dir / "roi_overlap_per_patient.csv"
    save_csv(df, overlap_csv)

    # Stats per ROI: left vs right
    results = []
    unique_rois = {(lab, name, hemi) for (lab, name, hemi, _) in roi_iter}
    for (lab, name, hemi) in unique_rois:
        dsub = df[df["roi_label"] == lab]
        for m in metrics:
            col_map = {"mean": "mean_in_roi", "coverage": "coverage_roi", "dice": "dice_roi"}
            col = col_map[m]
            left_vals = dsub.loc[dsub["group"] == "left", col].dropna().values
            right_vals = dsub.loc[dsub["group"] == "right", col].dropna().values
            if len(left_vals) == 0 or len(right_vals) == 0:
                continue
            u, p = stats.mannwhitneyu(left_vals, right_vals, alternative="two-sided")
            eff = cliffs_delta(left_vals, right_vals)
            results.append({
                "roi_label": lab,
                "roi_name": name,
                "roi_hemi": hemi,
                "metric": m,
                "n_left": int(len(left_vals)),
                "n_right": int(len(right_vals)),
                "mean_left": float(np.mean(left_vals)),
                "mean_right": float(np.mean(right_vals)),
                "effect_cliffs_delta": float(eff),
                "p_value": float(p),
            })

    stats_df = pd.DataFrame(results)
    if not stats_df.empty:
        stats_csv = out_dir / "roi_stats_left_vs_right.csv"
        stats_df["p_fdr"] = qvalues_bh(stats_df["p_value"].values)
        stats_df["p_bonf"] = pvalues_bonferroni(stats_df["p_value"].values)
    save_csv(stats_df, stats_csv)

    print("Done. Wrote:")
    print(f" - {out_dir/'roi_overlap_per_patient.csv'}")
    print(f" - {out_dir/'roi_stats_left_vs_right.csv'}")
    
    if vizualisation:
        plot_roi_stats(stats_csv, overlap_csv, out_dir, flip_right_to_left=flip_right_to_left, p_value_col='p_fdr')
        

###############################################################################################
out_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics'
models_dir = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/Models'
list_motor_score = ['ARAT_imput_M3', 'FM_imput_M3', 'FM_M3_J7_Max_Recovery_ratio']
list_layer = ['conv3', 'conv4']

atlas_database = [
    {'name' : 'JHU', 'Is_Prob': True, 'label' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/JHU-ICBM-tracts-prob-1mm.nii.gz', 'label_legend' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/JHU_labels.tsv'},
#    {'name' : 'Destrieux', 'Is_Prob': False, 'label' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/aparc_a2009s_MNI.nii.gz', 'label_legend' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/Destrieux_labels.tsv'},
    {'name' : 'HarvardOxford_Lat_Cerebellum', 'Is_Prob': False, 'label' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/HarvardOxford_Lat_Cerebellum_combined-maxprob-thr25-1mm.nii.gz', 'label_legend' : '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Deep_Learning/GradCam_Statistics/1_Atlas/HarvardOxford_Lat_Cerebellum_combined.tsv'},
         ]

for motor_score in list_motor_score:
    for layer in list_layer:
        left_dir = os.path.join(models_dir, 'Left', motor_score, f'Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_{motor_score}_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi/Validation_Images/Patients_Grad-CAM_{layer}_to_MNI')
        right_dir = os.path.join(models_dir, 'Right', motor_score, f'Mixed_CNN_4Conv_with_AveragePool_Vectors_LayerNorm_with_CrossAttention_Mid_Dropout_and_4FC_{motor_score}_irm_IRM_Diffusion_Norm-ADC_Norm_vect_Age-SAFE_J7-FM_J7-NIHSS_J7-PEM_plus1_ipsi/Validation_Images/Patients_Grad-CAM_{layer}_to_MNI')
        for atlas_dict in atlas_database:
            atlas_name = atlas_dict['name']
            atlas_img = atlas_dict['label']
            atlas_label = atlas_dict['label_legend']
            atlas_is_prob = atlas_dict['Is_Prob']
            out_subdirectory = os.path.join(out_directory, motor_score, layer, atlas_name)
            os.makedirs(out_subdirectory, exist_ok=True)
            Gradcam_Atlas_Overlap(left_dir, right_dir, atlas_img, atlas_label, atlas_is_prob, out_subdirectory)