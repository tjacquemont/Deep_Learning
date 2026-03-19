#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12:36:09 2025

@author: thomas
"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression  # Pour la séparation des données
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import statsmodels.api as sm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Vérification de la disponibilité de CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PYTORCH : Utilisation de l'appareil : {device}")


##########################################################################################################################
# Function_definition
##########################################################################################################################

def regression_multivariée(cohorte_data, selected_variables, target_variable, output_dir):
    """
    Effectue une régression multivariée et enregistre le résumé dans un fichier texte.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        variables_indépendantes (list): Liste des variables indépendantes (prédicteurs).
        variable_dépendante (str): Nom de la variable dépendante (cible).
        output_dir (str): Le dossier où les fichiers seront enregistrés.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = sm.add_constant(cohorte_data[selected_variables])
    y = cohorte_data[target_variable]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    output_file = os.path.join(output_dir, f"regression_of_{target_variable}_with_{'_'.join(selected_variables)}.txt")
    with open(output_file, 'w') as f:
        f.write(str(summary))

def patient_filtration(cohorte_data, dependante_variable, independante_variables_to_conserve, working_directory):

    variables_to_conserve = independante_variables_to_conserve + dependante_variable 
    variables_to_conserve += ['numero']

    not_missable_variables_cohorte = cohorte_data[variables_to_conserve].dropna()
    nb_patient_selected = not_missable_variables_cohorte.shape[0]

    print(f"Number of selected patients : {nb_patient_selected}")

    filtered_cohorte = cohorte_data.loc[cohorte_data['numero'].isin(not_missable_variables_cohorte['numero'])]

    filtered_cohorte.to_csv(working_directory + '/' + 'Filtred_cohorte.csv', sep=';', index=False)

    return filtered_cohorte
    

def generer_tableau_recapitulatif(df, list_variables, working_directory):
    """
    Génère un tableau récapitulatif des variables spécifiées.
    """

    recap = []
    for var in list_variables:
        if var in df.columns: # Verification de l'existance de la colonne.
            infos = {}
            infos['Variable'] = var
            infos['N_manquant'] = df[var].isnull().sum()
            infos['N_present'] = df[var].notnull().sum()

            if pd.api.types.is_numeric_dtype(df[var]):
                infos['Moyenne +/- Ecart-type'] = f"{df[var].mean():.2f} +/- {df[var].std():.2f}"
            else:
                try:
                    counts = df[var].value_counts(normalize=True) * 100
                    for value, percentage in counts.items():
                        infos[f'Pourcentage_{value}'] = f"{percentage:.2f}%"
                except TypeError:
                     infos['Modalités non calculables'] = 'Variable non numérique non catégorisable'
            recap.append(infos)
        else :
            print(f"Attention: La colonne '{var}' n'existe pas dans le DataFrame.")
    pd.DataFrame(recap).to_csv(working_directory + '/Cohorte_Demographics_Summary.csv', sep=';', index=False)


##########################################################################################################################
# Machine learning Models Specification
##########################################################################################################################
def split_data(X, y, test_size):
    """
    Split the dataset in test set and validation test set in a balanced way
    Parameters
    ----------
    X : TYPE (DataFrame)
        DESCRIPTION : Dataset
    y : TYPE (DataFrame)
          DESCRIPTION : Target set
    test_size : TUPE (Float)
        DESCRIPTION : Test set size

    Returns
    -------
    (X_train_balanced, Y_train_balanced) : Balanced train dataset
    (X_test_balanced, Y_test_balanced) : Balanced test dataset
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2301)

    for train_index, test_index in splitter.split(X, y):
        X_train_balanced, X_test_balanced = X.iloc[train_index], X.iloc[test_index]
        y_train_balanced, y_test_balanced = y.iloc[train_index], y.iloc[test_index]
    
    return X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_proba, title, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importance, feature_names, title, save_path):
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(feature_names)), importance[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def logistic_regression_classifier(X_train, y_train, X_test, y_test, working_directory, machine_learning_features):
    """
    Parameters
    ----------
    X_train : TYPE (DataFrame)
        DESCRIPTION : Train datas
    y_train : TYPE (DataFrame)
        DESCRIPTION : Train labels
    X_test : TYPE (DataFrame)
        DESCRIPTION : Test datas
    y_test: TYPE (DataFrame)
        DESCRIPTION : Test labels
    working_directory : TYPE (str)
        DESCRIPTION : directory where write the results

    Returns
    -------
    None.

    """

    # Creating the output directory
    logistic_regression_dir = os.path.join(working_directory, 'logistic_regression_classifier')
    os.makedirs(logistic_regression_dir, exist_ok=True)

    # Training the Logit regressions
    print("Training of the logits regression...")
    X_train_sm = sm.add_constant(X_train)  # Adding the constant for intercept
    model_logistic = sm.Logit(y_train, X_train_sm).fit()

    # Prediction on test data
    X_test_sm = sm.add_constant(X_test)
    y_pred_logistic_proba = model_logistic.predict(X_test_sm)
    
    best_threshold = 0.5 # Seuil par défaut
    max_accuracy = 0.0

    # Itérer sur une plage de seuils .01 à 0.99 par pas de 0.01
    for threshold in np.arange(0.01, 1.0, 0.01): 
        y_pred_current = (y_pred_logistic_proba > threshold).astype(int)
        current_accuracy = accuracy_score(y_test, y_pred_current)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_threshold = threshold
    
    y_pred_logistic = (y_pred_logistic_proba > best_threshold).astype(int)

    # Confusion matrix
    cm_logistic = confusion_matrix(y_test, y_pred_logistic)
    plot_confusion_matrix(cm_logistic, ['0', '1'], 'Confusion Matrix (Logistic Regression)', os.path.join(logistic_regression_dir, 'confusion_matrix.png'))

    # Importance des features (coefficients)
    importance_logistic = model_logistic.params[1:] # Exclure l'intercept
    plot_feature_importance(np.abs(importance_logistic), machine_learning_features, 'Feature Importance (Logistic Regression)', os.path.join(logistic_regression_dir, 'feature_importance.png'))

    # AUC
    auc_logistic = roc_auc_score(y_test, y_pred_logistic_proba)
    with open(os.path.join(logistic_regression_dir, 'auc.txt'), 'w') as f:
        f.write(str(auc_logistic))
    
    # Accuracy ans threshold recording
    with open(os.path.join(logistic_regression_dir, 'accuracy.txt'), 'w') as f:
        f.write('Accuracy : ' + str(max_accuracy) + ' (threshold : ' + str(best_threshold) + ')')
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_logistic_proba, 'ROC Curve (Logistic Regression)', os.path.join(logistic_regression_dir, 'roc_curve.png'))

    print(f"Resuls of Logit Regression Classification save in : {logistic_regression_dir}")

def support_vector_machine(X_train, y_train, X_test, y_test, working_directory, machine_learning_features, kernel='linear'):
    """
    Parameters
    ----------
    X_train : TYPE (DataFrame)
        DESCRIPTION : Train datas
    y_train : TYPE (DataFrame)
        DESCRIPTION : Train labels
    X_test : TYPE (DataFrame)
        DESCRIPTION : Test datas
    y_test: TYPE (DataFrame)
        DESCRIPTION : Test labels
    working_directory : TYPE (str)
        DESCRIPTION : directory where write the results

    Returns
    -------
    None.

    """
    # Creating the output directory
    svm_dir = os.path.join(working_directory, 'support_vector_machine')
    os.makedirs(svm_dir, exist_ok=True)

    # Fitting SVM
    print("\nTraining of the Support Vector Machine...")
    model_svm = SVC(probability=True, kernel=kernel, random_state=42)
    model_svm.fit(X_train, y_train.values.ravel())

    # Prediction on test data
    y_pred_svm_proba = model_svm.predict_proba(X_test)[:, 1]
    y_pred_svm = model_svm.predict(X_test)

    # Confusion matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plot_confusion_matrix(cm_svm, ['0', '1'], 'Confusion Matrix (SVM)', os.path.join(svm_dir, 'confusion_matrix.png'))
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_svm_proba, 'ROC Curve (SVM)', os.path.join(svm_dir, 'roc_curve.png'))
    
    # Importance des features (coefficients) - Seulement pour noyau linéaire
    if model_svm.kernel == 'linear':
        importance_svm = np.abs(model_svm.coef_.flatten())
        plot_feature_importance(importance_svm, machine_learning_features, 'Feature Importance (SVM - Linear Kernel)', os.path.join(svm_dir, 'feature_importance.png'))
    else:
        print("L'importance des features n'est pas directement interprétable avec un noyau non linéaire.")

    # AUC
    auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
    with open(os.path.join(svm_dir, 'auc.txt'), 'w') as f:
        f.write(str(auc_svm))

    # Accuracy
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    with open(os.path.join(svm_dir, 'accuracy.txt'), 'w') as f:
        f.write(str(accuracy_svm))

    print(f"Results of SVM save in : {svm_dir}")

def random_forest(X_train, y_train, X_test, y_test, working_directory, machine_learning_features, n_estimators=100):
    """
    Parameters
    ----------
    X_train : TYPE (DataFrame)o
        DESCRIPTION : Train datas
    y_train : TYPE (DataFrame)
        DESCRIPTION : Train labels
    X_test : TYPE (DataFrame)
        DESCRIPTION : Test datas
    y_test: TYPE (DataFrame)
        DESCRIPTION : Test labels
    working_directory : TYPE (str)
        DESCRIPTION : directory where write the results
    n_estimators : TYPE (int)
        description : NUMBER OF TREE TO BUILD (DEFAULT IS 100)

    Returns
    -------
    None.

    """
    # Creating the output directory
    random_forest_dir = os.path.join(working_directory, 'random_forest')
    os.makedirs(random_forest_dir, exist_ok=True)

    print("\nTraining the Random Forest...")
    model_rf = RandomForestClassifier(random_state=42, n_estimators=n_estimators)
    model_rf.fit(X_train, y_train)

    # Prediction on test data
    y_pred_rf_proba = model_rf.predict_proba(X_test)[:, 1]
    y_pred_rf = model_rf.predict(X_test)

    # Matrice de confusion
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_confusion_matrix(cm_rf, ['0', '1'], 'Confusion Matrix (Random Forest)', os.path.join(random_forest_dir, 'confusion_matrix.png'))
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_rf_proba, 'ROC Curve (Random Forest)', os.path.join(random_forest_dir, 'roc_curve.png'))
    
    # Importance des features
    importance_rf = model_rf.feature_importances_
    plot_feature_importance(importance_rf, machine_learning_features, 'Feature Importance (Random Forest)', os.path.join(random_forest_dir, 'feature_importance.png'))

    # AUC
    auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
    with open(os.path.join(random_forest_dir, 'auc.txt'), 'w') as f:
        f.write(str(auc_rf))
    
    # Accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    with open(os.path.join(random_forest_dir, 'accuracy.txt'), 'w') as f:
        f.write(str(accuracy_rf))


    print(f"Results for random forest save in : {random_forest_dir}")

def deeplearning_sequential_dense_network(X_train, y_train, X_test, y_test, working_directory, batch_size=32, num_epochs=50, learning_rate=0.001):
    
    # Creating the output directory
    deep_learning_dir = os.path.join(working_directory, 'deep_learning') 
    os.makedirs(deep_learning_dir, exist_ok=True)
    
    # Features standardisations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Data conversion to PyTorch Tensors 
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    # DataLoaders creation to facilitate the batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out

    input_dim = X_train_scaled.shape[1]
    model_pt = SimpleNN(input_dim).to(device)
    
    # Listes to save the history of metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Definition of optimizer and loss function
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_pt.parameters(), lr=learning_rate)

    # Modele Training
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        model_pt.train()
        running_loss_train = 0.0
        correct_predictions_train = 0
        total_train_samples = 0
    
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model_pt(inputs)
            loss_train = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
    
            running_loss_train += loss_train.item() * inputs.size(0)
    
            # Calcul de la précision sur le train
            probabilities_train = outputs.detach().cpu().numpy()
            predicted_train = (probabilities_train > 0.5).astype(int)
            true_labels_train = labels.cpu().numpy().astype(int)
            correct_predictions_train += np.sum(predicted_train == true_labels_train)
            total_train_samples += labels.size(0)
    
        epoch_loss_train = running_loss_train / total_train_samples
        epoch_accuracy_train = correct_predictions_train / total_train_samples
        train_losses.append(epoch_loss_train)
        train_accuracies.append(epoch_accuracy_train)
    
        # Évaluation sur l'ensemble de validation
        model_pt.eval()
        running_loss_val = 0.0
        correct_predictions_val = 0
        total_val_samples = 0
    
        with torch.no_grad():
            for inputs_val, labels_val in test_loader:
                outputs_val = model_pt(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                running_loss_val += loss_val.item() * inputs_val.size(0)
    
                probabilities_val = outputs_val.cpu().numpy()
                predicted_val = (probabilities_val > 0.5).astype(int)
                true_labels_val = labels_val.cpu().numpy().astype(int)
                correct_predictions_val += np.sum(predicted_val == true_labels_val)
                total_val_samples += labels_val.size(0)
    
        epoch_loss_val = running_loss_val / total_val_samples
        epoch_accuracy_val = correct_predictions_val / total_val_samples
        val_losses.append(epoch_loss_val)
        val_accuracies.append(epoch_accuracy_val)
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train:.4f}, Train Accuracy: {epoch_accuracy_train:.4f}, Val Loss: {epoch_loss_val:.4f}, Val Accuracy: {epoch_accuracy_val:.4f}')
    
    # Sauvegarde de l'évolution des métriques dans un fichier CSV
    metrics_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    metrics_file_path = os.path.join(deep_learning_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nÉvolution des métriques sauvegardée dans : {metrics_file_path}")
    
    # Création et sauvegarde du graphique
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Évolution de la Perte')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Évolution de la Précision')
    plt.legend()
    
    graph_file_path = os.path.join(deep_learning_dir, 'training_metrics.png')
    plt.savefig(graph_file_path)
    plt.close()
    print(f"\nGraphique de l'évolution des métriques sauvegardé dans : {graph_file_path}")

    # Évaluation finale sur l'ensemble de validation pour sauvegarder la matrice de confusion et l'AUC
    model_pt.eval()
    all_preds_proba_final = []
    all_labels_final = []
    with torch.no_grad():
        for inputs_val, labels_val in test_loader:
            outputs_val = model_pt(inputs_val)
            probabilities_val = outputs_val.cpu().numpy().flatten()
            true_labels_val = labels_val.cpu().numpy().flatten().astype(int)
            all_preds_proba_final.extend(probabilities_val)
            all_labels_final.extend(true_labels_val)
    
    y_pred_dl_proba = np.array(all_preds_proba_final)
    y_pred_dl = (y_pred_dl_proba > 0.5).astype(int)
    y_val_np = np.array(all_labels_final)


    # Matrice de confusion
    cm_dl = confusion_matrix(y_val_np, y_pred_dl)
    plot_confusion_matrix(cm_dl, ['0', '1'], 'Confusion Matrix (Deep Learning)', os.path.join(deep_learning_dir, 'confusion_matrix.png'))
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_dl_proba, 'ROC Curve (DeepLearning)', os.path.join(deep_learning_dir, 'roc_curve.png'))

    # AUC 
    auc_dl = roc_auc_score(y_val_np, y_pred_dl_proba)
    with open(os.path.join(deep_learning_dir, 'auc.txt'), 'w') as f:
        f.write(str(auc_dl))
    
    # Accuracy
    accuracy_dl = accuracy_score(y_val_np, y_pred_dl)
    with open(os.path.join(deep_learning_dir, 'accuracy.txt'), 'w') as f:
        f.write(str(accuracy_dl))

    print(f"Résultats du Deep Learning (PyTorch) sauvegardés dans : {deep_learning_dir}")

    print(f"Shape of X before split: {X.shape}")
    print(f"Shape of y before split: {y.shape}")
    print(f"Data types of X:\n{X.dtypes}")
    print(f"Data types of y:\n{y.dtypes}")
##########################################################################################################################
study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'
database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'
working_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Standard_Machine_Learning_Methods'


list_demographics_variables =  ['Age', 'Sexe',
       'Taille_cm', 'Lateralite', 'Raw_IRM_Diffusion','Raw_IRM_FLAIR',
       'IRM_Diffusion_Norm', 'Lesion_Mask_Norm',
       'Type_I_ischemic_H_hemorrhagic', 'COTE_1G_2D', 'TIV','TM', 'Delai_AVC_NIHSS_J3', 'NIHSS_J3', 'NIHSS_MS_J3', 'NIHSS_neg_J3', 'NIHSS_apha_J3',
       'SAFE_J3', 'SAFE_J7', 'NIHSS_J7', 'NIHSS_MS_J7', 'NIHSS_neg_J7', 'NIHSS_aphaJ7', 
       'MOCA', 'Aphasie', 'Volume', 'Overlap_CST_cross', 'rMT_IPSI', 'rMT_CONTRO',
       'PEM_plus1_ipsi', 'PEM_plus1_contro', 'PEMmax_ipsi', 'PEMmax_contro',
       'FM_J7', 'Leucocytes', 'Lymphocytes','RNL', 'Neutrophiles', 'Plaquettes', 
       'Delai_follow_up_en_mois', 'deces_follow_up', 'FM_imput_M3', 'Categorie_M3_FM_binaire',
       'FM_M3_J7_Max_Recovery_ratio_binaire','ARAT_imput_M3', 'Categorie_M3_ARAT_binaire',
       'delai_AVC_TMS', 'delai_AVC_IRM', 'delai_AVC_prelevement']

liste_dependante_variable = [['Categorie_M3_FM_binaire'], ['FM_M3_J7_Max_Recovery_ratio_binaire'], ['Categorie_M3_ARAT_binaire']]

##########################################################################################################################
# Machine learning Evaluation
test_size = 0.2
svc_kernel = 'linear'
rf_nb_estimators = 1000
dl_batch_size=32
dl_num_epochs = 50
dl_learning_rate=0.001
list_cohorte_to_analyse = ['Whole', 'SAFE', 'PEM']
list_machine_learning_features = [['Age', 'NIHSS_J3', 'SAFE_J3', 'Volume', 'Overlap_CST_cross'], ['Age', 'NIHSS_J3', 'SAFE_J3', 'Volume', 'Overlap_CST_cross', 'PEM_plus1_ipsi'], ['Age', 'SAFE_J7', 'NIHSS_J7','Volume', 'Overlap_CST_cross', 'PEM_plus1_ipsi']]
remove_subpopulation_variable = False  # en pratique ne s'applioque que pour le SAFE_J3
split_error = []

for machine_learning_features in list_machine_learning_features:
    for cohorte_to_analyse in list_cohorte_to_analyse:
        for dependante_variable in liste_dependante_variable:
                ########################################################################
                # Récupération des données
                print(f"Loading data in the {database_path} CSV.")
                print(f"Cohorte to analyse : {cohorte_to_analyse}")
                cohorte_data = pd.read_csv(database_path, sep=';')
                if cohorte_to_analyse == 'Whole':
                            working_directory_cohorte = working_directory + f'/Whole_cohorte/Classification/{dependante_variable[0]}/{'_'.join(machine_learning_features)}'
                            cohorte_data_to_analyse = cohorte_data # Depending on the analysed cohorte 
                            liste_variables = machine_learning_features + dependante_variable
                            cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                            cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                            if not os.path.exists(working_directory_cohorte):
                                os.makedirs(working_directory_cohorte)
                            cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/Whole_cohorte.csv', sep=';', index=False)
                            generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                            regression_multivariée(cohorte_data_ml_without_NA, machine_learning_features, dependante_variable, working_directory_cohorte)
                            X = cohorte_data_ml_without_NA[machine_learning_features]
                            if 'Sexe' in machine_learning_features:
                                X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                            y =  cohorte_data_ml_without_NA[dependante_variable]
                            try :
                                X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                                print('Split worked !')
                                logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, machine_learning_features)
                                support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, machine_learning_features, kernel=svc_kernel)
                                random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, machine_learning_features, n_estimators=rf_nb_estimators)
                                deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                            except Exception as e:
                                print(f"Error append during split_data: {e}")
                                split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}"]
                                #continue
                elif cohorte_to_analyse == 'SAFE':
                        cohorte_data_SAFE_J3_cleaned = cohorte_data.dropna(subset=['SAFE_J3']).copy()
                        independante_variables_to_conserve_safe = machine_learning_features.copy()
                        if 'SAFE_J3' in independante_variables_to_conserve_safe and remove_subpopulation_variable:
                            independante_variables_to_conserve_safe.remove('SAFE_J3')
                        working_directory_safe = working_directory + f'/SAFE_J3/Classification/{dependante_variable[0]}/{'_'.join(independante_variables_to_conserve_safe)}'
                        # Groups constitution : [0-4] | [5-10] 
                        cohorte_data_SAFE_J3_cleaned_plus_5 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] >= 5].copy()
                        cohorte_data_SAFE_J3_cleaned_moins_5 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] < 5].copy()
                        ###################################################################
                        # Analysing the SAFE J3 >= 5 subcohort
                        print("Analysing the SAFE J3 >= 5 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/More_than_5'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_plus_5
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_more_than_5.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '0' when SAFE J3 > 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_more_than_5_subcohorte"]
                            #continue
                        ###################################################################
                        # Analysing the SAFE J3 < 5 subcohort
                        print("Analysing the SAFE J3 < 5 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/Less_than_5'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_moins_5
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_less_than_5.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '1' when SAFE J3 < 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_5_subcohorte"]
                            #continue
                        # Groups constitution : [0-5] | [6-10] 
                        cohorte_data_SAFE_J3_cleaned_plus_6 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] >= 6].copy()
                        cohorte_data_SAFE_J3_cleaned_moins_6 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] < 6].copy()
                        ###################################################################
                        # Analysing the SAFE J3 >= 6 subcohort
                        print("Analysing the SAFE J3 > 6 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/More_than_6'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_plus_6
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_more_than_6.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '0' when SAFE J3 > 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_more_than_6_subcohorte"]
                            #continue
                        ###################################################################
                        # Analysing the SAFE J3 < 6 subcohort
                        print("Analysing the SAFE J3 < 6 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/Less_than_6'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_moins_6
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_less_than_6.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '1' when SAFE J3 < 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_6_subcohorte"]
                            #continue
                        # Groups constitution : [0-7] | [8-10] 
                        cohorte_data_SAFE_J3_cleaned_plus_8 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] >= 8].copy()
                        cohorte_data_SAFE_J3_cleaned_moins_8 = cohorte_data_SAFE_J3_cleaned[cohorte_data_SAFE_J3_cleaned['SAFE_J3'] < 8].copy()
                        ###################################################################
                        # Analysing the SAFE J3 >= 8 subcohort
                        print("Analysing the SAFE J3 > 8 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/More_than_8'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_plus_8
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_more_than_8.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '0' when SAFE J3 > 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_more_than_8_subcohorte"]
                            #continue
                        ###################################################################
                        # Analysing the SAFE J3 < 6 subcohort
                        print("Analysing the SAFE J3 < 8 sub-cohort")
                        working_directory_cohorte = working_directory_safe + '/Less_than_8'
                        cohorte_data_to_analyse = cohorte_data_SAFE_J3_cleaned_moins_8
                        liste_variables = independante_variables_to_conserve_safe + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/SAFE_J3_less_than_8.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_safe, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                        if 'Sexe' in independante_variables_to_conserve_safe:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '1' when SAFE J3 < 5 
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_safe, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_8_subcohorte"]
                            #continue
                elif cohorte_to_analyse == 'PEM':
                        working_directory_pem = working_directory + f'/PEM/Classification/{dependante_variable[0]}/{'_'.join(machine_learning_features)}'
                        cohorte_data_PEM_cleaned = cohorte_data.dropna(subset=['PEM_plus1_ipsi']).copy()
                        independante_variables_to_conserve_pem = machine_learning_features.copy()
                        if 'PEM_plus1_ipsi' in independante_variables_to_conserve_pem:
                            independante_variables_to_conserve_pem.remove('PEM_plus1_ipsi')
                        cohorte_data_PEM_pos = cohorte_data_PEM_cleaned[cohorte_data_PEM_cleaned['PEM_plus1_ipsi']==1].copy()
                        cohorte_data_PEM_neg = cohorte_data_PEM_cleaned[cohorte_data_PEM_cleaned['PEM_plus1_ipsi']==0].copy()
                        ############################################################
                        # Analysing the PEM positive subcohort
                        print("Analysing the PEM positive subcohort")
                        working_directory_cohorte = working_directory_pem + '/Present'
                        cohorte_data_to_analyse = cohorte_data_PEM_pos
                        independante_variables_to_conserve_PEM_pos = independante_variables_to_conserve_pem + ['PEMmax_ipsi']
                        liste_variables = independante_variables_to_conserve_PEM_pos + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/PEM_Present.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_PEM_pos, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_PEM_pos]
                        if 'Sexe' in independante_variables_to_conserve_PEM_pos:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '0' when PEM is +
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_PEM_pos)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_PEM_pos, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_PEM_pos, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_pem_positive"]
                            continue
                        ###################################################################
                        # Analysing the PEM negative subcohort
                        print("Analysing the PEM negative subcohort")
                        working_directory_cohorte = working_directory_pem + '/Absent'
                        cohorte_data_to_analyse = cohorte_data_PEM_neg
                        liste_variables = independante_variables_to_conserve_pem + dependante_variable
                        cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                        cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                        if not os.path.exists(working_directory_cohorte):
                            os.makedirs(working_directory_cohorte)
                        cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/PEM_Absent.csv', sep=';', index=False)
                        generer_tableau_recapitulatif(cohorte_data.iloc[cohorte_data_ml_without_NA.index], list_demographics_variables, working_directory_cohorte)
                        regression_multivariée(cohorte_data_ml_without_NA, independante_variables_to_conserve_pem, dependante_variable, working_directory_cohorte)
                        X = cohorte_data_ml_without_NA[independante_variables_to_conserve_pem]
                        if 'Sexe' in independante_variables_to_conserve_pem:
                            X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0}) 
                        y =  cohorte_data_ml_without_NA[dependante_variable]
                        try:
                            X_train_balanced,y_train_balanced, X_test_balanced, y_test_balanced = split_data(X, y, test_size=test_size)
                            print('Split worked !')
                            # Logistic regression is removed because not enought patient are classed as '1' when PEM is -
                            logistic_regression_classifier(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_pem)
                            support_vector_machine(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_pem, kernel=svc_kernel)
                            random_forest(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, independante_variables_to_conserve_pem, n_estimators=rf_nb_estimators)
                            deeplearning_sequential_dense_network(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, working_directory_cohorte, batch_size=dl_batch_size, num_epochs=dl_num_epochs, learning_rate=dl_learning_rate)
                        except Exception as e:
                            print(f"Error append during split_data: {e}")
                            split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_6_subcohorte"]
                            #continue
                else:
                    print("ERROR IN COHORTE DEFINITION")