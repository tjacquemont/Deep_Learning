import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

study_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC'
database_path = study_directory + '/PREP_Database/Base_PREP_Thomas.csv'
working_directory = '/home/thomas.jacquemont/cenir_thomas.jacquemont/PREP_AVC/Results/Standard_Machine_Learning_Methods'

test_size = 0.2
ridge_alpha = 10
list_cohorte_to_analyse = ['Whole', 'SAFE', 'PEM']
liste_dependante_variable = [['FM_imput_M3'], ['FM_M3_J7_Recovery'], ['FM_M3_J7_Max_Recovery_ratio'], ['ARAT_imput_M3']]
list_machine_learning_features = [['Age', 'NIHSS_J3', 'SAFE_J3', 'Volume', 'Overlap_CST_cross'], ['Age', 'NIHSS_J3', 'SAFE_J3', 'Volume', 'Overlap_CST_cross', 'PEM_plus1_ipsi'], ['Age', 'SAFE_J7', 'NIHSS_J7','Volume', 'Overlap_CST_cross', 'PEM_plus1_ipsi']]
split_error = []

def plot_regression_with_train_test(X_train, y_train, X_test, y_test, model, independent_variable, dependent_variable_name, working_directory, cohorte_name):
    """
    Crée un scatter plot de la variable indépendante vs. la variable dépendante,
    différenciant les points d'entraînement et de test par couleur, et affiche
    la droite de régression linéaire apprise sur l'ensemble d'entraînement.

    Args:
        X_train (pd.DataFrame): Ensemble des features d'entraînement.
        y_train (pd.DataFrame): Variable dépendante d'entraînement.
        X_test (pd.DataFrame): Ensemble des features de test.
        y_test (pd.DataFrame): Variable dépendante de test.
        model (sklearn.linear_model): Modèle de régression linéaire entraîné.
        independent_variable (str): Nom de la variable indépendante à plotter.
        dependent_variable_name (str): Nom de la variable dépendante.
        working_directory (str): Répertoire de sauvegarde des figures.
        cohorte_name (str): Nom de la cohorte/sous-cohorte pour le titre.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot des données d'entraînement
    sns.scatterplot(x=X_train[independent_variable], y=y_train.squeeze(), color='blue', label='Train Data')

    # Scatter plot des données de test
    sns.scatterplot(x=X_test[independent_variable], y=y_test.squeeze(), color='red', label='Test Data')

    # Plot de la droite de régression
    if len(X_train[independent_variable]) > 0:
        x_range = pd.concat([X_train[independent_variable], X_test[independent_variable]]).sort_values()
        y_pred = model.predict(pd.DataFrame({independent_variable: x_range}))
        plt.plot(x_range, y_pred, color='green', linestyle='-', linewidth=2, label='Regression Line (Train)')

    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable_name)
    plt.title(f'Regression of {dependent_variable_name} on {independent_variable} - {cohorte_name}')
    plt.legend()
    plt.grid(True)

    filename = os.path.join(working_directory, f'regression_plot_{dependent_variable_name}_vs_{independent_variable}_{cohorte_name}.png')
    plt.savefig(filename)
    plt.close()
    
def linear_regression_model(X_train, y_train, X_test, y_test, working_directory, features, dependent_variable_name, cohorte_name):
    """Entraîne un modèle de régression linéaire et évalue le R2."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"R2 score on training set (Linear): {r2_train:.4f}")
    print(f"R2 score on test set (Linear): {r2_test:.4f}")

    # Sauvegarde des résultats (optionnel)
    results_path = os.path.join(working_directory, f'linear_regression_results_{"_".join(features)}.txt')
    with open(results_path, 'w') as f:
        f.write(f"R2 score on training set (Linear): {r2_train:.4f}\n")
        f.write(f"R2 score on test set (Linear): {r2_test:.4f}\n")
    
    # Sauvegarde des coefficients
    coefficients_path = os.path.join(working_directory, f'linear_regression_coefficients_{"_".join(features)}.txt')
    with open(coefficients_path, 'w') as f:
        f.write(f"Coefficients (Linear Regression):\n")
        for feature, coef in zip(features, model.coef_[0]):
            f.write(f"{feature}: {coef:.4f}\n")
        if model.intercept_ is not None:
            f.write(f"Intercept: {model.intercept_[0]:.4f}\n")
    
    # Plot des régressions pour chaque variable indépendante
    for feature in features:
        plot_regression_with_train_test(X_train[[feature]], y_train, X_test[[feature]], y_test, model, feature, dependent_variable_name, working_directory, cohorte_name)


def ridge_regression_model(X_train, y_train, X_test, y_test, working_directory, features, dependent_variable_name, cohorte_name, alpha=1.0):
    """Entraîne un modèle de régression Ridge et évalue le R2."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"R2 score on training set (Ridge): {r2_train:.4f}")
    print(f"R2 score on test set (Ridge): {r2_test:.4f}")

    # Sauvegarde des résultats (optionnel)
    results_path = os.path.join(working_directory, f'ridge_regression_results_{str(alpha)}_{"_".join(features)}.txt')
    with open(results_path, 'w') as f:
        f.write(f"R2 score on training set (Ridge): {r2_train:.4f}\n")
        f.write(f"R2 score on test set (Ridge): {r2_test:.4f}\n")
        f.write(f"Alpha value: {alpha}\n")
    
    # Sauvegarde des coefficients
    coefficients_path = os.path.join(working_directory, f'ridge_regression_coefficients_{"_".join(features)}.txt')
    with open(coefficients_path, 'w') as f:
        f.write(f"Coefficients (Ridge Regression, alpha={alpha}):\n")
        for feature, coef in zip(features, model.coef_[0]):
            f.write(f"{feature}: {coef:.4f}\n")
        if model.intercept_ is not None:
            f.write(f"Intercept: {model.intercept_[0]:.4f}\n")
    
    # Plot des régressions pour chaque variable indépendante
    for feature in features:
        plot_regression_with_train_test(X_train[[feature]], y_train, X_test[[feature]], y_test, model, feature, dependent_variable_name, working_directory, cohorte_name)


def split_data(X, y, test_size=0.2):
    """Divise les données en ensembles d'entraînement et de test."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) 
    return X_train, y_train, X_test, y_test

for machine_learning_features in list_machine_learning_features:
    for cohorte_to_analyse in list_cohorte_to_analyse:
        for dependante_variable in liste_dependante_variable:
            print(f'Analysing : {dependante_variable}')
            print(f"Features analyzed : {'_'.join(machine_learning_features)}")
            ########################################################################
            # Récupération des données
            print(f"Loading data in the {database_path} CSV.")
            print(f"Cohorte to analyse : {cohorte_to_analyse}")
            cohorte_data = pd.read_csv(database_path, sep=';')
            if cohorte_to_analyse == 'Whole':
                working_directory_cohorte = working_directory + f'/Whole_cohorte/Regression/{dependante_variable[0]}/{'_'.join(machine_learning_features)}'
                cohorte_data_to_analyse = cohorte_data # Depending on the analysed cohorte
                liste_variables = machine_learning_features + dependante_variable
                cohorte_data_ml = cohorte_data_to_analyse[liste_variables]
                cohorte_data_ml_without_NA = cohorte_data_ml.dropna()
                if not os.path.exists(working_directory_cohorte):
                    os.makedirs(working_directory_cohorte)
                cohorte_data_ml_without_NA.to_csv(working_directory_cohorte + '/Whole_cohorte.csv', sep=';', index=False)
                X = cohorte_data_ml_without_NA[machine_learning_features]
                if 'Sexe' in machine_learning_features:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try :
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}"]
                    continue
            elif cohorte_to_analyse == 'SAFE':
                working_directory_safe = working_directory + f'/SAFE_J3/Regression/{dependante_variable[0]}/{'_'.join(machine_learning_features)}'
                cohorte_data_SAFE_J3_cleaned = cohorte_data.dropna(subset=['SAFE_J3']).copy()
                independante_variables_to_conserve_safe = machine_learning_features.copy()
                if 'SAFE_J3' in independante_variables_to_conserve_safe:
                    independante_variables_to_conserve_safe.remove('SAFE_J3')
                # Groups constitution : [0-4] | [5-10]dependent_variable, cohorte_to_analyse
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                if 'Sexe' in independante_variables_to_conserve_safe:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_safe, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_more_than_5_subcohorte"]
                    continue
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                if 'Sexe' in independante_variables_to_conserve_safe:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_safe, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_5_subcohorte"]
                    continue
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                if 'Sexe' in independante_variables_to_conserve_safe:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_safe, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_more_than_6_subcohorte"]
                    continue
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_safe]
                if 'Sexe' in independante_variables_to_conserve_safe:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_safe, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_safe_J3_less_than_6_subcohorte"]
                    continue
            elif cohorte_to_analyse == 'PEM':
                working_directory_pem = working_directory + f'/PEM/Regression/{dependante_variable[0]}/{'_'.join(machine_learning_features)}'
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_PEM_pos]
                if 'Sexe' in independante_variables_to_conserve_PEM_pos:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_PEM_pos, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
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
                X = cohorte_data_ml_without_NA[independante_variables_to_conserve_pem]
                if 'Sexe' in independante_variables_to_conserve_pem:
                    X['Sexe'] = X['Sexe'].replace({'H': 0.0, 'F': 1.0})
                y =  cohorte_data_ml_without_NA[dependante_variable]
                try:
                    X_train, y_train, X_test, y_test = split_data(X, y, test_size=test_size)
                    print('Split worked !')
                    linear_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, independante_variables_to_conserve_pem, dependante_variable, cohorte_to_analyse)
                    ridge_regression_model(X_train, y_train, X_test, y_test, working_directory_cohorte, machine_learning_features, dependante_variable, cohorte_to_analyse, alpha=ridge_alpha)
                except Exception as e:
                    print(f"Error append during split_data: {e}")
                    split_error += [f"Cohorte_{cohorte_to_analyse}_{dependante_variable[0]}_according_to_{'_'.join(machine_learning_features)}_pem_negative"]
                    continue
            else:
                print("ERROR IN COHORTE DEFINITION")

if split_error:
    print("\nErrors occurred during data splitting for the following configurations:")
    for error_config in split_error:
        print(f"- {error_config}")