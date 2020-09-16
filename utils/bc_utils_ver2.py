import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from IPython.display import clear_output
import pickle

from qiskit import BasicAer
from qiskit.providers.aer import QasmSimulator
from qiskit.ml.datasets import *
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Data preprocessing
def get_breast_cancer_data():
    # Read the data
    df = pd.read_csv('data/breast-cancer.data', \
                     header=None, \
                     names=['target', \
                            'age', \
                            'menopause', \
                            'tumor-size', \
                            'inv-nodes', \
                            'node-caps', \
                            'deg-malig', \
                            'breast', \
                            'breast-quad', \
                            'irradiat'])  
    # Ordinal Encoding
    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
  
    return df.drop(['target'], axis=1), df.target

# K-fold Random Forest Classification
def kfold_randomforest(X, y, k=4, seed=123123, positivedata_duplicate_ratio=0):
    print('='*50)
    print(f'{k}-fold Random Forest Classification')
    acc_list, f1_list, feature_importances_list = [], [], []
    np.random.seed(seed)
    for train_id, test_id in KFold(n_splits=k, shuffle=True).split(X):
        # Split the data
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        if positivedata_duplicate_ratio:
            X_duplicate_shape = tuple([0] + list(X_train.shape)[1:])
            X_duplicate = np.empty(X_duplicate_shape)
            if positivedata_duplicate_ratio > 0:
                for i in range(int(positivedata_duplicate_ratio)):
                    X_duplicate = np.concatenate((X_duplicate, X_train[y_train==1]), axis=0)
                X_duplicate = np.concatenate((X_duplicate, X_train[y_train == 1][:int((positivedata_duplicate_ratio - int(positivedata_duplicate_ratio))*X_train[y_train==1].shape[0])]), axis=0)
                X_train = np.concatenate((X_train, X_duplicate), axis=0)
                y_train = np.concatenate((y_train, np.ones(len(X_duplicate))))
            else:
                raise ValueError('Please enter nonnegative real number')
        # Train the model
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        # Get scores and feature importances
        acc_list.append(rf_classifier.score(X_test, y_test))
        f1_list.append(f1_score(y_test, rf_classifier.predict(X_test)))
        feature_importances_list.append(rf_classifier.feature_importances_)

    # Average accuracy and f1 score
    print(f'Mean Accuracy: {np.mean(acc_list):.2%}')
    print(f'Mean F1 score: {np.mean(f1_list):.2%}')
    print('='*50)

    return acc_list, f1_list, feature_importances_list

def randomforest_featureselection(X_df, y_df, k=4, selected_features_num=4, seed=123123, positivedata_duplicate_ratio=0):
    X, y = X_df.values, y_df.values
    acc_list, f1_list, feature_importances_list = kfold_randomforest(X, y, k, seed, positivedata_duplicate_ratio)

    # Feature selection from feature importances
    selected_features = X_df.columns[sorted(range(X.shape[1]), key=lambda i: np.mean(feature_importances_list, axis=0)[i])[:-selected_features_num-1:-1]]
    print('The four most important features are', ", ".join(selected_features[:-1]).upper() + f' and {selected_features[-1].upper()} respectively.')

    # Visualize the feature importances
    plt.figure(figsize=(10,8))
    plt.bar(X_df.columns, np.mean(feature_importances_list, axis=0))
    plt.title('Feature Importances')
    plt.show()

    return acc_list, f1_list, selected_features

# Dictionary to feed VQC
def get_input_dict_for_VQC(X_train, X_test, y_train, y_test, positivedata_duplicate_ratio):
    X_duplicate_shape = tuple([0] + list(X_train.shape)[1:])
    X_duplicate = np.empty(X_duplicate_shape)
    if positivedata_duplicate_ratio >= 0:
        for i in range(int(positivedata_duplicate_ratio)):
            X_duplicate = np.concatenate((X_duplicate, X_train[y_train==1]), axis=0)
        X_duplicate = np.concatenate((X_duplicate, X_train[y_train == 1][:int((positivedata_duplicate_ratio - int(positivedata_duplicate_ratio))*X_train[y_train==1].shape[0])]), axis=0)
    else:
        raise ValueError('Please enter nonnegative real number')
    training_input = { 0: X_train[y_train == 0],
                       1: np.concatenate((X_train[y_train == 1], X_duplicate), axis=0) }
    test_input = { 0: X_test[y_test == 0],
                   1: X_test[y_test == 1] }
    return training_input, test_input

# Train VQC
def train_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer, \
              seed, \
              X_train, X_test, y_train, y_test, \
              model_directory, \
              result_filename, \
              positivedata_duplicate_ratio=1, \
              shots=1024,
              vqc_gen=None):
  
    # Input preparation
    # Input dict
    training_input, test_input = get_input_dict_for_VQC(X_train, X_test, y_train, y_test, positivedata_duplicate_ratio)
    # Quantum instance
    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)
    print('='*100 + f'\nWorking directory: {os.getcwd()}\n' + '='*100)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)  

    # Callback function for collecting models' parameters and losses along the way
    training_loss_list, training_acc_list, training_f1_list = [], [], []
    validation_loss_list, validation_acc_list, validation_f1_list = [], [], []
    def callback_collector(step, model_params, loss, _, *args):
        # Save the temp model
        temp_model_filename = os.path.join(model_directory, f'step{step+1}.npz')
        np.savez(temp_model_filename, opt_params = model_params)
        # Load the temp model
        # vqc_val = VQC(optimizer, feature_map, var_form, training_input, test_input, quantum_instance=quantum_instance)
        if vqc_gen is None:
            vqc_val = VQC(optimizer, feature_map, var_form, training_input, test_input, quantum_instance=quantum_instance)
        else:
            vqc_val = vqc_gen(optimizer, feature_map, var_form, training_input, test_input, quantum_instance=quantum_instance)
        vqc_val.load_model(temp_model_filename)
        # Collect training loss and accuracy
        y_train_prob, y_train_pred = vqc_val.predict(X_train)
        train_loss = -np.mean(y_train*np.log(y_train_prob[:,1]) + (1 - y_train)*np.log(y_train_prob[:,0]))
        training_loss_list.append(train_loss)
        training_acc_list.append(np.mean(y_train_pred==y_train))
        training_f1_list.append(f1_score(y_train, y_train_pred))
        # Collect validation loss and accuracy
        y_test_prob, y_test_pred = vqc_val.predict(X_test)
        val_loss = -np.mean(y_test*np.log(y_test_prob[:,1]) + (1 - y_test)*np.log(y_test_prob[:,0]))
        validation_loss_list.append(val_loss)
        validation_acc_list.append(np.mean(y_test_pred==y_test))
        validation_f1_list.append(f1_score(y_test, y_test_pred))

    # Run VQC
    if vqc_gen is None:
        vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, callback=callback_collector, quantum_instance=quantum_instance)
    else:
        vqc = vqc_gen(optimizer, feature_map, var_form, training_input, test_input, callback=callback_collector, quantum_instance=quantum_instance)
    vqc.random.seed(seed)
    result = vqc.run()
    clear_output()

    # Save results
    result['Training losses'], result['Validation losses'] = np.array(training_loss_list), np.array(validation_loss_list)
    result['Training F1 scores'], result['Training accuracies'] = np.array(training_f1_list), np.array(training_acc_list)
    result['Test F1 scores'], result['Test accuracies'] = np.array(validation_f1_list), np.array(validation_acc_list)
    if os.path.dirname(result_filename):
        if not os.path.isdir(os.path.dirname(result_filename)):
            os.makedirs(os.path.dirname(result_filename))
    with open(result_filename, 'wb') as f:
        pickle.dump(result, f)
    
    # Report final results
    print('Trained successfully!')
    print(f'Final accuracy (test set): {validation_acc_list[-1]:.2%} | Final accuracy (training set): {training_acc_list[-1]:.2%}')
    print(f'Final F1 score (test set): {validation_f1_list[-1]:.2%} | Final F1 score (training set): {training_f1_list[-1]:.2%}')
    print(f'All models are saved at {os.path.join(os.getcwd(), model_directory)}.')
    print(f'Result is saved at {os.path.join(os.getcwd(), result_filename)}.')

    return result

def kfold_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer_generator, \
              seed, \
              X, y, \
              model_directory, \
              result_directory, \
              k=5, \
              positivedata_duplicate_ratio=1, \
              shots=1024, \
              seed_kfold=123123,
              vqc_gen=None):

    print('='*100)
    print(f'{k}-fold VQC Classification')
    # Create working directory (if needed)
    print('='*100 + f'\nWorking directory: {os.getcwd()}\n' + '='*100)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
    # Final result initialization (dict)
    params_to_collect = ['Training losses', 'Validation losses', \
                         'Training accuracies', 'Test accuracies', \
                         'Training F1 scores', 'Test F1 scores']
    result = {key:[] for key in params_to_collect}
    # result['Default test accuracies'] = [] # Uncomment for validating the predicted accuracy
    np.random.seed(seed_kfold)
    kf = KFold(n_splits=k, shuffle=True)
    kf_id = list(kf.split(X))
    for (fold, (train_id, test_id)) in enumerate(kf_id, start=1):
        print('='*100 + f'\nFold number {fold}\n' + '='*100)
        # Split the data
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        # Train a model
        model_directory_fold = os.path.join(model_directory, f'foldnumber{fold}') 
        result_filename_fold = os.path.join(result_directory, f'foldnumber{fold}.pkl')
        optimizer = optimizer_generator()
        result_onefold = train_vqc(feature_map, \
                                var_form, \
                                backend, \
                                optimizer, \
                                seed, \
                                X_train, X_test, y_train, y_test, \
                                model_directory_fold, \
                                result_filename_fold, \
                                positivedata_duplicate_ratio, \
                                shots,
                                vqc_gen)
        # Collect results
        # result['Default test accuracies'].append(result_onefold['testing_accuracy']) # Uncomment for validating the predicted accuracy
        for key in params_to_collect:
            result[key].append(result_onefold[key])
        
    # Average accuracies and f1 scores
    dict_items_without_meanvalues = list(result.items())
    for key, value in dict_items_without_meanvalues:
        result[key + ' (mean)'] = np.mean(value, axis=0)
    # Convert to numpy arrays
    for key, value in result.items():
        if type(value)==list:
            result[key] = np.array(value)
    # Save final results
    result_filename_allfolds = os.path.join(result_directory, f'allfolds.pkl')
    with open(result_filename_allfolds, 'wb') as f:
        pickle.dump(result, f)
    clear_output()
    print('='*97)
    print('='*35 + f' {k}-fold VQC Classification ' + '='*35)
    print(f"Final training accuracy (mean): {result['Training accuracies (mean)'][-1]:.2%} | Final test accuracy (mean): {result['Test accuracies (mean)'][-1]:.2%}")
    print(f"Final training F1 score (mean): {result['Training F1 scores (mean)'][-1]:.2%} | Final test F1 score (mean): {result['Test F1 scores (mean)'][-1]:.2%}")
    print(f'All models are saved at {os.path.join(os.getcwd(), model_directory)}.\nResults are saved at {os.path.join(os.getcwd(), result_directory)}.')
    print('='*97)

    return result

# Convert a 3-bit string into inputs for U3 gate
def convert_to_angle(b_st):
    if b_st=='111':
        return [np.arccos(1/np.sqrt(3)), np.pi/4]
    if b_st=='110':
        return [np.arccos(1/np.sqrt(3)), 3*np.pi/4]
    if b_st=='101':
        return [np.arccos(1/np.sqrt(3)), -np.pi/4]
    if b_st=='100':
        return [np.arccos(1/np.sqrt(3)), -3*np.pi/4]
    if b_st=='011':
        return [np.arccos(-1/np.sqrt(3)), np.pi/4]
    if b_st=='010':
        return [np.arccos(-1/np.sqrt(3)), 3*np.pi/4]
    if b_st=='001':
        return [np.arccos(-1/np.sqrt(3)), -np.pi/4]
    if b_st=='000':
        return [np.arccos(-1/np.sqrt(3)), -3*np.pi/4]

# Binary Encoder
def binary_encoder(X):
    # The number of necessary bits in each column
    bit_each_col = [int(np.ceil(np.log2(len(np.unique(X[:,col]))))) for col in range(X.shape[1])]
    # Padding check in order to make an input string into quantum circuit divisible by three
    if sum(bit_each_col)%3 != 0:
        pad = 3 - (sum(bit_each_col)%3)
    else:
        pad = 0 
    # Encode X into a binary string
    X_binary_encoded = []
    for sample in X:
        bit_string = ''
        for value, num_bit in zip(sample, bit_each_col):
            bit_string += f'{value:010b}'[-num_bit:]
        bit_string += pad*'0'
        X_binary_encoded.append(bit_string)
    return np.array(X_binary_encoded)

# U3gate Input Encoder
def U3gate_input_encoder(X):
    X_binary_encoded = binary_encoder(X)
    if len(X_binary_encoded[0]) % 3 != 0:
        raise ValueError('The input string is not divisible by three')
    else:
        X_U3gate_input = []
        for bitstring in X_binary_encoded:
            U3gate_input = []
            for qubit in range(len(X_binary_encoded[0])//3):
                U3gate_input.extend(convert_to_angle(bitstring[qubit*3: (qubit+1)*3]))
            X_U3gate_input.append(U3gate_input)
        return np.array(X_U3gate_input)