from utils.var_utils_ZZ import MyRYRZ as MyRYRZ_zz
from utils.var_utils import MyRYRZ
from utils.TEvqc_ZZ import MyVQC as MyVQC_zz
from utils.TEvqc import MyVQC
from utils.bc_utils_ver2 import get_breast_cancer_data, kfold_vqc, binary_encoder, U3gate_input_encoder
from utils.quantum_utils import select_features, CustomFeatureMap

import argparse
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import f1_score

from qiskit.aqua.components.optimizers import optimizer
from qiskit.aqua.components.variational_forms.ryrz import RYRZ
from qiskit.circuit.quantumcircuit import QuantumCircuit


from qiskit.circuit import Parameter, QuantumRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import SPSA
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)


def run_exp(
    method='qrac',
    epochs=200,
    positive_factor=1,
    depth=4,
    seed=10598,
    reg=0.,
    model_directory=None,
    result_directory=None
):
    assert method in ['qrac', 'te', 'zz'], f"method {method} not exist"

    if model_directory is None:
        model_directory = f'models/bc_{method}_{epochs}_{positive_factor}_{depth}_{seed}_{reg}_model'
    if result_directory is None:
        result_directory = f'results/bc_{method}_{epochs}_{positive_factor}_{depth}_{seed}_{reg}_result'

    df_train, y_df_train = get_breast_cancer_data()
    y_train = y_df_train.values

    # Get mvp column
    mvp_col = ['tumor-size', 'breast-quad', 'deg-malig', 'age']
    df_train = df_train[mvp_col]
    vqc_gen = None
    feature_map = None
    var_form = None
    X_train = None
    if method in ['zz']:
        X_train = df_train.values
        feature_map = ZZFeatureMap(4)
        var_form = RYRZ(4, depth=depth)
        vqc_gen = VQC

    if method in ['qrac', 'te']:
        X_train = binary_encoder(df_train.values)
        print(X_train)
        num_qubit = len(X_train[0]) // 3
        if method == 'qrac':
            feature_map = CustomFeatureMap('ALL3in1', 1, num_qubit)
            var_form = RYRZ(num_qubit, depth=depth)
            vqc_gen = VQC
        if method == 'te':
            feature_map = QuantumCircuit(num_qubit)
            var_form = MyRYRZ(num_qubit, depth=depth)
            vqc_gen = MyVQC

    assert feature_map is not None, "Feature map is none"
    assert var_form is not None, "Varform is none"

    backend = QasmSimulator({"method": "statevector_gpu"})
    def optimizer_gen(): return SPSA(epochs)

    result = kfold_vqc(feature_map,
                       var_form,
                       backend,
                       optimizer_gen,
                       seed,
                       X_train,
                       y_train,
                       model_directory=model_directory,
                       result_directory=result_directory,
                       k=4,
                       positivedata_duplicate_ratio=positive_factor,
                       vqc_gen=vqc_gen)

    return_result = {
        'train_acc': result['Training accuracies (mean)'][-1],
        'train_f1': result['Training F1 scores (mean)'][-1],
        'test_acc': result['Test accuracies (mean)'][-1],
        'test_f1': result['Test F1 scores (mean)'][-1]
    }

    return return_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Choose parameter and experiment')
    parser.add_argument('--method', dest='method', type=str, default='qrac')
    parser.add_argument('--epochs', dest='epochs', type=int, default=200)
    parser.add_argument('--seed', dest='seed', type=int, default=10598)
    parser.add_argument('--reg', dest='reg', type=float, default=0.)
    parser.add_argument('--depth', dest='depth', type=int, default=4)
    parser.add_argument('--positive_factor',
                        dest='positive_factor', type=float, default=1)
    parser.add_argument('--model_directory',
                        dest='model_directory', type=str, default=None)
    parser.add_argument('--result_directory',
                        dest='result_directory', type=str, default=None)
    args = parser.parse_args()
    result = run_exp(**vars(args))
