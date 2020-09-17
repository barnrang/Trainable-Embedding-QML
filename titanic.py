from utils.var_utils_ZZ import MyRYRZ as MyRYRZ_zz
from utils.var_utils import MyRYRZ
from utils.TEvqc_ZZ import MyVQC as MyVQC_zz
from utils.TEvqc import MyVQC
from utils.bc_utils_ver2 import kfold_vqc, binary_encoder, U3gate_input_encoder
from utils.quantum_utils import select_features, CustomFeatureMap
from utils.data_provider import load_titanic_pd

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from qiskit.aqua.components.optimizers import optimizer
from qiskit.aqua.components.variational_forms.ryrz import RYRZ
from qiskit.circuit.quantumcircuit import QuantumCircuit


from qiskit.circuit import Parameter, QuantumRegister
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)


def run_exp(
    method='qrac',
    epochs=300,
    positive_factor=1/3,
    depth=4,
    seed=10598,
    reg=0.,
    model_directory=None,
    result_directory=None
):
    assert method in ['qrac', 'qrac_zz', 'te', 'te_zz',
                      'zz_dis', 'zz_dis_cont'], f"method {method} not exist"

    if model_directory is None:
        model_directory = f'models/Titanic_{method}_{epochs}_{positive_factor}_{depth}_{seed}_{reg}_model'
    if result_directory is None:
        result_directory = f'results/Titanic_{method}_{epochs}_{positive_factor}_{depth}_{seed}_{reg}_result'

    all_discrete = method in ['qrac', 'te', 'zz_dis']

    df_train, _, y_train, _ = load_titanic_pd(
        'data/Titanic_train.csv', 'data/Titanic_test.csv', as_category=all_discrete)

    # Get mvp column
    mvp_col = ['Sex', 'Age', 'Pclass', 'Fare']
    df_train = df_train[mvp_col]
    vqc_gen = None
    feature_map = None
    var_form = None
    X_train = None
    if method in ['zz_dis']:
        X_train = df_train.values
        feature_map = ZZFeatureMap(4)
        var_form = RYRZ(4, depth=depth)
        vqc_gen = VQC

    if method in ['zz_dis_cont']:
        df_train['Fare'] = np.log(df_train['Fare'] + 1)
        df_train['Age'] = df_train['Age'] / 60
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

    if method in ['qrac_zz']:
        df_train['Fare'] = np.log(df_train['Fare'] + 1)
        df_train['Age'] = df_train['Age'] / 60
        X_train_num = U3gate_input_encoder(df_train[['Sex', 'Pclass']].values)
        X_train_con = df_train[['Age', 'Fare']].values
        X_train = np.concatenate([X_train_con, X_train_num], axis=1)

        X = [Parameter(f'x[{i}]') for i in range(X_train_num.shape[1])]
        qr = QuantumRegister(1, 'cat')
        qc = QuantumCircuit(qr)

        for i in range(1):
            qc.u3(X[2*i], X[2*i+1], 0, i)
        feature_map_cat = qc

        X1 = Parameter('x[2]')
        X2 = Parameter('x[3]')
        feature_map_con = ZZFeatureMap(2)
        feature_map_con = feature_map_con.assign_parameters([X1, X2])

        feature_map = feature_map_con.combine(feature_map_cat)
        var_form = RYRZ(3, depth=depth)
        vqc_gen = VQC

    if method in ['te_zz']:
        df_train['Fare'] = np.log(df_train['Fare'] + 1)
        df_train['Age'] = df_train['Age'] / 60
        X_train_num = binary_encoder(df_train[['Sex', 'Pclass']].values)
        X_train_con = df_train[['Age', 'Fare']].values

        X_train = np.concatenate([X_train_num, X_train_con], axis=1)

        X1 = Parameter('x[0]')
        X2 = Parameter('x[1]')
        feature_map_con = ZZFeatureMap(2)
        feature_map_con = feature_map_con.assign_parameters([X1, X2])

        qr = QuantumRegister(1, 'cat')
        feature_map_num = QuantumCircuit(qr)

        feature_map = feature_map_num.combine(feature_map_con)
        var_form = MyRYRZ_zz(2, 1, depth=depth)
        vqc_gen = MyVQC_zz

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
    # pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Choose parameter and experiment')
    parser.add_argument('--method', dest='method', type=str, default='qrac', help='Please choose the method from the following [' + ', '.join(['qrac', 'qrac_zz', 'te', 'te_zz',
                                                                                                                                              'zz_dis', 'zz_dis_cont']) + '] (default qrac)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=300)
    parser.add_argument('--seed', dest='seed', type=int,
                        default=10598, help='set random seed')
    parser.add_argument('--reg', dest='reg', type=float,
                        default=0., help='regularized weight (default 0)')
    parser.add_argument('--depth', dest='depth', type=int, default=4, help="Depth of RYRZ variational form (default 4)")
    parser.add_argument('--positive_factor',
                        dest='positive_factor', type=float, default=1/3, help="Augmented data with positive sample ratio (default 1/3)")
    parser.add_argument('--model_directory',
                        dest='model_directory', type=str, default=None, help='folder to save the models')
    parser.add_argument('--result_directory',
                        dest='result_directory', type=str, default=None, help='folder to save the results')
    args = parser.parse_args()
    result = run_exp(**vars(args))
