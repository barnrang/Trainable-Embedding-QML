from qiskit.aqua import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit import QuantumCircuit
from qiskit.aqua.components import variational_forms
import numpy as np
import itertools
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from utils.var_utils import MyRYRZ
from utils.TEvqc import MyVQC
from utils.quantum_utils import CustomFeatureMap

from qiskit.aqua import set_qiskit_aqua_logging
import logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

def run_exp(
    method='naive',
    epochs=200,
    bit=3,
    dup=1,
    reg=0.,
    depth=4,
    seed=111,
):
    assert bit % 3 == 0, f"number of bit should be x3"
    assert method in ['naive', 'qrac',
                      'te'], f"Method {method} does not exist"
    num_bits = bit

    x_train = []
    y_train = []

    for comb in itertools.product('01', repeat=num_bits):
        comb = [int(x) for x in comb]
        x_train.append(comb)
        y_train.append(sum(comb) % 2)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Encode information
    if method in ['qrac', 'te']:
        num_qubit = num_bits // 3 * dup
        x_st = []
        for x in x_train:
            x_st.append(''.join(x.astype(str)) * dup)

        x_st = np.array(x_st)
    # Naive
    else:
        num_qubit = num_bits
        x_st = []
        for x in x_train:
            x_st.append(''.join(x.astype(str)) * dup)

        x_st = np.array(x_st)

    vqc_ordinal_log = []

    def loss_history_callback(_, __, loss, ___, *args):
        vqc_ordinal_log.append(loss)

    if method == 'naive':
        feature_map = CustomFeatureMap('X', 1, num_qubit)
    elif method == 'te':
        feature_map = QuantumCircuit(num_qubit)
    else:
        feature_map = CustomFeatureMap('ALL3in1', 1, num_qubit)

    if method == 'te':
        var_form = MyRYRZ(num_qubit, depth=depth)
    else:
        var_form = variational_forms.RYRZ(num_qubit, depth=depth)

    training_input = {
        0: x_st[y_train == 0],
        1: x_st[y_train == 1]
    }

    if method == 'te':
        qsvm = MyVQC(SPSA(epochs), feature_map, var_form, training_input,
                   callback=loss_history_callback, lamb=reg)
    else:
        qsvm = VQC(SPSA(epochs), feature_map, var_form,
                   training_input, callback=loss_history_callback)

    qsvm.random.seed(seed)

    backend_options = {"method": "statevector_gpu"}
    backend = QasmSimulator(backend_options)

    quantum_instance = QuantumInstance(
        backend, shots=1024, seed_simulator=seed, seed_transpiler=seed, optimization_level=3)

    result = qsvm.run(quantum_instance)

    y_pred_train = qsvm.predict(x_st)[1]

    # F1 score
    acc = np.mean(y_pred_train == y_train)
    import pickle
    try:
        os.mkdir('models/')
    except:
        pass

    try:
        os.mkdir('results/')
    except:
        pass

    if reg == 0.:
        qsvm.save_model(
            f'models/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}')
        with open(f'results/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}', 'wb') as f:
            pickle.dump([vqc_ordinal_log, acc], f)
    else:
        qsvm.save_model(
            f'models/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}_{reg}')
        with open(f'results/Parity_check_{method}_{bit}_{dup}_{seed}_{depth}_{reg}', 'wb') as f:
            pickle.dump([vqc_ordinal_log, acc], f)

    print("=" * 97)
    print(f"Method: {method} depth={depth} dup={dup} (reg: {reg})")
    print(f"Number of bit {bit}")
    print(f"Classified: {acc*100}%")
    print("=" * 97)

    return {'train_acc': acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose METHOD')
    parser.add_argument('--method', dest='method',
                        type=str, default='qrac')
    parser.add_argument('--epochs', dest="epochs", type=int, default=1)
    parser.add_argument('--dup', dest='dup', type=int, default=1)
    parser.add_argument('--bit', dest='bit', type=int, default=3)
    parser.add_argument('--seed', dest='seed', type=int, default=111)
    parser.add_argument('--reg', dest='reg', type=float, default=0.)
    parser.add_argument('--depth', dest='depth', type=int, default=4)
    args = parser.parse_args()
    run_exp(**vars(args))
