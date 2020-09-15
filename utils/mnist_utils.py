import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import collections

from utils.quantum_utils import convert_to_angle
from utils.te_layer_tf import EmbeddingTE

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for x,y in zip(xs, ys):
      labels = mapping[tuple(x.flatten())]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(list(labels)[0])
      else:
          # Throw out images that match more than one label.
          pass

    num_3 = sum(1 for value in mapping.values() if True in value)
    num_6 = sum(1 for value in mapping.values() if False in value)
    num_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of 3s: ", num_3)
    print("Number of 6s: ", num_6)
    print("Number of contradictory images: ", num_both)
    print()
    print("Initial number of examples: ", len(xs))
    print("Remaining non-contradictory examples: ", len(new_x))

    return np.array(new_x), np.array(new_y)

def qrac(data):
    data = data.reshape(-1).tolist()
    data += [0] * ((3-len(data)) % 3)


    var_list = []
    for i in range(0, len(data), 3):
        var_list += convert_to_angle(data[i:i+3])

    return var_list

def TE_31(data):
    data = data.reshape(-1).tolist()
    data += [0] * ((3-len(data)) % 3)

    var_list = []
    for i in range(0, len(data), 3):
        encode_dat = data[i:i+3]
        bt = ''
        for x in encode_dat:
            bt += str(x)
        
        var_list += [int(bt, 2)]

    return var_list

def conv(data):
    data = np.squeeze(data)

    var_list = []
    for i in range(4):
        for j in range(2):
            encode_dat = data[i, j:j+3].reshape(-1)
            var_list += convert_to_angle(encode_dat)

    dataT = data.T
    for i in range(4):
        for j in range(2):
            encode_dat = dataT[i, j:j+3].reshape(-1)
            var_list += convert_to_angle(encode_dat)

    return var_list

def conv_TE(data):
    data = np.squeeze(data)

    var_list = []
    for i in range(4):
        for j in range(2):
            encode_dat = data[i, j:j+3].reshape(-1)
            bt = ''
            for x in encode_dat:
                bt += str(x)
            
            var_list += [int(bt, 2)]

    dataT = data.T
    for i in range(4):
        for j in range(2):
            encode_dat = dataT[i, j:j+3].reshape(-1)
            bt = ''
            for x in encode_dat:
                bt += str(x)
            
            var_list += [int(bt, 2)]

    return var_list

def conv_41(data):
    var_list = []
    for i in range(3):
        for j in range(3):
            encode_dat = data[i:i+2, j:j+2].reshape(-1)
            bt = ''
            for x in encode_dat:
                bt += str(x)

            var_list += [int(bt, 2)]

    return var_list

def create_quantum_model(METHOD, num_qubit, LAYER):
    """Create a QNN model circuit and readout operation to go along with it."""
    if METHOD == '16px':
        data_qubits = cirq.GridQubit.rect(4,4)  # a 4x4 grid.
    elif METHOD == '8px':
        data_qubits = cirq.GridQubit.rect(2,4)  # a 2x4 grid.
    else:
        data_qubits = cirq.GridQubit.rect(num_qubit,1)  # a num_qubitx1 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    for i in range(LAYER):
        builder.add_layer(circuit, cirq.XX, f"xx{i+1}")
        builder.add_layer(circuit, cirq.ZZ, f"zz{i+1}")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

def create_normal_model(METHOD, num_qubit, LAYER):
    model_circuit, model_readout = create_quantum_model(METHOD, num_qubit)
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
        tf.keras.layers.Lambda(lambda x: (x+1)/2),
    ])

    return model

def create_TE_model(METHOD, num_qubit, LAYER):
    model_circuit, model_readout = create_quantum_model(METHOD, num_qubit)
    inp = tf.keras.layers.Input((num_qubit,))
    outputs = []
    for i in range(num_qubit):
        if METHOD in ['TE41', 'conv_41', 'TE41_dup', 'conv_41_s2']:
            outputs.append(tf.keras.layers.Embedding(16, 2)(inp[:,i]))
        else:
            outputs.append(tf.keras.layers.Embedding(8, 2)(inp[:,i]))

    outputs = tf.keras.layers.Concatenate()(outputs)
    embed_angle = tf.keras.layers.Concatenate()([outputs[:,::2], outputs[:,1::2]])

    # embed_angle = tf.keras.layers.Lambda(tf.squeeze)(embed_angle)
    output = EmbeddingTE(model_circuit, num_qubit, model_readout, initializer=tf.keras.initializers.RandomNormal(0, 2 * np.pi))(embed_angle)

    output = tf.keras.layers.Lambda(lambda x: (x + 1) / 2)(output)

    model = tf.keras.Model(inp, output)

    return model


def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def convert_to_circuit_8px(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image[1:3,:])
    qubits = cirq.GridQubit.rect(2, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def convert_to_circuit_QRAC(data, num_qubit):
    """Encode truncated classical image by QRAC(3,1)."""
    values = data
    qubits = cirq.GridQubit.rect(num_qubit,1)
    circuit = cirq.Circuit()
    for i in range(len(data) // 2):
        circuit.append(cirq.rx(data[2*i])(qubits[i]))
        circuit.append(cirq.ry(data[2*i+1])(qubits[i]))
    return circuit

def scheduler(epoch, lr):
    if epoch > 5:
        return 1e-3
    else:
        return 1e-2