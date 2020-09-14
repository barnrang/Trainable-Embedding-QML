from qiskit import BasicAer
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, QuantumRegister

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def encoder_3bits_1qubit(df_q):
    data = []

    for col_name in df_q.columns:
        num_cat = len(np.unique(df_q[col_name]))
        num_bit = int(np.ceil(np.log2(num_cat)))

        print("Number of categories: %d\nNumber of bits: %d" % (num_cat, num_bit))

        # Padding to x3
        if num_bit % 3 != 0:
            num_bit = num_bit + (3 - (num_bit % 3))

        num_qubit = num_bit // 3

        features = []
        for size in df_q[col_name]:
            # Get last num_bit digit and reverse
            # 11 --> 001  | 011 --> 110 | 100

            all_b_st = f"{size:010b}"[-num_bit:][::-1]

            var_list = []
            for i in range(num_qubit):
                b_st = all_b_st[i * 3: (i + 1) * 3]

                # b_st = b_1, b_2, b_3 = \sqrt{3}r_x, \sqrt{3}r_y, \sqrt{3}r_z

                if b_st[0] == '1':
                    theta = np.arccos(1 / np.sqrt(3))
                else:
                    theta = np.arccos(-1 / np.sqrt(3))

                if b_st[1] == '1' and b_st[2] == '1':
                    varphi = np.pi / 4

                if b_st[1] == '1' and b_st[2] == '0':
                    varphi = 3 * np.pi / 4

                if b_st[1] == '0' and b_st[2] == '0':
                    varphi = -3 * np.pi / 4

                if b_st[1] == '0' and b_st[2] == '1':
                    varphi = -np.pi / 4

                var_list += [theta, varphi]

            features.append(var_list)
        #         print(size, var_list)
        data.append(np.array(features))

    data = np.concatenate(data, axis=1)
    return data


def select_features(df_train, y_train, df_test=None, y_test=None, feat_num=2, modelname="RandomForestClassifier"):
    # model
    print("-----\nFull features:")
    if modelname == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        raise NameError("Only support RandomForestClassifier.")
    model.fit(df_train, y_train)
    # Train score
    print("Final train score: %f" % model.score(df_train, y_train))
    # F1 score
    print("Final F1 score: %f" % f1_score(y_train, model.predict(df_train)))

    print("-----\nMajority")
    print("Final train acc: %f\nFinal train F1:%f" % (
    np.mean(np.zeros_like(y_train) == y_train), f1_score(np.zeros_like(y_train), y_train)))

    # select important features
    col_num = feat_num
    mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                      key=lambda x: model.feature_importances_[x],
                                      reverse=True)].tolist()
    print("Feature rank based on importance")
    print(mvp_col)
    mvp_col = mvp_col[:col_num]
    print("Selected features: %s" % ",".join(mvp_col))
    return mvp_col


class CustomFeatureMap(FeatureMap):
    def __init__(self, circuit_type, depth, feature_dimension, qmap=None):
        super().__init__()
        self._feature_dimension = feature_dimension
        self._num_qubits = feature_dimension
        self._depth = depth
        self._circuit_type = circuit_type
        self._qmap = qmap

    def construct_circuit(self, x, qr):
        qc = QuantumCircuit(qr)

        from math import sqrt, cos, acos, pi, sin
        from numpy import exp

        #compute the value of theta
        theta = acos(sqrt(0.5 + sqrt(3.0)/6.0))

        #to record the u3 parameters for encoding 000, 010, 100, 110, 001, 011, 101, 111
        rotationParams = {
                           "000": np.array( [ cos(theta), sin(theta)*exp(1j*pi/4) ] ),
                           "001": np.array( [ cos(theta), sin(theta)*exp(-1j*pi/4)] ),
                           "010": np.array( [ cos(theta), sin(theta)*exp(3j*pi/4) ] ),
                           "011": np.array( [ cos(theta), sin(theta)*exp(-3j*pi/4)] ),
                           "100": np.array( [ sin(theta), cos(theta)*exp(1j*pi/4) ] ),
                           "101": np.array( [ sin(theta), cos(theta)*exp(-1j*pi/4)] ),
                           "110": np.array( [ sin(theta), cos(theta)*exp(3j*pi/4) ] ),
                           "111": np.array( [ sin(theta), cos(theta)*exp(-3j*pi/4)] )
                           #"000": np.array([ np.sqrt( 1/2 + 1/2/np.sqrt(3) ), 1/np.sqrt( 6 + 2*np.sqrt(3) ) * ( 1 + 1j ) ]),
                           #"100": np.array([ np.sqrt( 1/2 + 1/2/np.sqrt(3) ), 1/np.sqrt( 6 + 2*np.sqrt(3) ) * ( -1 + 1j ) ]),
                           #"010": np.array([ np.sqrt( 1/2 + 1/2/np.sqrt(3) ), 1/np.sqrt( 6 + 2*np.sqrt(3) ) * ( 1 - 1j ) ]),
                           #"110": np.array([ np.sqrt( 1/2 + 1/2/np.sqrt(3) ), 1/np.sqrt( 6 + 2*np.sqrt(3) ) * ( -1 - 1j ) ]),
                           #"001": np.array([ np.sqrt( 1/2 - 1/2/np.sqrt(3) ), 1/np.sqrt( 6 - 2*np.sqrt(3) ) * ( 1 + 1j ) ]),
                           #"101": np.array([ np.sqrt( 1/2 - 1/2/np.sqrt(3) ), 1/np.sqrt( 6 - 2*np.sqrt(3) ) * ( -1 + 1j ) ]),
                           #"011": np.array([ np.sqrt( 1/2 - 1/2/np.sqrt(3) ), 1/np.sqrt( 6 - 2*np.sqrt(3) ) * ( 1 - 1j ) ]),
                           #"111": np.array([ np.sqrt( 1/2 - 1/2/np.sqrt(3) ), 1/np.sqrt( 6 - 2*np.sqrt(3) ) * ( -1 - 1j ) ])
                          }
        sq3= 1./np.sqrt(3)
        threetwoqracs = {
                           "000": np.array( [ 1., 0., 0., 0. ] ),
                           "001": np.array( [ sq3, sq3, sq3, 0. ] ),
                           "010": np.array( [ sq3, -sq3, 0, sq3 ] ),
                           "011": np.array( [ 0, 1., 0., 0.] ),
                           "100": np.array( [ -sq3, 0, sq3, sq3 ] ),
                           "101": np.array( [ 0, 0, 1, 0] ),
                           "110": np.array( [ 0, 0, 0, 1 ] ),
                           "111": np.array( [ 0, sq3, -sq3, sq3] )
        }

        if self._circuit_type == "X":
            assert len(x) == len(qr), "Number of qubit not match"
            n = len(x)
            for i in range(n):
                if x[i] == '1': qc.x(i)


        if self._circuit_type == "ALL3in2":
            """ using (3,2)-QRAC for encoding 3 bits """
            assert len(x) % 3 == 0, "at construct_circuit, the length of x must be multiplier of 3"
            n = len(x) // 3
            assert len(qr) >= n, "at construct_circuit, the number of qubits must be enough to encode binary string"
            for i in range(n):
                xi = "".join([ str(_) for _ in x[3*i:3*(i+1)] ])
                v = threetwoqracs[xi]
                qc.initialize(v, qr[2*i:2*(i+1)])

        elif self._circuit_type == "ALL3in1":
            #rotate qr[0] by 3bits of x by appending "0" if less than 3 are encountered
            #assume x is a string
            assert len(x) % 3 == 0, "at construct_circuit, the length of x must be multiplier of 3"
            n = len(x) // 3
            assert len(qr) >= n, "at construct_circuit, the number of qubits must be enough to encode binary string"
            for i in range(n):
                xi = "".join([ str(_) for _ in x[3*i:3*(i+1)] ])
                if self._qmap is None:
                    v = rotationParams[xi]
                else:
#                     print('yeah')
                    v = rotationParams[self._qmap[i][xi]]
                qc.initialize(v, qr[i])
            #for i in range(n, len(qr)):
            #    qc.h(qr[i])
            #    qc.s(qr[i])
            for i in range(n, len(qr)):
                ii = n - i
                xii = "".join([ str(_) for _ in x[3*ii:3*(ii+1)] ])
                if self._qmap is None:
                    v = rotationParams[xii]
                else:
                    v = rotationParams[self._qmap[i][xii]]
                qc.initialize(v, qr[i])

        else: #ALL2in1 and embedded into (3,1)-QRAC
            #print("Original x:", "".join([ str(_) for _ in x ]))

            assert (len(x) // 2) + (len(x) % 2) <= len(qr), "number of qubits is not enough"
            newx = [ _ for _ in x ]
            if len(x) % 2 == 1:
                newx.append(0) #appending to make it even length
            n = len(newx) // 2
            for i in range(n):
                xi = "".join( [ str(_) for _ in newx[2*i:2*(i+1)] ] )
                if xi in ("00", "11"):
                    xi += "0"
                else:
                    xi += "1"
                #print("\t",xi)
                v = rotationParams[xi]
                qc.initialize(v, qr[i])


        #qc.barrier()

        return transpile(qc, optimization_level=1, basis_gates=['u3','cx'])
