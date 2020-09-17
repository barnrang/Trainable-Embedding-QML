from utils.mnist_utils import *
import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


THRESHOLD = 0.5


def run_exp(
    method='qrac',
    batch=32,
    epochs=10,
    depth=1,
    seed=111,
    result_filename=None
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if result_filename is None:
        result_filename = f"results/{method}_{epochs}_{depth}_{seed}_history.pk"

    input_converters = {
        'qrac': qrac,
        'conv': conv,
        'te': TE_31,
        'conv_te': conv_TE,
        'conv_41': conv_41
    }

    circuit_makers = {
        'QRAC': convert_to_circuit_QRAC,
        '16px': convert_to_circuit,
        '8px': convert_to_circuit_8px
    }

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / \
        255.0, x_test[..., np.newaxis]/255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))

    x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
    x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

    # Filter same images
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.int32)
    x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.int32)

    # Need to convert angle or embedding index
    if method not in ['8px', '16px']:
        data_train, data_test = [], []
        for row in x_train_bin:
            data_train.append(input_converters[method](row))

        for row in x_test_bin:
            data_test.append(input_converters[method](row))

        data_train = np.array(data_train)
        data_test = np.array(data_test)
        if method in ['qrac', 'conv']:
            num_qubit = data_train.shape[1]//2
        else:
            num_qubit = data_train.shape[1]
    else:
        data_train = x_train_bin
        data_test = x_test_bin
        # No use later
        num_qubit = None

    # Create input circuit
    if method in ['8px', '16px']:
        x_train_circ = [circuit_makers[method](x) for x in data_train]
        x_test_circ = [circuit_makers[method](x) for x in data_test]
    elif method in ['qrac', 'conv']:
        x_train_circ = [circuit_makers['QRAC']
                        (x, num_qubit) for x in data_train]
        x_test_circ = [circuit_makers['QRAC'](x, num_qubit) for x in data_test]
    else:
        x_train_circ = None
        x_test_circ = None

    if method in ['te', 'conv_te', 'conv_41']:
        # Input the numpy directly
        x_train_tfcirc = data_train
        x_test_tfcirc = data_test
    else:
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    # Create model circuit

    if method in ['te', 'conv_te', 'conv_41']:
        model = create_TE_model(method, num_qubit, depth)
    else:
        model = create_normal_model(method, num_qubit, depth)

    cb = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)]

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['acc'])

    print(epochs)

    # Train model
    qnn_history = model.fit(
        x_train_tfcirc, y_train_nocon,
        batch_size=32,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test),
        callbacks=cb)

    qnn_results = model.evaluate(x_test_tfcirc, y_test)
    import pickle
    with open(result_filename, 'wb') as f:
        pickle.dump(qnn_history.history, f)

    return_result = {
        'train_acc': qnn_history.history['acc'][-1],
        'test_acc': qnn_history.history['val_acc'][-1]
    }

    return return_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose method')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help="Number of epochs (default 10)")
    parser.add_argument('--batch', dest='batch', type=int, default=32, help="Batch size (default 32)")
    parser.add_argument('--depth', dest='depth', type=int, default=1, help="Depth of NN (default 1)")
    parser.add_argument('--method', dest='method', type=str, default='16px', help="Choose from the following method [16px, 8px, qrac, conv, conv_41] (default 16px)")
    parser.add_argument('--seed', dest='seed', type=int, default=111)
    parser.add_argument('--result_filename',
                        dest='result_filename', type=str, default=None)
    args = parser.parse_args()
    qnn_results = run_exp(**vars(args))
    print(qnn_results)
