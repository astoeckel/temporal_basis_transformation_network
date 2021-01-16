#!/usr/bin/env python3

import numpy as np
import utils
import tempfile
import itertools
import multiprocessing
import tqdm
import random
import datetime

import dlop_ldn_function_bases as bases

MNIST = {
    "train_imgs": utils.read_idxgz('datasets/train-images-idx3-ubyte.gz'),
    "train_lbls": utils.read_idxgz('datasets/train-labels-idx1-ubyte.gz'),
    "test_imgs": utils.read_idxgz('datasets/t10k-images-idx3-ubyte.gz'),
    "test_lbls": utils.read_idxgz('datasets/t10k-labels-idx1-ubyte.gz')
}


def mk_psmnist_dataset(n_validate=10000, seed=103891, batch_size=100):
    import tensorflow as tf

    # Generate a random number generator for the given seed
    rng = np.random.RandomState(57503 + 15173 * seed)

    mnist_train_orig_imgs, mnist_train_orig_lbls, \
    mnist_test_imgs, mnist_test_lbls = \
        np.copy(MNIST["train_imgs"]), np.copy(MNIST["train_lbls"]), \
        np.copy(MNIST["test_imgs"]), np.copy(MNIST["test_lbls"])

    # Randomly split the validation dataset off the validation data
    idcs = rng.permutation(np.arange(mnist_train_orig_imgs.shape[0]))
    idcs_train = idcs[n_validate:]
    idcs_val = idcs[:n_validate]

    mnist_train_imgs = mnist_train_orig_imgs[idcs_train]
    mnist_train_lbls = mnist_train_orig_lbls[idcs_train]

    mnist_val_imgs = mnist_train_orig_imgs[idcs_val]
    mnist_val_lbls = mnist_train_orig_lbls[idcs_val]

    # Generate a random permutation of the pixels
    perm = rng.permutation(np.arange(28 * 28))
    def permute(imgs):
        res_imgs = np.zeros((imgs.shape[0], 28 * 28), dtype=np.float32)
        for i in range(imgs.shape[0]):
            res_imgs[i] = 2.0 * imgs[i].astype(np.float32).flatten()[perm] / 255.0 - 1.0
        return res_imgs

    ds_train = tf.data.Dataset.from_tensor_slices((permute(mnist_train_imgs), mnist_train_lbls))
    ds_train = ds_train.batch(batch_size)

    ds_val = tf.data.Dataset.from_tensor_slices((permute(mnist_val_imgs), mnist_val_lbls))
    ds_val = ds_val.batch(batch_size)

    ds_test = tf.data.Dataset.from_tensor_slices((permute(mnist_test_imgs), mnist_test_lbls))
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_val, ds_test


BASES = [
    (bases.mk_ldn_basis, "ldn"),
    (bases.mk_dlop_basis, "dlop"),
    (bases.mk_fourier_basis, "fourier"),
    (bases.mk_cosine_basis, "cosine"),
]

def run_single_experiment(params):
    import tensorflow as tf
    from temporal_basis_transformation_network.keras import TemporalBasisTrafo

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


    # Destructure the parameters
    idcs, basis_idx, q, seed = params
    basis_ctor, basis_name = BASES[basis_idx]

    # Generate a dataset
    ds_train, ds_val, ds_test = mk_psmnist_dataset(seed=seed)

    # Run the experiment
    with tempfile.NamedTemporaryFile() as f:
        N = 28 * 28
        N_neurons = 346
        N_units = 1
        H = basis_ctor(q, N)

        model = tf.keras.models.Sequential([
          tf.keras.layers.Reshape((N, 1)),                      # (N, 1)
          TemporalBasisTrafo(H, n_units=N_units, pad=False),    # (1, q)
          tf.keras.layers.Dropout(0.5),                         # (1, q)
          tf.keras.layers.Dense(N_neurons, activation='relu'),  # (1, N_neurons)
          tf.keras.layers.Dense(10, use_bias=False),            # (1, 10)
          tf.keras.layers.Reshape((10,))                        # (10)
        ])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f.name,
            save_weights_only=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            save_best_only=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model.fit(
            ds_train,
            epochs=20,
            validation_data=ds_val,
            callbacks=[model_checkpoint_callback],
            verbose=False,
            use_multiprocessing=False,
        )

        model.load_weights(f.name)

        return idcs, model.evaluate(ds_test, verbose=False)[1] # Return the test accuracy

if __name__ == '__main__':
    multiprocessing.freeze_support()

    qs = np.unique(np.logspace(np.log2(1), np.log2(484), 50, base=2, dtype=np.int))
    basis_idcs = range(len(BASES))
    seeds = range(10)
    params = list([
        ((i, j, k), basis_idx, q, seed)
        for i, basis_idx in enumerate(basis_idcs)
        for j, q in enumerate(qs)
        for k, seed in enumerate(seeds)
    ])
    random.shuffle(params)

    errs = np.zeros((len(basis_idcs), len(qs), len(seeds)))
    with multiprocessing.get_context('spawn').Pool(32) as pool:
        for (i, j, k), E in tqdm.tqdm(pool.imap_unordered(run_single_experiment, params), total=len(params)):
            errs[i, j, k] = E

    np.savez(datetime.datetime.now().strftime("psmnist_results_%Y_%m_%d_%H_%M_%S.npz"), **{
        "errs": errs,
        "basis_names": [x[1] for x in BASES],
        "qs": qs,
    })