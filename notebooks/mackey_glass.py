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

from utils import mackey_glass

def mk_mackey_glass_dataset(dt=0.1, T=10e3, tau=17.0, N_wnd=171,
                            N_pred=512, N_smpls=100000, N_test=20000, N_batch=100,
                            seed=4718):
    import tensorflow as tf

    # Create two MG datasets; one for training and validation, and one for\
    # testing
    rng = np.random.RandomState(57503 + 15173 * seed)
    xs = mackey_glass(int(T / dt), tau=tau, dt=dt, rng=rng)
    xs_test = mackey_glass(int(T / dt), tau=tau, dt=dt, rng=rng)

    # Randomly slice observation and prediction windows out of the dataset
    smpls_x, smpls_x_test = np.zeros((2, N_smpls, N_wnd))
    smpls_t, smpls_t_test = np.zeros((2, N_smpls, N_pred))
    for i in range(N_smpls):
        i0 = rng.randint(N_smpls - N_wnd - N_pred)
        i1, i2 = i0 + N_wnd, i0 + N_wnd + N_pred
        smpls_x[i] = xs[i0:i1]
        smpls_t[i] = xs[i1:i2]
        smpls_x_test[i] = xs_test[i0:i1]
        smpls_t_test[i] = xs_test[i1:i2]

    # Compute which samples to use as training, valuation and test data
    i_train0, i_train1 = 0, N_smpls - 2 * N_test
    i_val0, i_val1 = N_smpls - 2 * N_test, N_smpls - N_test
    i_test0, i_test1 = N_smpls - N_test, N_smpls

    ds_train = tf.data.Dataset.from_tensor_slices((
                smpls_x[i_train0:i_train1], smpls_t[i_train0:i_train1]))
    ds_train = ds_train.shuffle(i_train1 - i_train0)
    ds_train = ds_train.batch(N_batch)

    ds_val = tf.data.Dataset.from_tensor_slices((
                smpls_x[i_val0:i_val1], smpls_t[i_val0:i_val1]))
    ds_val = ds_val.batch(N_batch)

    ds_test = tf.data.Dataset.from_tensor_slices((
                smpls_x_test[i_test0:i_test1], smpls_t_test[i_test0:i_test1]))
    ds_test = ds_test.batch(N_batch)

    return ds_train, ds_val, ds_test

BASES = [
    (bases.mk_ldn_basis, "ldn"),
    (bases.mk_dlop_basis, "dlop"),
    (bases.mk_fourier_basis, "fourier"),
    (bases.mk_cosine_basis, "cosine"),
    (bases.mk_haar_basis, "haar"),
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
    ds_train, ds_val, ds_test = mk_mackey_glass_dataset(seed=seed)

    # Fetch the window and the prediction size
    N_wnd = ds_train.element_spec[0].shape[1]
    N_pred = ds_train.element_spec[1].shape[1]

    # Run the experiment
    with tempfile.NamedTemporaryFile() as f:
        N_units = 1
        N_neurons = 500
        H = basis_ctor(q, N_wnd)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Reshape((N_wnd, 1)),                   # (N_wnd, 1)
          TemporalBasisTrafo(H, n_units=N_units, pad=False),     # (1, q)
          tf.keras.layers.Dense(N_neurons, activation='relu'),   # (1, N_neurons)
          tf.keras.layers.Dropout(0.5),                          # (1, N_neurons)
          tf.keras.layers.Dense(N_neurons, activation='relu'),   # (1, N_neurons)
          tf.keras.layers.Dropout(0.5),                          # (1, N_neurons)
          tf.keras.layers.Dense(N_pred, use_bias=False),         # (1, N_wnd)
          tf.keras.layers.Reshape((N_pred,))                     # (N_pred)
        ])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f.name,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.mse,
            metrics=['mse',]
        )

        model.fit(
            ds_train,
            epochs=20,
            validation_data=ds_val,
            callbacks=[model_checkpoint_callback],
            verbose=False,
        )

        model.load_weights(f.name)

    return idcs, np.sqrt(model.evaluate(ds_test, verbose=False)[0]) # Return the RMSE

if __name__ == '__main__':
    multiprocessing.freeze_support()

    qs = np.linspace(1, 32, 17, dtype=np.int)
    basis_idcs = range(len(BASES))
    seeds = range(11)
    params = list([
        ((i, j, k), basis_idx, q, seed)
        for i, basis_idx in enumerate(basis_idcs)
        for j, q in enumerate(qs)
        for k, seed in enumerate(seeds)
    ])
    random.shuffle(params)
    print(params)

    errs = np.zeros((len(basis_idcs), len(qs), len(seeds)))
    with multiprocessing.get_context('spawn').Pool() as pool:
        for (i, j, k), E in tqdm.tqdm(pool.imap_unordered(run_single_experiment, params), total=len(params)):
            errs[i, j, k] = E

    np.savez(datetime.datetime.now().strftime("mackey_glass_results_%Y_%m_%d_%H_%M_%S.npz"), **{
        "errs": errs,
        "basis_names": [x[1] for x in BASES],
        "qs": qs,
    })