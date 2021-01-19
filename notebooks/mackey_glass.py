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

from utils import mk_mackey_glass_dataset

def mk_eye_basis(q, N):
    assert q == N
    return np.eye(q)


BASES = [
    (bases.mk_ldn_basis, "ldn"),          # 0
    (bases.mk_dlop_basis, "dlop"),        # 1
    (bases.mk_fourier_basis, "fourier"),  # 2
    (bases.mk_cosine_basis, "cosine"),    # 3
    (bases.mk_haar_basis, "haar"),        # 4
    (mk_eye_basis, "eye"),                # 5
]

    
def run_single_experiment(params, verbose=False):
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
    idcs, basis_idx, seed = params
    basis_ctor, basis_name = BASES[basis_idx]

    # Set the TF seed
    tf.random.set_seed(seed=131 + 513 * seed)

    # Generate the dataset
    N_wnd0, N_wnd1, N_wnd2, N_wnd3 = N_wnds = (16, 8, 8, 4)
    ds_train, ds_val, ds_test = mk_mackey_glass_dataset(N_wnds=N_wnds, seed=seed, verbose=verbose)
    N_wnd = ds_train.element_spec[0].shape[1]
    N_pred = ds_train.element_spec[1].shape[1]
    rms = 0.223

    # Run the experiment
    with tempfile.NamedTemporaryFile() as f:
        N_units0 = 1
        N_units1 = 10
        N_units2 = 10
        N_units3 = 10
        q0, q1, q2, q3 = 16, 8, 8, 4
        H0 = basis_ctor(q0, N_wnd0)
        H1 = basis_ctor(q1, N_wnd1)
        H2 = basis_ctor(q2, N_wnd2)
        H3 = basis_ctor(q3, N_wnd3)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Reshape((N_wnd, 1)),                       # (N_wnd0 + N_wnd1 + N_wnd2 + N_wnd3, 1)
          TemporalBasisTrafo(H0, n_units=N_units0, pad=False),       # (N_wnd1 + N_wnd2 + N_wnd3, q * N_units0)

          tf.keras.layers.Dense(N_units1, activation='relu'),        # (N_wnd1 + N_wnd2 + N_wnd3, N_units1)
          TemporalBasisTrafo(H1, n_units=N_units1, pad=False),       # (N_wnd2 + N_wnd3, q * N_units1)

          tf.keras.layers.Dense(N_units2, activation='relu'),        # (N_wnd2 + N_wnd3, N_units2)
          TemporalBasisTrafo(H2, n_units=N_units2, pad=False),       # (N_wnd3, q * N_units2)

          tf.keras.layers.Dense(N_units3, activation='relu'),        # (N_wnd3, N_units3)
          TemporalBasisTrafo(H3, n_units=N_units3, pad=False),       # (1, q * N_units3)

          tf.keras.layers.Dense(N_pred, use_bias=False),             # (1, N_pred)
          tf.keras.layers.Reshape((N_pred,))                         # (N_pred)
        ])


        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f.name,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.mse,
        )

        model.fit(
            ds_train,
            epochs=20,
            validation_data=ds_val,
            callbacks=[model_checkpoint_callback],
            verbose=verbose,
        )

        model.load_weights(f.name)

        return idcs, np.sqrt(model.evaluate(ds_test, verbose=verbose)) / rms

if __name__ == '__main__':
    multiprocessing.freeze_support()

    basis_idcs = range(len(BASES))
    seeds = range(101)
    params = list([
        ((i, j), basis_idx, seed)
        for i, basis_idx in enumerate(basis_idcs)
        for j, seed in enumerate(seeds)
    ])
    random.shuffle(params)

    errs = np.zeros((len(basis_idcs), len(seeds)))
    with multiprocessing.get_context('spawn').Pool() as pool:
        for (i, k), E in tqdm.tqdm(pool.imap_unordered(run_single_experiment, params), total=len(params)):
            errs[i, k] = E

    np.savez(datetime.datetime.now().strftime("mackey_glass_results_%Y_%m_%d_%H_%M_%S.npz"), **{
        "errs": errs,
        "basis_names": [x[1] for x in BASES],
    })