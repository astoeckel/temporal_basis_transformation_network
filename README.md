# Temporal Basis Transformation Network

This repository implements a TensorFlow 2 Keras network layer for temporal
convolution with a set of FIR filters forming a temporal basis (also known as
a “generalised Fourier transformation”).

This code has been tested with TensorFlow 2.4 and Python 3.8.

## Usage

```python
# TensorFlow 2.4 or later
import tensorflow as tf

# Import the TemporalBasisTrafo layer from this package
from temporal_basis_transformation_network.keras import TemporalBasisTrafo

# See https://github.com/astoeckel/dlop_ldn_function_bases
import dlop_ldn_function_bases as bases

# Generate a Legendre Delay Network basis
q, N = 20, 100 # Compress N=100 samples into q=20 dimensions
H = bases.mk_ldn_basis(q=q, N=N)

# Build a simple model with a linear readout
model = tf.keras.models.Sequential([
    # [n_batch, N] ==> [n_batch, N, q]
    TemporalBasisTrafo(H=H, n_units=1),

    # [n_batch, N, q] ==> [n_batch, N, 1]
    tf.keras.layers.Dense(1, activation='linear', use_bias=False)
])

# Compile, fit, evaluate the model as usual...
```


## Dependencies

This code has no dependencies apart from TensorFlow 2.4 or later and numpy
1.19 or later.

However, to run the unit tests or to play around with the Jupyter notebooks
in the `notebooks` folder, you need to install `scipy`, and
`matplotlib`, as well as the `dlop_ldn_function_bases` package, which
can be found [here](https://github.com/astoeckel/dlop_ldn_function_bases).

## Testing

Unit tests use `pytest`. Run
```sh
pip3 install pytest
```
to install `pytest` if it isn't already available on your system.

Simply run `pytest` from the main directory of this repository to run the
unit tests.
