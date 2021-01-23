# Temporal Basis Transformation Network

This repository implements a TensorFlow 2 Keras network layer for temporal
convolution with a set of FIR filters which may form a temporal basis.

This layer is mostly designed for networks performing online, zero-delay processing of data, i.e., applications where a new output must be produced on every time-step.

This code has been tested with TensorFlow 2.4 and Python 3.8.


## Installation

You can simply install this package via `pip`. For example, run

```sh
pip3 install --user -e .
```

Depending on your environment, you may need to use `pip` instead of `pip3`. Also, if you're inside a virtual environment, you may have to skip the `--user` argument.


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
    # [..., N, units] ==> [..., 1, units * q]
    TemporalBasisTrafo(H=H, units=1),

    # [..., 1, q] ==> [..., 1, 1]
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
