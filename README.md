# Temporal Basis Transformation Network

This repository implements a TensorFlow 2 network layer for temporal
convolution with a set of FIR filters forming a temporal basis.
This code has been tested with TensorFlow 2.4 and Python 3.8.

## Testing

Unit tests use `pytest`. Run
```sh
pip3 install pytest
```
to install `pytest` if it isn't already available on your system. You'll
additionally need to install the `dlop_ldn_function_bases` package,
which in turn depends on `numpy` and `scipy`.
You can find this package [here](https://github.com/astoeckel/dlop_ldn_function_bases).

Simply run `pytest` from the main directory of this repository to run the
unit tests.
