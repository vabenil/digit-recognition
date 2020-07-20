# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.
"""

# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    tr_data, v_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (tr_data, v_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is an array containing 50,000
    ``(x, y)``.  ``x`` is a 784x1 numpy.ndarray
    containing the input image.  ``y`` is a 10x1 numpy.ndarray
    representing the unit vector corresponding to the
    correct digit for ``x``.  ``validation_data`` and ``test_data``
    are lists containing 10,000 ``(x, y)``. In each case,
    ``x`` is a 784x1 numpy.ndarry containing the input image,
    and ``y`` is the expected result corresponding to ``x``.
    """
    dtype = np.dtype([
        ('tx', np.float, (784, 1)), ('ty', np.float, (10, 1))
    ])

    tr_d, va_d, te_d = load_data()
    tr_data = np.array([
        (x.reshape(784, 1), vectorized_result(y))
        for x, y in zip(tr_d[0], tr_d[1])], dtype=dtype)

    v_data = np.array([
        (x.reshape(784, 1), vectorized_result(y))
        for x, y in zip(va_d[0], va_d[1])], dtype=dtype)

    test_data = np.array([
        (x.reshape(784, 1), vectorized_result(y))
        for x, y in zip(te_d[0], te_d[1])], dtype=dtype)
    return (tr_data, v_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
