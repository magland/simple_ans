# simple_ans

A Python package that provides lossless compression of integer datasets through [Asymmetric Numeral Systems (ANS)](https://ieeexplore.ieee.org/document/7170048), implemented in [C++](./simple_ans/cpp) with pybind11 bindings.

The implementation is based on [this guide](https://graphallthethings.com/posts/streaming-ans-explained/).

## Installation

First, install the required dependencies:

```bash
pip install pybind11 numpy
```

Then install the package:

```bash
pip install .
```

Or install from source:

```bash
cd simple_ans
pip install -e .
```

## Usage

This package is designed for compressing quantized numerical data.

```python
import numpy as np
from simple_ans import ans_encode, ans_decode

# Example: Compressing quantized Gaussian data
# Generate sample data following normal distribution
n_samples = 10000
# Generate Gaussian data, scale by 4, and quantize to integers
signal = np.round(np.random.normal(0, 1, n_samples) * 4).astype(np.int32)

# Encode (automatically determines optimal symbol counts)
encoded = ans_encode(signal)

# Decode
decoded = ans_decode(encoded)

# Verify
assert np.all(decoded == signal)

# Get compression stats
original_size = signal.nbytes
compressed_size = encoded.size()  # in bits
compression_ratio = original_size / compressed_size
print(f"Compression ratio: {compression_ratio:.2f}x")
```

The package supports four integer types: `int16`, `uint16`, `int32`, and `uint32`.

## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute
