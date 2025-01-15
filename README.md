# simple_ans

A Python package for Asymmetric Numeral Systems (ANS) encoding/decoding using a C++ implementation with pybind11.

The algorithm is based on [this guide](https://graphallthethings.com/posts/streaming-ans-explained/).

## Installation

First, install the required dependencies:

```bash
pip install pybind11 numpy
```

Then install the package:

```bash
pip install .
```

## Usage

```python
import numpy as np
from simple_ans import ans_encode, ans_decode

# Create a signal to encode (supports int32, int16, uint32, uint16)
signal = np.array([0, 1, 2, 1, 0], dtype=np.int32)  # or np.int16, np.uint32, np.uint16

# You can also specify the dtype explicitly
signal = np.array([0, 1, 2, 1, 0])
encoded = ans_encode(signal, dtype=np.uint32)

# Encode (automatically determines optimal symbol counts)
encoded = ans_encode(signal)

# Decode
decoded = ans_decode(encoded)

# Verify
assert np.all(decoded == signal)

# Advanced usage with manual symbol counts
signal = np.array([0, 1, 2, 1, 0], dtype=np.int32)
symbol_counts = np.array([3, 3, 2], dtype=np.uint32)  # Must sum to power of 2
symbol_values = np.array([0, 1, 2], dtype=np.int32)  # Must match signal dtype
encoded = ans_encode(signal, symbol_counts, symbol_values)
decoded = ans_decode(encoded)
assert np.all(decoded == signal)
```

## Supported Types

The package supports the following NumPy data types for signals:
- `np.int32`: 32-bit signed integers
- `np.int16`: 16-bit signed integers
- `np.uint32`: 32-bit unsigned integers
- `np.uint16`: 16-bit unsigned integers

Symbol counts are always `np.uint32`, and the bitstream is always `np.uint64`.
