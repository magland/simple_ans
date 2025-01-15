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

# Create a signal to encode (uint32 array)
signal = np.array([0, 1, 2, 1, 0], dtype=np.uint32)

# Encode (automatically determines optimal symbol counts)
encoded = ans_encode(signal)

# Decode
decoded = ans_decode(encoded)

# Verify
assert np.all(decoded == signal)
```

