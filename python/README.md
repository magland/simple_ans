# simple_ans

A Python package for Asymmetric Numeral Systems (ANS) encoding/decoding using a simple C++ implementation.

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
from simple_ans import encode, decode, choose_symbol_counts

# Create a signal to encode
signal = np.array([0, 1, 2, 1, 0], dtype=np.uint32)

# Define symbol frequencies
symbol_counts = np.array([2, 2, 1], dtype=np.uint32)  # For symbols 0,1,2

# Encode
encoded = encode(signal, symbol_counts)

# Decode
decoded = decode(encoded.state, encoded.bits, encoded.num_bits,
                symbol_counts, len(signal))

# Helper function to convert probabilities to counts
proportions = [0.5, 0.3, 0.2]
L = 1024  # Should be power of 2
counts = choose_symbol_counts(proportions, L)
```

## Testing

Run the tests:

```bash
python test_simple_ans.py
```

## License

MIT License
