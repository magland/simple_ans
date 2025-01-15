from ._simple_ans import encode as _encode, decode as _decode, choose_symbol_counts
from dataclasses import dataclass
from collections import Counter

__version__ = "0.2.0"


@dataclass
class EncodedSignal:
    """Container for ANS-encoded signal data.

    Attributes:
        state: Integer representing the final encoder state
        bitstream: List of 64-bit integers containing the encoded bitstream
        num_bits: Number of bits used in the encoding (may be less than bitstream.size() * 64)
        symbol_counts: Integer list of symbol counts
        signal_length: Length of the original signal
    """

    state: int
    bitstream: list
    num_bits: int
    symbol_counts: list
    symbol_values: list
    signal_length: int


def determine_symbol_counts_and_values(signal, index_length=None):
    """Determine symbol counts from input data.

    Args:
        signal: List of integers representing the signal
        index_length: Length of the ANS index (must be power of 2), defaults to 2^16

    Returns:
        counts: integer list of symbol counts
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")

    # Convert signal to a list
    import numpy as np

    if isinstance(signal, np.ndarray):
        # make sure type is integer
        if not np.issubdtype(signal.dtype, np.integer):
            raise ValueError("Signal must be of integer type")
        signal = signal.tolist()

    if index_length is None:
        index_length = 2**16
    elif not isinstance(index_length, int) or index_length <= 0:
        raise ValueError("Index length must be a positive integer")
    elif not (index_length & (index_length - 1) == 0):  # Check if power of 2
        raise ValueError("Index length must be a power of 2")

    # Get unique values and count frequencies
    counts = Counter(signal)
    unique_values = sorted(counts.keys())
    total = sum(counts.values())

    # Convert to proportions
    proportions = [counts.get(val, 0) / total for val in unique_values]

    # Use existing choose_symbol_counts to convert proportions to integer counts
    symbol_counts = choose_symbol_counts(proportions, index_length)

    return symbol_counts, unique_values


def ans_encode(signal, symbol_counts=None, symbol_values=None):
    """Encode a signal using ANS (Asymmetric Numeral Systems).

    Args:
        signal: List of integers representing the signal to encode
        symbol_counts: List of integer symbol counts, defaults to None
        symbol_values: List of integer symbol values, defaults to None

    Returns:
        EncodedSignal: Object containing all encoding information
    """
    # If either is None, determine both
    if symbol_counts is None or symbol_values is None:
        auto_counts, auto_values = determine_symbol_counts_and_values(signal)
        symbol_counts = auto_counts if symbol_counts is None else symbol_counts
        symbol_values = auto_values if symbol_values is None else symbol_values

    # make sure signal is a list of integers
    import numpy as np

    if isinstance(signal, np.ndarray):
        # make sure type is integer
        if not np.issubdtype(signal.dtype, np.integer):
            raise ValueError("Signal must be of integer type")
        signal = signal.tolist()

    encoded = _encode(signal, symbol_counts, symbol_values)
    return EncodedSignal(
        state=encoded.state,
        bitstream=list(encoded.bitstream),  # Already uint64 array from C++
        num_bits=encoded.num_bits,
        symbol_counts=symbol_counts,
        symbol_values=symbol_values,
        signal_length=len(signal),
    )


def ans_decode(encoded):
    """Decode an ANS-encoded signal.

    Args:
        encoded: EncodedSignal object containing the encoded data and metadata

    Returns:
        list: Decoded signal as a list of integers
    """
    # bitstream is already a list of uint64
    bitstream_list = encoded.bitstream
    return _decode(
        encoded.state,
        bitstream_list,
        encoded.num_bits,
        encoded.symbol_counts,
        encoded.symbol_values,
        encoded.signal_length,
    )


__all__ = [
    "ans_encode",
    "ans_decode",
    "choose_symbol_counts",
    "determine_symbol_counts_and_values",
    "EncodedSignal",
]
