from ._simple_ans import encode as _encode, decode as _decode, choose_symbol_counts, add_one_test
from dataclasses import dataclass
import numpy as np

__version__ = "0.2.0"


@dataclass
class EncodedSignal:
    """Container for ANS-encoded signal data.

    Attributes:
        state (int): Integer representing the final encoder state
        bitstream (numpy.ndarray): uint64 numpy array with encoded bits
        num_bits (int): Number of bits used in the encoding (may be somewhat less than len(bitstream) * 64)
        symbol_counts (numpy.ndarray): uint32 numpy array containing frequency counts for each symbol
        symbol_values (numpy.ndarray): int32 numpy array containing the actual symbol values
        signal_length (int): Length of the original signal in number of elements
    """

    state: int
    bitstream: np.ndarray  # uint64 array
    num_bits: int
    symbol_counts: np.ndarray  # uint32 array
    symbol_values: np.ndarray  # int32 array
    signal_length: int

    def __post_init__(self):
        """Validate and convert data types after initialization."""
        # Convert lists to numpy arrays if needed
        if not isinstance(self.symbol_counts, np.ndarray):
            self.symbol_counts = np.array(self.symbol_counts, dtype=np.uint32)
        if not isinstance(self.symbol_values, np.ndarray):
            self.symbol_values = np.array(self.symbol_values, dtype=np.int32)
        if not isinstance(self.bitstream, np.ndarray):
            self.bitstream = np.array(self.bitstream, dtype=np.uint64)

        # Validate types and sizes
        if not isinstance(self.state, int):
            raise TypeError("state must be an integer")
        if not isinstance(self.num_bits, int):
            raise TypeError("num_bits must be an integer")
        if not isinstance(self.signal_length, int):
            raise TypeError("signal_length must be an integer")

        assert self.symbol_counts.size == self.symbol_values.size, "symbol_counts and symbol_values must have the same size"
        assert self.symbol_counts.dtype == np.uint32, "symbol_counts must be uint32"
        assert self.symbol_values.dtype == np.int32, "symbol_values must be int32"
        assert self.bitstream.dtype == np.uint64, "bitstream must be uint64"


def determine_symbol_counts_and_values(signal: np.ndarray | list, index_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Determine symbol counts and unique values from input data.

    Args:
        signal: List or numpy array of integers representing the signal
        index_length: Length of the ANS index (must be a power of 2). If None, automatically
            determines minimum valid power of 2 based on number of unique symbols.

    Returns:
        tuple: A pair of numpy arrays (symbol_counts, symbol_values) where:
            - symbol_counts is a uint32 array of frequency counts for each symbol
            - symbol_values is an int32 array of the corresponding unique symbol values
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")

    import numpy as np

    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=np.int32)

    assert signal.dtype == np.int32

    # Get unique values and count frequencies using numpy
    unique_values, counts = np.unique(signal, return_counts=True)
    total = counts.sum()
    num_symbols = len(unique_values)

    def is_power_of_two(n: int) -> bool:
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    if index_length is None:
        # Start at 2^10 and increase until we have enough space for all symbols
        index_length = 2**10
        while index_length < num_symbols:
            index_length *= 2
    elif not isinstance(index_length, int) or index_length <= 0:
        raise ValueError("Index length must be a positive integer")
    elif not is_power_of_two(index_length):
        raise ValueError("Index length must be a power of 2")
    if index_length < num_symbols:
        raise ValueError(f"Index length ({index_length}) must be greater than or equal to the number of unique symbols ({num_symbols})")

    # Convert to proportions as numpy array
    proportions = counts.astype(np.float64) / total

    # Use existing choose_symbol_counts to convert proportions to integer counts
    symbol_counts = choose_symbol_counts(proportions, index_length)

    return symbol_counts, unique_values.astype(np.int32)


def ans_encode(signal: np.ndarray | list,
               symbol_counts: np.ndarray | list | None = None,
               symbol_values: np.ndarray | list | None = None) -> EncodedSignal:
    """Encode a signal using ANS (Asymmetric Numeral Systems).

    Args:
        signal: int32 numpy array of signal to encode
        symbol_counts: uint32 numpy array of symbol counts, defaults to None
        symbol_values: int32 numpy array of symbol values, defaults to None

    Returns:
        EncodedSignal: Object containing all encoding information
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=np.int32)
    if signal.dtype != np.int32:
        raise ValueError("Signal must be of type int32")

    # If either is None, determine both
    if symbol_counts is None:
        if symbol_values is not None:
            raise ValueError("If symbol_values is provided, symbol_counts must also be provided")
        auto_counts, auto_values = determine_symbol_counts_and_values(signal)
        symbol_counts = auto_counts
        symbol_values = auto_values
    if symbol_values is None:
        raise ValueError("If symbol_counts is provided, symbol_values must also be provided")

    # Ensure arrays are the right type
    if not isinstance(symbol_counts, np.ndarray):
        symbol_counts = np.array(symbol_counts, dtype=np.uint32)
    if not isinstance(symbol_values, np.ndarray):
        symbol_values = np.array(symbol_values, dtype=np.int32)
    if symbol_counts.dtype != np.uint32:
        raise ValueError("Symbol counts must be of type uint32")
    if symbol_values.dtype != np.int32:
        raise ValueError("Symbol values must be of type int32")

    encoded = _encode(signal, symbol_counts, symbol_values)
    return EncodedSignal(
        state=encoded.state,
        bitstream=encoded.bitstream,  # uint64 array from C++
        num_bits=encoded.num_bits,
        symbol_counts=symbol_counts,  # Already numpy array from above
        symbol_values=symbol_values,  # Already numpy array from above
        signal_length=len(signal),
    )


def ans_decode(encoded: EncodedSignal) -> np.ndarray:
    """Decode an ANS-encoded signal.

    Args:
        encoded: EncodedSignal object containing the encoded data and metadata

    Returns:
        decoded_signal: int32 numpy array of the decoded signal
    """
    # bitstream is already a list of uint64
    return _decode(
        encoded.state,
        encoded.bitstream,
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
    "add_one_test",
]
