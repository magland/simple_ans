import numpy as np
import pytest
from simple_ans import (
    ans_encode,
    ans_decode,
    choose_symbol_counts,
    determine_symbol_counts_and_values,
    EncodedSignal,
)


def test_encode_decode():
    # Create a simple test signal
    signal = np.array([0, 1, 2, 1, 0], dtype=np.uint32)

    # Create symbol counts and values
    symbol_counts = np.array([3, 3, 2], dtype=np.uint32)  # For symbols 0,1,2
    symbol_values = np.array([0, 1, 2], dtype=np.int32)  # Corresponding values

    # Encode
    encoded = ans_encode(signal, symbol_counts, symbol_values)
    assert isinstance(encoded, EncodedSignal), "Result should be EncodedSignal object"
    assert isinstance(
        encoded.bitstream, bytes
    ), "Encoded bitstream should be bytes object"

    # Decode
    decoded = ans_decode(encoded)

    # Verify
    assert np.array_equal(signal, decoded), "Decoded signal does not match original"
    print("Test passed: encode/decode works correctly")


def test_choose_symbol_counts():
    # Test with some probabilities
    proportions = [0.5, 0.3, 0.2]
    L = 1024  # Should be power of 2

    counts = choose_symbol_counts(proportions, L)
    total = sum(counts)

    assert total == L, f"Total counts should sum to {L}"
    print("Test passed: choose_symbol_counts works correctly")


def test_determine_symbol_counts_and_values():
    # Test with default index length
    signal = [0, 1, 2, 1, 0]
    counts, values = determine_symbol_counts_and_values(signal)
    assert isinstance(counts, list), "Counts should be a list"
    assert isinstance(values, list), "Values should be a list"
    assert len(counts) == len(values), "Counts and values should have same length"
    assert values == [0, 1, 2], "Values should match unique signal values"
    assert sum(counts) == 2**16, "Total counts should sum to default index length"

    # Test with custom index length
    counts, values = determine_symbol_counts_and_values(signal, index_length=1024)
    assert sum(counts) == 1024, "Total counts should sum to specified index length"

    # Test error cases
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values([])  # Empty signal
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values(
            signal, index_length=1000
        )  # Non-power-of-2 index length
    with pytest.raises(ValueError):
        determine_symbol_counts_and_values(
            signal, index_length=-1
        )  # Negative index length


def test_auto_symbol_counts():
    print("Starting test_auto_symbol_counts")
    # Test encoding with auto-determined symbol counts
    signal = np.array(
        [0, 1, 2, 1, 0], dtype=np.int32
    )  # Changed to int32 to match new type
    print("Signal created")
    encoded = ans_encode(signal)  # No symbol_counts provided
    print("Signal encoded")
    decoded = ans_decode(encoded)
    print("Signal decoded")
    assert np.array_equal(signal, decoded), "Decoded signal does not match original"
    print("Test passed: auto symbol counts works correctly")


if __name__ == "__main__":
    test_encode_decode()
    test_choose_symbol_counts()
    test_determine_symbol_counts_and_values()
    test_auto_symbol_counts()
