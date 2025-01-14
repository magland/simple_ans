import numpy as np
import pytest
from simple_ans import ans_encode, ans_decode, choose_symbol_counts, determine_symbol_counts, EncodedSignal

def test_encode_decode():
    # Create a simple test signal
    signal = np.array([0, 1, 2, 1, 0], dtype=np.uint32)

    # Create symbol counts (frequencies)
    symbol_counts = np.array([3, 3, 2], dtype=np.uint32)  # For symbols 0,1,2

    # Encode
    encoded = ans_encode(signal, symbol_counts)
    assert isinstance(encoded, EncodedSignal), "Result should be EncodedSignal object"
    assert isinstance(encoded.bitstream, bytes), "Encoded bitstream should be bytes object"

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

def test_determine_symbol_counts():
    # Test with default index length
    signal = [0, 1, 2, 1, 0]
    counts = determine_symbol_counts(signal)
    assert isinstance(counts, list), "Result should be a list"
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0, "All symbols should have counts"
    assert sum(counts) == 2**16, "Total counts should sum to default index length"

    # Test with custom index length
    counts = determine_symbol_counts(signal, index_length=1024)
    assert sum(counts) == 1024, "Total counts should sum to specified index length"

    # Test error cases
    with pytest.raises(ValueError):
        determine_symbol_counts([])  # Empty signal
    with pytest.raises(ValueError):
        determine_symbol_counts(signal, index_length=1000)  # Non-power-of-2 index length
    with pytest.raises(ValueError):
        determine_symbol_counts(signal, index_length=-1)  # Negative index length

def test_auto_symbol_counts():
    # Test encoding with auto-determined symbol counts
    signal = np.array([0, 1, 2, 1, 0], dtype=np.uint32)
    encoded = ans_encode(signal)  # No symbol_counts provided
    decoded = ans_decode(encoded)
    assert np.array_equal(signal, decoded), "Decoded signal does not match original"

if __name__ == "__main__":
    test_encode_decode()
    test_choose_symbol_counts()
    test_determine_symbol_counts()
    test_auto_symbol_counts()
