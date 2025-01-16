# Based on https://graphallthethings.com/posts/streaming-ans-explained/
import numpy as np
from typing import List


def ans_encode(*, signal: List[int], symbol_counts: List[int]):
    """
    Performs streaming Asymmetric Numeral Systems (ANS) encoding of a signal.
    Processes the input signal sequentially, generating a compressed bitstream.

    Parameters
    ----------
    signal : List[int]
        The input signal to encode, represented as a list of symbol indices.
    symbol_counts : List[int]
        List containing the frequency/count of each symbol in the alphabet.
        The sum of counts must be a power of 2.

    Returns
    -------
    tuple
        A tuple containing:
        - state (int): The final encoder state
        - bitstream (List[int]): The encoded bitstream as a list of 0s and 1s

    Raises
    ------
    ValueError
        If the sum of symbol_counts is not a power of 2
        If the encoder state becomes invalid during encoding
    """
    L = sum(symbol_counts)
    # check that L is a power of 2
    if L & (L - 1) != 0:
        raise ValueError(f"L={L} is not a power of 2")
    symbol_counts_cumsum = [0]
    for c in symbol_counts:
        symbol_counts_cumsum.append(symbol_counts_cumsum[-1] + c)
    state = L
    # state will stay between L and 2L
    bitstream = []
    for ii in range(len(signal)):
        s = signal[ii]
        state_normalized = state
        L_s = symbol_counts[s]
        # we need state_normalized to be in the range [L_s, 2L_s)
        while state_normalized >= 2 * L_s:
            bitstream.append(state_normalized % 2)
            state_normalized = state_normalized // 2
        # now find where the state_normalized^th instance of s occurs in the index
        # state = (state_normalized // symbol_counts[s]) * L + symbol_counts_cumsum[s] + (state_normalized % symbol_counts[s])
        state = L + symbol_counts_cumsum[s] + state_normalized - L_s
        if state < L:
            raise ValueError(f"State is < L {L}: {state}")
        if state >= 2 * L:
            raise ValueError(f"State is >= 2L {2*L}: {state}")
    return state, bitstream


def ans_decode(*, state: int, bitstream: List[int], symbol_counts: List[int], n: int):
    """
    Performs streaming Asymmetric Numeral Systems (ANS) decoding of an encoded signal.
    Processes the compressed data sequentially to recover the original signal.

    Parameters
    ----------
    state : int
        The final encoder state from the encoding process
    bitstream : List[int]
        The encoded bitstream as a list of 0s and 1s
    symbol_counts : List[int]
        List containing the frequency/count of each symbol in the alphabet.
        Must match the counts used during encoding.
    n : int
        The length of the original signal to decode

    Returns
    -------
    List[int]
        The decoded signal as a list of symbol indices

    Raises
    ------
    ValueError
        If the sum of symbol_counts is not a power of 2
        If the decoder state becomes invalid during decoding
        If the lookup value is not found in the cumulative sum table
    """
    L = sum(symbol_counts)
    # check that L is a power of 2
    if L & (L - 1) != 0:
        raise ValueError(f"L={L} is not a power of 2")
    if state < L:
        raise ValueError(f"State is < L {L}: {state}")
    if state >= 2 * L:
        raise ValueError(f"State is >= 2L {2*L}: {state}")
    symbol_counts_cumsum = [0]
    for c in symbol_counts:
        symbol_counts_cumsum.append(symbol_counts_cumsum[-1] + c)

    def symbol_counts_cumsum_lookup(y):
        for i, _s in enumerate(symbol_counts_cumsum):
            if y < _s:
                return i - 1
        raise ValueError(f"y={y} is not less than any of the cumsum values")

    signal = []
    bitstream_idx = len(bitstream) - 1
    for _ in range(n):
        # s = index[state % L]
        s = symbol_counts_cumsum_lookup(state % L)
        # state_normalized is the number of times s has occurred in the signal up to (but not including) state
        # state_normalized = (state // L) * symbol_counts[s] + (state % L - symbol_counts_cumsum[s])
        state_normalized = symbol_counts[s] + state - L - symbol_counts_cumsum[s]
        L_s = symbol_counts[s]
        if state_normalized < L_s:
            raise ValueError(f"State_normalized is < L_s {L_s}: {state_normalized}")
        if state_normalized >= 2 * L_s:
            raise ValueError(f"State_normalized is >= 2L_s {2*L_s}: {state_normalized}")
        state = state_normalized
        # we need state to be between L and 2L by using the bitstream
        while state < L:
            state = state * 2 + bitstream[bitstream_idx]
            bitstream_idx -= 1
        if state >= 2 * L:
            raise ValueError(f"State is >= 2L {2*L}: {state}")
        signal.append(s)
    # reverse the signal
    signal = signal[::-1]
    return signal


def choose_symbol_counts(proportions, L):
    """
    Convert real-valued proportions into integer counts summing to L,
    ensuring each count is at least 1.

    Parameters
    ----------
    proportions : list of float
        The target proportions [p1, p2, ..., pk].
        Ideally sum(proportions) = 1.0, but we re-normalize if needed.
    L : int
        Total number of items to be distributed among k categories.

    Returns
    -------
    counts : list of int
        The list [c1, c2, ..., ck] of integer counts that sum to L
        and reflect the proportions as closely as possible, with c_i >= 1.
    """
    if len(proportions) > L:
        raise ValueError("More proportions than items to distribute")

    # normalize the proportions to sum to 1
    proportions = [p / sum(proportions) for p in proportions]

    # first give everyone one to start
    counts = [1] * len(proportions)

    # real-valued target counts
    target_counts = [p * L for p in proportions]

    while sum(counts) < L:
        residuals = [t - c for t, c in zip(target_counts, counts)]
        residuals_int_part = [int(r) for r in residuals]
        residuals_frac_part = [r - int(r) for r in residuals]
        # if any of the integer parts are positive, then let's distribute those
        if any([r > 0 for r in residuals_int_part]):
            for i in range(len(counts)):
                if residuals_int_part[i] > 0:
                    counts[i] += residuals_int_part[i]
        else:
            # otherwise, let's give one to each in order of largest fractional part
            idx = np.argsort(residuals_frac_part)[::-1]
            for i in idx:
                counts[i] += 1
                if sum(counts) == L:
                    break
    return counts


if __name__ == "__main__":
    aa = np.round(np.random.normal(0, 1, 1000) * 5).astype(np.int16)
    vals, counts = np.unique(aa, return_counts=True)
    counts = np.sort(counts)

    num_symbols = len(counts)
    probs = np.array(counts) / np.sum(counts)

    ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))

    symbol_counts = choose_symbol_counts(proportions=probs, L=2**16)

    n = 10000
    signal = np.random.choice(num_symbols, n, p=probs).tolist()
    print(f"Signal length: {len(signal)}")
    signal_encoded, bitstream = ans_encode(signal=signal, symbol_counts=symbol_counts)
    signal_decoded = ans_decode(
        state=signal_encoded, bitstream=bitstream, n=n, symbol_counts=symbol_counts
    )
    assert len(signal_decoded) == len(signal)
    assert np.all(signal_decoded == signal)
    print("Decoded signal matches original signal")

    compressed_size_bits = len(bitstream)
    compression_ratio = (len(signal) * 16) / compressed_size_bits
    print(f"Compression ratio: {compression_ratio}")
    print(f"Ideal compression ratio: {ideal_compression_ratio}")
    print(
        f"Pct of ideal compression: {compression_ratio/ideal_compression_ratio*100:.2f}%"
    )
