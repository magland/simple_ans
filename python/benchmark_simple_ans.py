import time
import numpy as np
from simple_ans import ans_encode, ans_decode

# Generate random test data from normal distribution
n = 10_000_000
# Generate signal with normal distribution, ensuring positive values
signal = np.round(np.random.normal(0, 1, n) * 1).astype(np.int32)
signal = (signal - signal.min()).astype(np.uint32)

# Calculate ideal compression ratio
vals, counts = np.unique(signal, return_counts=True)
probs = counts / len(signal)
ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))
print(f'Ideal compression ratio: {ideal_compression_ratio}')

timer = time.time()
encoded = ans_encode(signal=signal)  # Using auto-determined symbol counts
elapsed_encode = time.time() - timer

timer = time.time()
signal_decoded = ans_decode(encoded)
elapsed_decode = time.time() - timer

assert len(signal_decoded) == len(signal)
assert np.all(signal_decoded == signal)
print("Decoded signal matches original signal")

compressed_size_bits = encoded.num_bits + 32  # actual bits used + 32 bits for state
compression_ratio = (len(signal) * 16) / compressed_size_bits
print(f"Compression ratio: {compression_ratio}")
print(f'Ideal compression ratio: {ideal_compression_ratio}')
print(f"Pct of ideal compression: {compression_ratio/ideal_compression_ratio*100:.2f}%")
print("")
signal_bytes = len(signal) * 2
print(f"Time to encode: {elapsed_encode:.2f} seconds ({signal_bytes/elapsed_encode/1e6:.2f} MB/s)")
print(f"Time to decode: {elapsed_decode:.2f} seconds ({signal_bytes/elapsed_decode/1e6:.2f} MB/s)")

# import zlib
# timer = time.time()
# buf_compressed = zlib.compress(np.array(signal, dtype=np.int16).tobytes())
# elapsed_zlib = time.time() - timer
# print(f"Zlib compression ratio: {signal_bytes/len(buf_compressed):.2f}")
# print(f"Time to zlib compress: {elapsed_zlib:.2f} seconds ({signal_bytes/elapsed_zlib/1e6:.2f} MB/s)")
