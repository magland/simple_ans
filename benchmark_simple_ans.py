import time
import numpy as np
from simple_ans import ans_encode, ans_decode

# Generate random test data from normal distribution
n = 10_000_000
# Generate signal with normal distribution, ensuring positive values
signal = np.round(np.random.normal(0, 1, n) * 1).astype(np.int16)

# Calculate ideal compression ratio
vals, counts = np.unique(signal, return_counts=True)
probs = counts / len(signal)
ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))
print(f"Ideal compression ratio: {ideal_compression_ratio}")

timer = time.time()
encoded = ans_encode(signal=signal)  # Using auto-determined symbol counts
elapsed_encode = time.time() - timer

timer = time.time()
signal_decoded = ans_decode(encoded)
elapsed_decode = time.time() - timer

assert len(signal_decoded) == len(signal)
assert np.all(signal_decoded == signal)
print("Decoded signal matches original signal")

compressed_size_bits = (
    len(encoded.bitstream) * 64 + 32
)  # actual bits used + 32 bits for state
compression_ratio = (len(signal) * 16) / compressed_size_bits
print(f"Ideal compression ratio: {ideal_compression_ratio}")
print(f"simple_ans: Compression ratio: {compression_ratio}")
print(f"simple_ans: Pct of ideal compression: {compression_ratio/ideal_compression_ratio*100:.2f}%")
print("")
signal_bytes = len(signal) * 2
print(
    f"simple_ans: Time to encode: {elapsed_encode:.2f} seconds ({signal_bytes/elapsed_encode/1e6:.2f} MB/s)"
)
print(
    f"simple_ans: Time to decode: {elapsed_decode:.2f} seconds ({signal_bytes/elapsed_decode/1e6:.2f} MB/s)"
)
print("")

import zlib
timer = time.time()
buf_compressed = zlib.compress(np.array(signal, dtype=np.int16).tobytes(), level=6)
elapsed_zlib = time.time() - timer
zlib_compression_ratio = signal_bytes / len(buf_compressed)
print(f"Zlib (level 6) compression ratio: {zlib_compression_ratio:.2f}")
print(f'Zlib (level 6) pct of ideal compression: {zlib_compression_ratio/ideal_compression_ratio*100:.2f}%')
print(
    f"Time to zlib compress: {elapsed_zlib:.2f} seconds ({signal_bytes/elapsed_zlib/1e6:.2f} MB/s)"
)
timer = time.time()
signal_decompressed = np.frombuffer(zlib.decompress(buf_compressed), dtype=np.int16)
elapsed_zlib_decode = time.time() - timer
print(
    f"Time to zlib decompress: {elapsed_zlib_decode:.2f} seconds ({signal_bytes/elapsed_zlib_decode/1e6:.2f} MB/s)"
)
print("")

import zstandard as zstd
cctx = zstd.ZstdCompressor(level=13)
timer = time.time()
compressed = cctx.compress(np.array(signal, dtype=np.int16).tobytes())
elapsed_zstd = time.time() - timer
zstd_compression_ratio = signal_bytes / len(compressed)
print(f"Zstandard (level 13) compression ratio: {zstd_compression_ratio:.2f}")
print(f'Zstandard (level 13) pct of ideal compression: {zstd_compression_ratio/ideal_compression_ratio*100:.2f}%')
print(
    f"Time to zstd compress: {elapsed_zstd:.2f} seconds ({signal_bytes/elapsed_zstd/1e6:.2f} MB/s)"
)
dctx = zstd.ZstdDecompressor()
timer = time.time()
signal_decompressed = np.frombuffer(dctx.decompress(compressed), dtype=np.int16)
elapsed_zstd_decode = time.time() - timer
print(
    f"Time to zstd decompress: {elapsed_zstd_decode:.2f} seconds ({signal_bytes/elapsed_zstd_decode/1e6:.2f} MB/s)"
)
print("")

import lzma
timer = time.time()
compressed = lzma.compress(np.array(signal, dtype=np.int16).tobytes(), preset=3)
elapsed_lzma = time.time() - timer
lzma_compression_ratio = signal_bytes / len(compressed)
print(f"LZMA compression ratio: {lzma_compression_ratio:.2f}")
print(f'LZMA pct of ideal compression: {lzma_compression_ratio/ideal_compression_ratio*100:.2f}%')
print(
    f"Time to lzma compress: {elapsed_lzma:.2f} seconds ({signal_bytes/elapsed_lzma/1e6:.2f} MB/s)"
)
timer = time.time()
signal_decompressed = np.frombuffer(lzma.decompress(compressed), dtype=np.int16)
elapsed_lzma_decode = time.time() - timer
print(
    f"Time to lzma decompress: {elapsed_lzma_decode:.2f} seconds ({signal_bytes/elapsed_lzma_decode/1e6:.2f} MB/s)"
)

print("\nTesting data transfer overhead:")
from simple_ans import add_one_test
timer = time.time()
signal_plus_one = add_one_test(signal)
elapsed_add_one = time.time() - timer

# Verify result
expected = signal + 1
assert np.all(signal_plus_one == expected)
print("Add one test result verified correct")

print(
    f"Time for add_one_test: {elapsed_add_one:.2f} seconds ({signal_bytes/elapsed_add_one/1e6:.2f} MB/s)"
)
