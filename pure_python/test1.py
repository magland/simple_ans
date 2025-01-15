import time
import numpy as np
from ans_encode_decode import ans_encode, ans_decode, choose_symbol_counts

aa = np.round(np.random.normal(0, 1, 1000) * 5).astype(np.int16)
vals, counts = np.unique(aa, return_counts=True)
counts = np.sort(counts)

num_symbols = len(counts)
probs = np.array(counts) / np.sum(counts)

ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))

symbol_counts = choose_symbol_counts(proportions=probs, L=2**16)

ideal_compression_ratio = 16 / -np.sum(probs * np.log2(probs))

n = 1_000_000
signal = np.random.choice(num_symbols, n, p=probs).tolist()
print(f"Signal length: {len(signal)}")

timer = time.time()
signal_encoded, bitstream = ans_encode(signal=signal, symbol_counts=symbol_counts)
elapsed_encode = time.time() - timer

timer = time.time()
signal_decoded = ans_decode(
    state=signal_encoded, bitstream=bitstream, n=n, symbol_counts=symbol_counts
)
elapsed_decode = time.time() - timer

assert len(signal_decoded) == len(signal)
assert np.all(signal_decoded == signal)
print("Decoded signal matches original signal")

compressed_size_bits = len(bitstream)
compression_ratio = (len(signal) * 16) / compressed_size_bits
print(f"Compression ratio: {compression_ratio}")
print(f"Ideal compression ratio: {ideal_compression_ratio}")
print(f"Pct of ideal compression: {compression_ratio/ideal_compression_ratio*100:.2f}%")
print("")
signal_bytes = len(signal) * 2
print(
    f"Time to encode: {elapsed_encode:.2f} seconds ({signal_bytes/elapsed_encode/1e6:.2f} MB/s)"
)
print(
    f"Time to decode: {elapsed_decode:.2f} seconds ({signal_bytes/elapsed_decode/1e6:.2f} MB/s)"
)

# import zlib
# timer = time.time()
# buf_compressed = zlib.compress(np.array(signal, dtype=np.int16).tobytes())
# elapsed_zlib = time.time() - timer
# print(f"Zlib compression ratio: {signal_bytes/len(buf_compressed):.2f}")
# print(f"Time to zlib compress: {elapsed_zlib:.2f} seconds ({signal_bytes/elapsed_zlib/1e6:.2f} MB/s)")
