# Asymmetric Numeral Systems (ANS) - Technical Overview

## 1. Introduction and Basic Concept

The core idea of ANS is to encode a sequence of symbols into a single integer state, with each symbol affecting the state in a way that reflects its probability of occurrence. This encoding is reversible, allowing us to recover the original sequence.

Consider a sequence of symbols $(s_1, s_2, ..., s_k)$ where:
- Each $s_i \in \{0, 1, ..., S-1\}$
- Symbols occur with relative frequencies $f_0, ..., f_{S-1}$ (positive integers)
- $\sum_{i=0}^{S-1} f_i = L = 2^l$ - We assume this is a power of two for reasons of efficiency.

## 2. Basic ANS Encoding/Decoding

### The Symbol Table Concept
To encode our symbols efficiently, we need a way to map between states and symbols that reflects their frequencies. We accomplish this using an infinite table $T$ that maps natural numbers to symbols: $T: \mathbb{N} \to \{0,...,S-1\}$

We construct this table with a periodic pattern where:
- Each period of length $L$ consists of consecutive blocks:
  * $f_0$ occurrences of symbol 0
  * $f_1$ occurrences of symbol 1
  * ...
  * $f_{S-1}$ occurrences of symbol $S-1$
- This pattern repeats every $L$ positions
- For any position $n$, $T(n) = T(n$ mod $L)$
- Define $C(s) = \sum_{i=0}^{s-1} f_i$ as the cumulative frequency (start position of symbol $s$ in each period)

### The Encoding/Decoding Process

#### Encoding Intuition
The encoding process works by maintaining a state value $x$ that evolves as we process each symbol. Starting from state $x_0$:

1. To encode symbol $s_1$:
   - Find the $x_0$-th occurrence of symbol $s_1$ (counting from 0)
   - This position becomes our new state $x_1$
2. Repeat this process for each symbol, using the previous state to find the next one

This process effectively "pushes" each symbol onto our state value in a way that can be reversed.

With the periodic structured table, the formula for this is

$$x_{i+1} = (x_i // f_{s_i}) \cdot L + C(s_i) + (x_i mod f_{s_i})$$

#### Decoding Intuition
Given a final state $x_k$, we can recover the original sequence:

1. The last symbol $s_k$ is simply $T[x_k]$ (the symbol at position $x_k$)
2. To get the previous state $x_{k-1}$:
   - Count how many times $s_k$ appeared before position $x_k$
   - This count is our previous state $x_{k-1}$
3. Continue this process to recover all symbols in reverse order

The formula for this is

$$x_i = (x_{i+1} // L) \cdot f_{s_i} + (x_{i+1} mod L) - C[s_i]$$

## 3. Streaming ANS

### The Need for Streaming
In practice, we can't work with arbitrarily large integers. Our state $x$ would grow indefinitely as we encode more symbols. We need a way to keep the state within a manageable range while preserving all information.

### The Solution: State Normalization
We'll maintain our state $x$ within the interval $[L, 2L)$ by by streaming out some bits to a separate bitstream at each iteration. We need this process to be reversible.

Let B be a sequence of bits (bit stream), initialized to the empty sequence. We'll encode symbol $s$ with state $x$ as follows:

#### Encoding $(x,B) \to (x',B')$
When encoding symbol $s$ with state $x$:
1. Find normalization factor $d$: the unique integer where $x // 2^d \in [f_s, 2f_s)$
2. Stream out the lower $d$ bits: $V = x$ mod $2^d$ to bitstream $B$
3. Use normalized value to compute next state:
   $$x' = ((x // 2^d) // f_s) \cdot L + C(s) + ((x // 2^d) mod f_s)$$

#### Decoding $(x',B') \to (x,B)$
To reverse the process:
1. Read current symbol: $s = T[x']$
2. Compute intermediate state: $x_2 = (x'$ // $L) \cdot f_s + (x'$ mod $L) - C[s]$
3. Find $d$: unique integer where $x_2 \cdot 2^d \in [L, 2L)$
4. Read $d$ bits ($V$) from the end of bitstream $B$
5. Reconstruct original state: $x = (x_2 \cdot 2^d) + V$

The beauty of this approach is that we can decode the exact number of bits needed at each step without requiring explicit markers in the bitstream.

## 5. Performance Considerations

### Efficient Operations
- Powers of 2 operations ($\cdot 2^d$, // $2^d$)
- Modulo with power of 2 (mod $2^d$)
- Bitstream operations

### Expensive Operations
- Integer division by $f_s$
- Finding normalization factor $d$
- Multiplication by $f_s$

### Optimization Opportunities
- Lookup table for $d$ values when $x_2 \in [0,L)$
- Precomputed $C(s)$ values
