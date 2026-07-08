# Parallel Neural Network Engine from Scratch in C

A zero-dependency, cache-coherent, multi-threaded neural network engine written in pure C99.

Trains on the full 60,000-image MNIST dataset and achieves **91.84% test accuracy** in **2.7 seconds** on an 8-core Apple Silicon CPU.

![Architecture Diagram](./images/cnn%20illustration.png)

---

## Performance

The engine is optimized to exploit CPU cache structures and hardware vector registers. It is **100x faster** than the traditional single-threaded baseline under identical hyperparameters (5 epochs, batch size 32, learning rate 0.1):

| Version | Source Files | Architecture | Training Time | Speedup |
|---|---|---|---|---|
| **V1 (Baseline)** | `mnist_digit_cnn.c` | Hardcoded pointer-chased `double**` | 270.0 s | 1x |
| **V2 (Optimized)** | `main.c` (+ modular files) | Fully dynamic cache-coherent `double*` | **2.7 s** | **100x** |

### Where the 100x Speedup Comes From

* **Memory Layout**: Restructured scattered double-pointer matrices (`double**`) into a single flat contiguous `double*` array, matching CPU cache line prefetches and eliminating pointer chasing.
* **SIMD Auto-Vectorization**: Aligned sequential memory accesses to enable the compiler to auto-vectorize inner loops using hardware SIMD registers (NEON on ARM/Apple Silicon, AVX on x86).
* **Parallelism**: Split mini-batch execution across all available hardware cores dynamically using POSIX Threads (`pthreads`).
* **Lock-Free Concurrency**: Designed thread-local gradient buffers to allow independent forward/backward passes with zero mutex lock contention.
* **Complexity Reduction**: Derived recursive analytical gradients, reducing backward pass complexity from $O(N^3)$ to $O(N^2)$.

---

## Systems Architecture

### Flat 1D Memory Layout
All network biases, activations, and weights are stored in flat, contiguous memory segments. Layer offsets are managed via precomputed prefix sums (`p_sums`) and weight index offset tables (`w_indices`). This sequential layout ensures that adjacent elements are loaded into L1/L2 caches in a single cache line fetch.

```
biases:   [ --- L0 (784, unused) --- | --- L1 --- | --- L2 --- | ... | --- Ln --- ]
                                       ^                         ^
                                  p_sums[1]                 p_sums[n]

weights:  [ --- L0→L1 --- | --- L1→L2 --- | ... | --- L(n-1)→Ln --- ]
                ^                                       ^
           w_indices[0]                            w_indices[n-1]
```

### SIMD Auto-Vectorization
Compiling with `-O3` and `-Rpass=loop-vectorize` verifies that the contiguous memory layout allows the compiler to auto-vectorize critical execution loops:
* **Feedforward dot-products** are vectorized (width 2, interleave 4).
* **Activation loading** is vectorized (width 16, interleave 1).
* **Gradient accumulation** is vectorized (width 2, interleave 4).

### Dynamic Multi-Threading Model
The engine implements a high-throughput, fork-join parallel execution model using POSIX Threads (`pthreads`). Concurrency is designed to eliminate synchronization overhead and maximize hardware utilization through the following pipeline:

1. **Dynamic Core-to-Thread Mapping**: 
   At runtime, the program queries the operating system for online CPU cores (using `sysconf(_SC_NPROCESSORS_ONLN)` on UNIX/macOS and `GetSystemInfo` on Windows). It dynamically spawns exactly $T = \min(\text{available\_cores}, \text{batch\_size})$ worker threads to adapt to the host hardware.

2. **Workload Partitioning**: 
   The mini-batch of size $B$ is partitioned across the $T$ threads. Each thread is assigned a workload of $\lfloor B / T \rfloor$ images, with any remainder $B \pmod T$ allocated to the final thread. This prevents load-imbalance bottlenecks while ensuring all images in the mini-batch are processed.

3. **Lock-Free Concurrency (Thread-Local Accumulators)**:
   * **Read-Only Shared State**: During the forward pass, all worker threads read from the shared global network weights and biases. Because the network parameters are immutable during this phase, no locks are required.
   * **Isolated Write State**: During backpropagation, threads write exclusively to their own **thread-local gradient buffers** (`dL_dw`, `dL_db`). This design achieves **zero race conditions and zero mutex/lock contention** during the computationally heavy backward pass.

4. **Barrier Synchronization & Gradient Aggregation**:
   The main thread acts as the orchestrator. After spawning the workers via `pthread_create`, it performs a barrier synchronization by calling `pthread_join` on all threads. Once all threads terminate, the main thread runs a single-threaded loop to aggregate the local gradient buffers, normalize them by the batch size, and apply the Stochastic Gradient Descent (SGD) update step.

---

## Algorithmic Optimizations

* **Recursive Backpropagation**: Propagates error signals ($\delta$) recursively backward through hidden layers, avoiding redundant calculations of the forward chain.
* **Fused Softmax-Cross-Entropy Gradient**: Combines output activation and loss functions analytically, simplifying the output derivative to a single subtraction ($\mathcal{L}' = a_i - y_i$). This removes expensive division, logarithm, and exponent operations from the backpropagation path.

---

## File Structure

* `main.c` — Program entry, dynamic OS core detection, thread orchestration, and training loops.
* `mlp.c` / `mlp.h` — Allocations, flat memory layouts, parameter initializations, and cleanup.
* `thread_handler.c` / `thread_handler.h` — Worker thread execution, forward passes, backpropagation.
* `file_handler.c` / `file_handler.h` — High-speed big-endian IDX file parsing for MNIST binaries.

---

## Build & Run

### Prerequisites
Download the raw MNIST IDX files from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and place them in a `mnist/` folder:
* `train-images-idx3-ubyte` / `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte` / `t10k-labels-idx1-ubyte`

### Compilation
Compile the optimized multi-threaded binary:
```bash
gcc -Wall -O3 main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

Verify SIMD auto-vectorization diagnostics:
```bash
gcc -Wall -O3 -Rpass=loop-vectorize main.c mlp.c file_handler.c thread_handler.c -o main -lm
```

### Run
```bash
./main
```
The program will prompt you to input the network topology (number of hidden layers and nodes), learning rate, batch size, and training epochs at runtime.
