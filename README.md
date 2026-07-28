# Parallel Neural Network Engine from Scratch in C

A zero-dependency, cache-coherent, multi-threaded neural network engine written in pure C99.

Trains on the full 60,000-image MNIST dataset and achieves **~93% test accuracy** in **~2.7 seconds** on an 8-core Apple Silicon CPU.

![Architecture Diagram](./images/cnn%20illustration.png)

---

## Performance Benchmark

The engine is built from first principles in pure C to exploit CPU cache structures, lock-free concurrency, and hardware vector registers. Version 3 achieves **93.51% accuracy** with **102x speedup** over the single-threaded V1 baseline under identical hyperparameters (5 epochs, batch size 32, learning rate 0.1):

| Version | Directory / Source | Key Systems Features | Training Time | Test Accuracy | Speedup |
|---|---|---|---|---|---|
| **V1 (Baseline)** | `mnist_digit_cnn.c` | Hardcoded pointer-chased `double**` | 270.0 s | 88.20% | 1x |
| **V2 (Optimized)** | `main.c` (+ modular files) | Flat `double*` layout, basic `pthreads` | 2.70 s | 91.84% | ~100x |
| **V3 (Production)** | `v3/` | **Fisher-Yates Shuffler, Xavier-Glorot Init, Strict Alloc Checks** | **2.65 s** | **93.51%** | **~102x** |

---

## What's New in V3

1. **Thread-Safe Epoch Mini-Batch Shuffler**: 
   - Replaced worker thread `rand()` calls with a **main-thread Fisher-Yates shuffle** on a master index tracking array (`indices`).
   - Eliminates RNG state data races completely, making thread execution 100% lock-free and deterministic.
   - Guarantees sampling *without* replacement—every image is processed exactly once per epoch.

2. **Xavier-Glorot Uniform Weight Initialization**:
   - Replaced uniform `[-0.5, 0.5]` initialization with variance-scaled **Xavier-Glorot uniform bounds**:
     $$d = \sqrt{\frac{6}{N_{\text{in}} + N_{\text{out}}}}$$
   - Prevents vanishing/exploding gradients and boosts initial accuracy from ~91.8% to **93.51%**.

3. **Strict Allocation & Concurrency Validation**:
   - Added explicit `NULL` validation for all `malloc` calls across network and buffer structures.
   - Added return-code verification for all `pthread_create` thread spawning calls.

---

## Low-Level Systems & Engineering Decisions

### 1. 1D Prefix Sum Index Arithmetic (`p_sums` & `w_indices`) to eliminate Overhead
Traditional neural network tutorials allocate layer weights using multi-dimensional dynamic pointers (`double***` or `double**`). This creates severe pointer-chasing overhead and fragment memory across the heap, causing L1/L2 cache misses on every forward/backward iteration.

Our engine allocates **all biases, activations, and weights in flat contiguous 1D memory blocks**. To dynamically index layers of arbitrary sizes without multi-dimensional pointers, we precompute **Prefix Sum Tables**:

* **`p_sums[l]`**: Stores the exact starting index offset for layer $l$'s activations and biases in the flat 1D array.
  $$\text{p\_sums}[l] = \sum_{k=0}^{l-1} \text{summary}[k]$$
* **`w_indices[l]`**: Stores the starting index offset for the weight matrix connecting layer $l \rightarrow l+1$.
  $$\text{w\_indices}[l] = \sum_{k=0}^{l-1} (\text{summary}[k] \times \text{summary}[k+1])$$

To access the weight connecting node $i$ in layer $l$ to node $j$ in layer $l+1$:
```c
int weight_idx = w_indices[l] + i * summary[l + 1] + j;
double w = weights[weight_idx];
```
This row-major 1D layout guarantees adjacent weights are fetched into 64-byte L1 CPU cache lines in a single instruction.

```
biases:   [ --- L0 (784, unused) --- | --- L1 --- | --- L2 --- | ... | --- Ln --- ]
                                       ^                         ^
                                  p_sums[1]                 p_sums[n]

weights:  [ --- L0→L1 --- | --- L1→L2 --- | ... | --- L(n-1)→Ln --- ]
                ^                                       ^
           w_indices[0]                            w_indices[n-1]
```

---

### 2. Cross-Platform Runtime Core Detection & Thread Saturation (`get_cores()`)
To prevent thread over-subscription (which causes expensive OS context switching) or thread under-utilization, the engine queries host hardware at startup:

* On **UNIX / macOS**: Uses `sysconf(_SC_NPROCESSORS_ONLN)` to detect active physical/logical cores.
* On **Windows**: Uses `GetSystemInfo(&sysinfo)` and extracts `dwNumberOfProcessors`.

```c
int avl_cores = get_cores();
int num_threads_ideal = (avl_cores > batch_size ? batch_size : avl_cores);
```
The thread count $T$ is dynamically clamped to $\min(\text{cores}, \text{batch\_size})$. This ensures every active thread is assigned a non-zero workload partition and maximizes hardware core utilization without thread thrashing.

---

### 3. Lock-Free Concurrency & Thread-Local Accumulators
Multithreaded backpropagation usually suffers from lock contention when multiple threads attempt to update shared weight gradients simultaneously.

We solved this by giving each worker thread its own **thread-local state buffer** (`state` struct):
* **Read-Only Forward Pass**: All worker threads concurrently read from the global `net->weights` array. Because weights are immutable during feedforward, no mutex locks are required.
* **Isolated Thread-Local Backprop**: During backpropagation, worker threads write exclusively to their private local gradient accumulators (`arg->dL_dw`, `arg->dL_db`).
* **Zero Mutex Overhead**: Threads run at 100% CPU speed with zero locks, zero mutex contention, and zero atomic pauses.

After thread barrier synchronization (`pthread_join`), the main thread aggregates the thread-local buffers into the global network weights in a single linear pass:
```c
for (int i = 0; i < network->total_weights; i++) {
    double temp = 0;
    for (int t = 0; t < num_threads_ideal; t++) {
        temp += buffer_arr[t].dL_dw[i];
    }
    network->weights[i] -= (learning_rate * temp) / batch_size;
}
```

---

### 4. Thread-Safe Epoch Mini-Batch Shuffler (Fisher-Yates)
Standard `rand()` calls inside worker threads create severe data races on the C library's global RNG state. 

In **V3**, we move randomness entirely to the main thread:
1. **Master Index Allocation**: The main thread creates an `int* indices` array `[0, 1, 2, ..., 59999]`.
2. **Epoch Shuffle**: Before starting each epoch, the main thread shuffles `indices` in $O(N)$ using the Fisher-Yates algorithm:
   ```c
   for (int i = num - 1; i > 0; i--) {
       int swap = abs(rand()) % (i + 1);
       int temp = index[i];
       index[i] = index[swap];
       index[swap] = temp;
   }
   ```
3. **Partition Slice Passing**: Worker threads receive their offset slice of `indices`. Each worker looks up image indices via `int img_idx = indices[arg->start + i];`.

This guarantees **zero RNG data races**, **deterministic execution**, and **true sampling without replacement**.

---

### 5. Fused Softmax + Cross-Entropy Analytical Derivative
Instead of calculating Softmax activations and Cross-Entropy loss gradients separately (which requires expensive logarithms and divisions), we derived the combined analytical derivative:

$$\frac{\partial \mathcal{L}}{\partial z_i} = a_i - y_i$$

In `thread_handler.c`:
```c
int y = (arg->labels[img_idx] == node) ? 1 : 0;
arg->dL_dz[p_sums[last_layer] + node] = arg->activation[p_sums[last_layer] + node] - y;
```
This analytical simplification replaces logarithmic and division chains with a single floating-point subtraction per output neuron!

---

### 6. Fast Big-Endian IDX Binary File Parser (`file_handler.c`)
The raw MNIST dataset is stored in Yann LeCun's IDX binary format, which uses **Big-Endian byte order**. Because modern x86 and ARM CPUs are **Little-Endian**, reading raw 32-bit integer headers directly causes byte-order corruption.

Our `file_handler.c` parser implements a high-performance bit-shift byte swapper:
```c
int read_int(FILE* f) {
    unsigned char buffer[4];
    fread(buffer, sizeof(unsigned char), 4, f);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}
```
This cleanly converts big-endian header fields (magic numbers, image dimensions, label counts) into native host integers on any CPU architecture.

---

## Repository Structure

```
├── README.md               # Documentation & engineering analysis
├── mnist/                  # MNIST binary dataset folder
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── v3/                     # V3 Production Engine (Thread-Safe + Xavier Init)
│   ├── main.c              # Core detection, Fisher-Yates shuffler, training loop
│   ├── mlp.c / mlp.h       # 1D Prefix-sum allocations, Xavier init, cleanup
│   ├── thread_handler.c/.h # Lock-free worker threads, index lookup, SGD backprop
│   └── file_handler.c/.h   # Big-endian IDX file parser & bit swapper
├── legacy/                 # Historical baseline implementations (V1)
└── images/                 # Architecture diagrams
```

---

## Build & Run

### Prerequisites
Ensure the raw MNIST binary files are placed inside the `mnist/` directory at the project root.

### Build & Run V3 (Production Engine)
Compile and run the optimized V3 engine directly from the root directory:

```bash
# 1. Compile
gcc -Wall -O3 v3/main.c v3/mlp.c v3/file_handler.c v3/thread_handler.c -o v3/main -lm

# 2. Run
./v3/main
```

At runtime, input your desired network parameters:
* **Hidden Layers**: `2`
* **Nodes in Hidden Layer 0**: `16`
* **Nodes in Hidden Layer 1**: `16`
* **Learning Rate**: `0.1`
* **Batch Size**: `32`
* **Epochs**: `5`
