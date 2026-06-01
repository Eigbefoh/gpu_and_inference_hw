import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn

# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []

    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(*args)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return torch.median(torch.tensor(times)).item()

# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    total_flops = num_elements * num_ops * 2

    if variant == "compiled":
        bytes_moved = num_elements * 2 * bytes_per_element
    elif variant == "eager":
        bytes_moved = num_elements * num_ops * 6 * bytes_per_element
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ai = total_flops / bytes_moved
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?

# Q1 Answer:
# The compiled element-wise operations get faster in terms of FLOP/s because they do much more arithmetic while taking almost the same amount of time.
# From the results, 1 ops and 64 ops both took about 0.216 ms, but 64 ops performs far more work per element. Since torch.compile fuses the operations, the tensor is read and written roughly once, while the GPU does more computation before writing the result back.
# So the memory movement stays almost the same, but the useful work increases. That raises arithmetic intensity and allows the GPU to achieve much higher performance, moving from 0.62 TFLOP/s at 1 ops to 39.84 TFLOP/s at 64 ops.



# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.

# Q2 Answer:
# Reason 1 : 1024×1024 is relatively small for an H100, so the matmul may not fully keep the GPU busy. That is why it only reached 31.96 TFLOP/s.
# Reason 2 : The 128 ops compiled element-wise kernel is simple, large, and fused, so it does lots of work efficiently with little extra memory traffic. That is why it reached 52.96 TFLOP/s.
# So the key point is: this result does not mean element-wise operations are generally better than matmul. It means this specific small matmul did not use the H100 as fully as the large fused element-wise workload.


#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?

# Q3 Answer:
# Between 64 ops and 128 ops, runtime increased from 0.216 ms to 0.324 ms.
# This suggests the bottleneck is shifting from memory bandwidth to compute. At 64 ops, the arithmetic intensity was 16 FLOP/Byte, but at 128 ops it rose to 32 FLOP/Byte, which is above the H100 ridge point of 20 FLOP/Byte.
# So the GPU is now doing enough arithmetic that compute capacity, not memory movement, is becoming the limiting factor.

#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
# The eager ops-K points look different because eager PyTorch does not fuse the operations. Each multiply and add creates extra intermediate tensors and extra memory traffic.
# In our results, the eager operations stayed at about 0.0833 FLOP/Byte and only around 0.26 to 0.30 TFLOP/s, even as num_ops increased.
# The compiled operations were fused, so they reused data more efficiently and did more work per memory access. That is why compiled performance rose up to 52.96 TFLOP/s, while eager stayed low