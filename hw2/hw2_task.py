import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)

def optimized_loop(model, input_ids, n_steps):
    generated_tokens = []

    with torch.inference_mode():
        # Prefill: process the full prompt once and build the KV cache.
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # First generated token comes from the final prompt position.
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)

        # Decode: now only pass the newest token, using the cached past.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id)

    return torch.cat(generated_tokens, dim=0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    trace_path = RESULTS_DIR / trace_name

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")




def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    return optimized_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_bf16_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================


### Changes made and speedup per fix:
#
# 1. Added KV-cache decoding.
#    The original loop recomputed the full growing sequence every step.
#    The optimized loop does one prefill pass, stores past_key_values,
#    then decodes using only the latest token. This was the biggest speedup.
#
# 2. Wrapped generation in torch.inference_mode().
#    This removes unnecessary autograd/training overhead during inference.
#
# 3. Removed .item() from the decode loop.
#    The original code moved each generated token from GPU to CPU every step,
#    causing synchronization. The optimized version keeps tokens on GPU and
#    converts them only at the end.
#
# 4. Removed torch.cat on the full generated sequence inside the hot loop.
#    With KV cache, we no longer need to rebuild and resend the full sequence
#    every step.
#
# 5. Tested lower precision model loading.
#    FP16 improved the optimized run to about 0.21s and 4.49x speedup.
#    BF16 was slightly faster in our experiment, reaching 0.20s and 4.68x.
#
# Overall:
#    Slow baseline: 0.95s, 134.1 tok/s
#    Optimized BF16: 0.20s, 627.3 tok/s
#    Final speedup: 4.68x


### Biggest impact and why:
#
# The biggest impact came from adding KV-cache decoding.
# The original loop recomputed the full growing sequence for every new token,
# so each decode step repeated work from all previous tokens.
#
# With KV cache, the model processes the prompt once, stores past_key_values,
# and then only processes the newest token at each decode step. This directly
# targets the main inference bottleneck: repeated full-context computation.
#
# The result was a large speedup:
# Slow baseline: 0.95s, 134.1 tok/s
# Optimized BF16: 0.20s, 627.3 tok/s
# Final speedup: 4.68x
#
# BF16 helped slightly over FP16, but the main improvement came from changing
# the decode algorithm, not just changing dtype.