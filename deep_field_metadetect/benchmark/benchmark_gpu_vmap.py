"""Benchmark JAX GPU vmapping for deep field metadetection.

This script benchmarks the JAX implementation with GPU acceleration using vmap
for parallel processing. Results can be compared with CPU benchmarks run separately.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np

from deep_field_metadetect.benchmark.jax_simulation import (
    generate_multiband_sim_observations,
)
from deep_field_metadetect.jaxify import precision_config
from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_multi_band_deep_field_metadetect_jitted,
)
from deep_field_metadetect.jaxify.observation import (
    DFMdetMultiBandObsList,
    DFMdetObsList,
)

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print(f"Default device: {jax.devices()[0]}")
print()


def benchmark_gpu_vmapped(
    obs_fields_jax: list[tuple[dict, dict, dict]],
    bands: tuple[str, ...],
    nxy: int,
    nxy_psf: int,
    batch_size: int,
    n_runs: int = 3,
    warmup_runs: int = 2,
    profile: bool = False,
    profile_dir: str = None,
) -> dict[str, float]:
    """Benchmark JAX implementation with GPU vmapping.

    Parameters
    ----------
    obs_fields_jax : list of tuples
        List of (obs_w_dict, obs_d_dict, obs_dn_dict) for each field
    bands : tuple
        Band names
    nxy : int
        Image size
    nxy_psf : int
        PSF size
    batch_size : int
        Number of fields to process in parallel on GPU
    n_runs : int
        Number of benchmark runs
    warmup_runs : int
        Number of warmup runs for JIT
    profile : bool
        Enable XProf profiling (default: False)
    profile_dir : str
        Directory to save profiling traces (default: None)

    Returns
    -------
    results : dict
        Benchmark results including throughput and timing
    """
    n_fields = len(obs_fields_jax)
    n_bands = len(bands)

    # Create vmapped function
    vmapped_metadetect = jax.jit(
        jax.vmap(
            jax_multi_band_deep_field_metadetect_jitted,
            in_axes=(0, 0, 0, None, None, None, None),
        ),
        static_argnums=(3, 4, 5, 6),
    )

    # Prepare batches
    n_batches = (n_fields + batch_size - 1) // batch_size

    # Convert observations to batched format
    batches = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_fields)
        batch_len = end_idx - start_idx

        # Get observations for this batch
        mb_obs_wide_list = []
        mb_obs_deep_list = []
        mb_obs_deep_noise_list = []

        for field_idx in range(start_idx, end_idx):
            obs_w_dict, obs_d_dict, obs_dn_dict = obs_fields_jax[field_idx]

            mb_obs_wide_list.append(
                DFMdetMultiBandObsList(
                    [DFMdetObsList([obs_w_dict[band]]) for band in bands]
                )
            )
            mb_obs_deep_list.append(
                DFMdetMultiBandObsList(
                    [DFMdetObsList([obs_d_dict[band]]) for band in bands]
                )
            )
            mb_obs_deep_noise_list.append(
                DFMdetMultiBandObsList(
                    [DFMdetObsList([obs_dn_dict[band]]) for band in bands]
                )
            )

        # Stack into batched pytree
        mb_obs_wide_batched = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *mb_obs_wide_list
        )
        mb_obs_deep_batched = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *mb_obs_deep_list
        )
        mb_obs_deep_noise_batched = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *mb_obs_deep_noise_list
        )

        # Debug: Check dtypes on first batch
        if batch_idx == 0:
            print(f"\n  [DEBUG] Precision check for batch_size={batch_size}:")
            print(f"    precision_config.IMAGE_DTYPE: {precision_config.IMAGE_DTYPE}")
            print(f"    precision_config.MOMENT_DTYPE: {precision_config.MOMENT_DTYPE}")
            batch_img_dtype = (
                mb_obs_wide_batched._mb_obs_list[0]._obs_list[0].image.dtype
            )
            batch_wgt_dtype = (
                mb_obs_wide_batched._mb_obs_list[0]._obs_list[0].weight.dtype
            )
            print(f"    Batched image dtype: {batch_img_dtype}")
            print(f"    Batched weight dtype: {batch_wgt_dtype}")
            print(f"    jax.config.jax_enable_x64: {jax.config.jax_enable_x64}")

        batches.append(
            (
                mb_obs_wide_batched,
                mb_obs_deep_batched,
                mb_obs_deep_noise_batched,
                batch_len,
            )
        )

    # Warmup runs
    print(f"  Warming up JIT compilation for batch_size={batch_size}...")
    compile_start = time.time()
    for _ in range(warmup_runs):
        for mb_obs_w, mb_obs_d, mb_obs_dn, _ in batches:
            _ = vmapped_metadetect(
                mb_obs_w, mb_obs_d, mb_obs_dn, nxy, nxy_psf, n_bands, None
            )
    jax.block_until_ready(_)
    compile_time = time.time() - compile_start
    print(f"  Compilation complete in {compile_time:.2f}s")

    # Benchmark runs
    times = []

    # Detect precision mode from precision_config
    precision_mode = precision_config._CURRENT_MODE

    # Start profiling if requested (only profile n_runs, not compilation)
    if profile and profile_dir:
        profile_path = os.path.join(
            profile_dir, f"batch_{batch_size}_{precision_mode}_runs{n_runs}"
        )
        print(f"  Profiling {n_runs} runs (saving to {profile_path})...")
        profiler_context = jax.profiler.trace(profile_path)
        profiler_context.__enter__()
    else:
        profiler_context = None

    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end="\r")
        start = time.time()

        for mb_obs_w, mb_obs_d, mb_obs_dn, _ in batches:
            result = vmapped_metadetect(
                mb_obs_w, mb_obs_d, mb_obs_dn, nxy, nxy_psf, n_bands, None
            )
            _ = jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), result)

        elapsed = time.time() - start
        times.append(elapsed)

    print(f"  Run {n_runs}/{n_runs}... Done!")

    # Stop profiling
    if profiler_context is not None:
        profiler_context.__exit__(None, None, None)
        print(f"  Profile saved. View with: tensorboard --logdir={profile_path}")

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        "batch_size": batch_size,
        "n_fields": n_fields,
        "n_batches": n_batches,
        "compile_time": compile_time,
        "avg_total_time": avg_time,
        "std_total_time": std_time,
        "min_time": np.min(times),
        "max_time": np.max(times),
        "avg_per_field_time": avg_time / n_fields,
        "throughput": n_fields / avg_time,  # fields per second
        "all_times": times,
    }


def generate_test_data(
    n_fields: int,
    n_objs: int,
    bands: tuple[str, ...],
    dim: int,
    seed: int = 42,
) -> list[tuple[dict, dict, dict]]:
    """Generate test data for GPU benchmarks.

    Parameters
    ----------
    n_fields : int
        Number of fields to generate
    n_objs : int
        Number of objects per field
    bands : tuple
        Band names
    dim : int
        Image dimension
    seed : int
        Random seed

    Returns
    -------
    obs_fields : list
        List of (obs_w_dict, obs_d_dict, obs_dn_dict) tuples
    """
    print(f"Generating {n_fields} simulated fields for GPU benchmark...")
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_fields)

    obs_fields_jax = []

    for i, field_key in enumerate(keys):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generated {i + 1}/{n_fields} fields...", end="\r")

        # Generate JAX observations
        obs_w_jax, obs_d_jax, obs_dn_jax = generate_multiband_sim_observations(
            field_key,
            bands=bands,
            n_objs=n_objs,
            dim=dim,
        )
        obs_fields_jax.append((obs_w_jax, obs_d_jax, obs_dn_jax))

    print(f"  Generated {n_fields}")

    # Debug: Check dtypes of generated data
    if obs_fields_jax:
        first_obs = obs_fields_jax[0][0]  # First field, wide observation
        first_band = list(first_obs.keys())[0]
        print("\n  [DEBUG] Generated data dtypes:")
        print(f"    precision_config mode: {precision_config._CURRENT_MODE}")
        print(f"    Image dtype (from data): {first_obs[first_band].image.dtype}")
        print(f"    Weight dtype (from data): {first_obs[first_band].weight.dtype}")
        print(f"    Expected IMAGE_DTYPE: {precision_config.IMAGE_DTYPE}")
        print(f"    jax.config.jax_enable_x64: {jax.config.jax_enable_x64}\n")

    return obs_fields_jax


def run_gpu_benchmark(
    n_fields: int = 128,
    n_objs: int = 50,
    bands: tuple[str, ...] = ("g", "r", "i"),
    dim: int = 201,
    nxy: int = 201,
    nxy_psf: int = 53,
    gpu_batch_sizes: list[int] = None,
    n_runs: int = 3,
    warmup_runs: int = 2,
    seed: int = 42,
    profile: bool = False,
    profile_dir: str = None,
) -> dict:
    """Run GPU vmapping benchmark.

    Parameters
    ----------
    n_fields : int
        Total number of fields to process
    n_objs : int
        Number of objects per field
    bands : tuple
        Band names
    dim : int
        Image dimension
    nxy : int
        Image size for JAX
    nxy_psf : int
        PSF size for JAX
    gpu_batch_sizes : list
        List of GPU batch sizes to test
    n_runs : int
        Number of benchmark runs
    warmup_runs : int
        Number of warmup runs for JAX
    seed : int
        Random seed
    profile : bool
        Enable XProf profiling
    profile_dir : str
        Directory to save profiling traces

    Returns
    -------
    results : dict
        Benchmark results
    """
    if gpu_batch_sizes is None:
        gpu_batch_sizes = [1, 4, 8, 16]

    print("=" * 80)
    print("GPU VMAPPING BENCHMARK (JAX)")
    print("=" * 80)

    # Display precision configuration
    precision_summary = precision_config.get_precision_summary()
    print("\nPrecision Configuration:")
    print(f"  Mode: {precision_summary['mode']}")
    print(f"  Image dtype: {precision_summary['image_dtype']}")
    print(f"  Moment dtype: {precision_summary['moment_dtype']}")

    print("\nBenchmark Configuration:")
    print(f"  Total fields to process: {n_fields}")
    print(f"  Objects per field: {n_objs}")
    print(f"  Bands: {len(bands)} ({', '.join(bands)})")
    print(f"  Image size: {dim}x{dim}")
    print(f"  GPU batch sizes to test: {gpu_batch_sizes}")
    print(f"  Runs per test: {n_runs}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Random seed: {seed}")
    print()

    # Generate test data
    obs_fields_jax = generate_test_data(
        n_fields=n_fields,
        n_objs=n_objs,
        bands=bands,
        dim=dim,
        seed=seed,
    )

    print()

    results = {
        "config": {
            "n_fields": n_fields,
            "n_objs": n_objs,
            "bands": bands,
            "dim": dim,
            "nxy": nxy,
            "nxy_psf": nxy_psf,
            "gpu_batch_sizes": gpu_batch_sizes,
            "n_runs": n_runs,
            "warmup_runs": warmup_runs,
            "seed": seed,
            "device": str(jax.devices()[0]),
        },
        "gpu": {},
    }

    # GPU benchmarks
    print("-" * 80)
    print("RUNNING BENCHMARKS")
    print("-" * 80)

    for batch_size in gpu_batch_sizes:
        print(f"\nTesting with batch_size={batch_size}...")
        gpu_results = benchmark_gpu_vmapped(
            obs_fields_jax,
            bands=bands,
            nxy=nxy,
            nxy_psf=nxy_psf,
            batch_size=batch_size,
            n_runs=n_runs,
            warmup_runs=warmup_runs,
            profile=profile,
            profile_dir=profile_dir,
        )
        results["gpu"][batch_size] = gpu_results

        avg_time = gpu_results["avg_total_time"]
        std_time = gpu_results["std_total_time"]
        print(f"  Total time: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Per-field time: {gpu_results['avg_per_field_time']:.4f}s")
        print(f"  Throughput: {gpu_results['throughput']:.2f} fields/sec")

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\nGPU Performance:")
    baseline_throughput = results["gpu"][gpu_batch_sizes[0]]["throughput"]
    header = f"{'Batch':>6} | {'Throughput':>12} | {'Speedup':>8} | {'Per-field':>12}"
    print(header)
    print("-" * 55)
    for batch_size in gpu_batch_sizes:
        throughput = results["gpu"][batch_size]["throughput"]
        speedup = throughput / baseline_throughput
        per_field = results["gpu"][batch_size]["avg_per_field_time"]
        line = (
            f"{batch_size:6d} | {throughput:9.2f} f/s | "
            f"{speedup:6.2f}x | {per_field:9.4f}s"
        )
        print(line)

    best_batch = max(gpu_batch_sizes, key=lambda x: results["gpu"][x]["throughput"])
    best_throughput = results["gpu"][best_batch]["throughput"]
    print(
        f"\nBest configuration: batch={best_batch} → {best_throughput:.2f} fields/sec"
    )

    print(f"\nDevice: {results['config']['device']}")
    print("Note: Throughput comparisons exclude JIT compilation time.")

    if profile and profile_dir:
        # Detect precision mode for display
        precision_mode = precision_config._CURRENT_MODE
        precision_summary = precision_config.get_precision_summary()
        print(f"\n{'Profiling':}")
        print(f"  Traces saved to: {profile_dir}/")
        print(f"  Precision mode: {precision_mode}, Runs per profile: {n_runs}")

    print("\n" + "=" * 80)

    return results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vmapping for deep field metadetection"
    )
    parser.add_argument(
        "--n-fields",
        type=int,
        default=64,
        help="Total number of fields to process (default: 100)",
    )
    parser.add_argument(
        "--n-objs",
        type=int,
        default=32,
        help="Number of objects per field (default: 50)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=["g", "r", "i"],
        help="Bands for multi-band tests (default: g r i)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=151,
        help="Image dimension (default: 201)",
    )
    parser.add_argument(
        "--gpu-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
        help="GPU batch sizes to test (default: 1 8 16 32)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of runs per benchmark (default: 3)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs for JAX (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable XProf profiling for GPU utilization and memory analysis",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./jax-profile",
        help="Directory to save profiling traces (default: ./jax-profile)",
    )

    args = parser.parse_args()

    results = run_gpu_benchmark(
        n_fields=args.n_fields,
        n_objs=args.n_objs,
        bands=tuple(args.bands),
        dim=args.dim,
        nxy=args.dim,
        nxy_psf=53,
        gpu_batch_sizes=args.gpu_batch_sizes,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        profile=args.profile,
        profile_dir=args.profile_dir,
    )

    return results


if __name__ == "__main__":
    main()
