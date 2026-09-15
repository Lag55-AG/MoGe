# Triton Compatibility Patch for FlexGEMM

## Modified Files
* `flex_gemm/kernels/triton/sample/grid_sample.py`
* `flex_gemm/kernels/triton/hashmap.py`
* `flex_gemm/kernels/triton/neighbor_cache/neighbor_map.py`
* `flex_gemm/kernels/triton/neighbor_cache/output_coords.py`
* `flex_gemm/kernels/triton/neighbor_cache/post_process.py`
* `flex_gemm/kernels/triton/neighbor_cache/scatter_rank.py`
* `flex_gemm/kernels/triton/neighbor_cache/scatter_to_segment.py`
* `flex_gemm/kernels/triton/index_segment_reduce.py`
* `flex_gemm/kernels/triton/utils.py`
* `tests/gpu_integration_test.py` (added)
* `tests/test_triton_annotations.py` (added)

## Rationale
Triton 3.2.0 removes the `.itemsize` attribute from the dtype object, requiring the use of `.primitive_bitwidth // 8` instead. Also, Triton's JIT binder became stricter on Python type hints, breaking compatibility when using unsupported `tl.pointer_type` and `tl.tensor | None`. These issues caused compilation errors for the FlexGEMM kernels when running on newer environments.
To resolve this:
- Replaced `.itemsize` usages with `.primitive_bitwidth // 8`.
- Stripped unsupported annotations such as `tl.pointer_type`, `tl.tensor | None`, and `tl.pointer_type | None` from JIT helper logic and wrappers.
- For optional tensor pointers in JIT kernels, we modified the parameter list to accept `HAS_X: tl.constexpr`, and replaced inner `X is not None` logic with `HAS_X`. Python wrappers were also updated to properly pass `HAS_X` and a valid dummy pointer when `X` was previously passed as `None`.

## Remaining Risks
- The GPU integration test and Triton tests can only be fully executed on machines equipped with NVIDIA GPUs and installed drivers. Since this environment lacks actual NVIDIA hardware, a fallback check was added to exit gracefully without failure. However, it cannot be considered "passed" until run in a GPU-enabled environment.

## Exact Local Validation Commands
```bash
# 1. Run AST parser test to verify unsupported annotations have been removed
python3 tests/test_triton_annotations.py

# 2. Run the integration test on GPU
python3 tests/gpu_integration_test.py
```
