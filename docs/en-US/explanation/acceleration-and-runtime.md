---
title: Acceleration and Runtime
---

# Acceleration and Runtime

Koharu supports multiple runtime backends so the same pipeline can run across a wide range of hardware.

## CUDA on NVIDIA GPUs

CUDA is the primary GPU backend on systems with supported NVIDIA hardware.

- Koharu supports NVIDIA GPUs with compute capability 7.5 or higher
- Koharu bundles CUDA toolkit 13.0

On first run, Koharu extracts the required dynamic libraries into the application data directory.

!!! note

    CUDA acceleration depends on a recent NVIDIA driver. If the driver does not support CUDA 13.0, Koharu falls back to CPU.

## Metal on Apple Silicon

On macOS, Koharu supports Metal acceleration on Apple Silicon systems such as the M1 and M2 families.

## ZLUDA on Windows AMD GPUs

Koharu also supports a Windows-only AMD path for selected Candle models by staging a managed ZLUDA runtime and validating it against an installed HIP SDK.

- Koharu downloads its own ZLUDA bundle under the local runtime directory
- you still need the AMD HIP SDK installed locally; Koharu detects it but does not install it for you
- this v1 path accelerates only `comic-text-bubble-detector`, `comic-text-detector`, `comic-text-detector-seg`, and `aot-inpainting`
- other Candle models stay on CPU on Windows AMD when the backend resolves to ZLUDA
- llama.cpp stays on Vulkan on AMD even though ZLUDA exposes `nvcuda.dll`

If HIP is missing, the staged ZLUDA files are missing, or ZLUDA library validation fails, Koharu falls back to CPU for the unsupported Candle path and logs the reason.

## Vulkan on Windows and Linux

On Windows and Linux, Vulkan is available as an alternative GPU path for OCR and LLM inference when CUDA or Metal are not available.

AMD and Intel GPUs can benefit from Vulkan for llama.cpp and OCR workloads, but broader Candle acceleration on those systems remains limited.

## CPU fallback

Koharu can always run on CPU when GPU acceleration is unavailable or when you force CPU mode explicitly.

```bash
# macOS / Linux
koharu --cpu

# Windows
koharu.exe --cpu
```

## Why fallback matters

Fallback behavior makes Koharu usable on more machines, but it changes the performance profile:

- GPU inference is much faster when supported
- CPU mode is more compatible but can be substantially slower
- Smaller local LLMs are often the best choice on CPU-only systems

For model selection guidance, see [Models and Providers](models-and-providers.md).
