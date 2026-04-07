# Koharu — Planned Improvements

Research-backed upgrade plan focused on Korean manhwa support. All items below require code changes except LLM additions (those are config/API only).

---

## 1. OCR Improvements

### 1a. GLM-OCR (Drop-in upgrade)

**Why:** PaddleOCR-VL-1.5 does not officially document Korean or Japanese support — it was designed for Chinese/English documents. GLM-OCR explicitly supports Korean and Japanese among its 8 languages and ranks #1 on OmniDocBench V1.5. A recent March 2026 update also introduced an agent-friendly Skill mode via its SDK.

| Property | Value |
|---|---|
| HuggingFace | `zai-org/GLM-OCR` |
| GGUF | `ggml-org/GLM-OCR-GGUF` |
| MLX | `mlx-community/GLM-OCR-bf16` (2.21 GB), 4/5/6/8bit variants |
| Size | 0.9B parameters |
| Languages | Chinese, English, French, Spanish, Russian, German, **Japanese, Korean** |
| OCRBench | 94.62 on OmniDocBench V1.5 (#1) |
| Architecture | CogViT visual encoder + GLM-0.5B LM decoder + PP-DocLayout-V3 |

**Code change:** `koharu-ml/src/paddleocr_vl/mod.rs` — swap hardcoded HuggingFace repo ID from `PaddlePaddle/PaddleOCR-VL-1.5` to `zai-org/GLM-OCR`. Inference interface is compatible (GGUF via llama.cpp).

**Integration path:** GGUF → same `llama-cpp-rs` pipeline already used for PaddleOCR-VL-1.5-GGUF. Near drop-in replacement.

---

### 1b. Qwen3-VL OCR (High-accuracy alternative & Visual Agent)

**Why:** Upgraded from Qwen2.5-VL, **Qwen3-VL** is Alibaba’s latest multimodal powerhouse. It features vastly expanded OCR capabilities (now supporting 32 languages) and is highly robust against blur, tilt, and low-light conditions common in scanned manhwa. It easily parses complex long-document structures and rare/ancient characters. 

| Property | Value |
|---|---|
| HuggingFace | `QwenLM/Qwen3-VL` series |
| GGUF | Community quants available (e.g., `unsloth`, `bartowski`) |
| Context | Native **256K** (expandable to 1M) |
| Enhancements | Multi-modal coding, 3D grounding, and native visual agent control |

**Code change:** Same swap in `koharu-ml/src/paddleocr_vl/mod.rs`. Heavier than GLM-OCR but offers top-tier visual reasoning and extraction. Consider as a premium quality mode vs. GLM-OCR as a fast mode.

---

### 1c. Apple Vision OCR (macOS-native, zero download)

**Why:** Ships with every Mac. Runs on the Neural Engine — no model download, no memory footprint. Supports Korean and Japanese. For clean webtoon panel text (clear fonts, no heavy stylization) it can be highly accurate.

| Property | Value |
|---|---|
| Framework | `Vision.framework` (`VNRecognizeTextRequest`) |
| Korean support | ✅ Added macOS 13 / iOS 16 |
| Japanese support | ✅ Available since earlier |
| Model size | 0 (on-device OS framework) |
| Speed | Neural Engine — near-instant |
| Weakness | Cannot be fine-tuned; struggles with artistic/stylized fonts |

**Integration path:** Requires a Rust↔macOS FFI bridge. Options:
- Thin Swift CLI wrapper invoked as subprocess (simplest, adds process overhead)
- `objc2` + `icrate` Rust crates for direct Objective-C FFI (complex but in-process)

**Code change:** New module `koharu-ml/src/apple_vision_ocr/` with a platform-gated (`#[cfg(target_os = "macos")]`) implementation. Falls back to GLM-OCR on non-macOS.

---

## 2. Inpainting Improvements

### 2a. big-LaMa (Safe quality upgrade)

**Why:** The current `lama-manga` is a small manga-tuned variant optimized for speed. big-LaMa (~200MB) noticeably outperforms it on complex screentones, hatching, and gradient backgrounds common in manhwa.

| Property | Value |
|---|---|
| HuggingFace | `smartywu/big-lama` |
| Size | ~200MB |
| Speed on M-series | ~1-3s with MPS (vs ~0.5s for manga-tuned small) |
| Quality | Better on complex textures; comparable to small on clean white speech bubbles |
| Format | PyTorch checkpoint (same as current lama-manga) |

**Code change:** `koharu-ml/src/lama/mod.rs` — add `big-lama` as a second registered model package.

---

### 2b. MAT — Mask-Aware Transformer (Best quality upgrade)

**Why:** Transformer-based architecture with better edge coherence than LaMa on complex backgrounds. Available via IOPaint's ecosystem (`Sanster/MAT` on HuggingFace).

| Property | Value |
|---|---|
| HuggingFace | `Sanster/MAT` (IOPaint-compatible) |
| Size | ~350MB |
| Speed on M-series | ~3-8s with MPS |
| Architecture | Transformer (not GAN/Fourier) |

**Code change:** New module `koharu-ml/src/mat/` mirroring the structure of `koharu-ml/src/lama/`. Needs a new inference implementation.

---

## 3. LLM / Translation Additions

All LLM additions work **without code changes** — add them to the `ModelId` enum in `koharu-llm/src/lib.rs` and they're immediately available in the UI.

### 3a. OpenRouter API Support

**Why:** OpenRouter provides unified access to Claude, GPT-5/4o, Gemini 3, and hundreds of other models under a single API key with OpenAI-compatible endpoints.

| Property | Value |
|---|---|
| Base URL | `https://openrouter.ai/api/v1` |
| Compatibility | OpenAI-compatible |
| Top picks | `anthropic/claude-4.5`, `google/gemini-3.1-pro`, `qwen/qwen3.5` |

**Code change:** Add `openrouter` as a named API provider reusing the existing OpenAI provider implementation.

---

### 3b. EXAONE 4.0 1.2B (Best local lightweight reasoning model)

**Why:** LG AI Research's latest update, Exaone 4.0, brings massive improvements to its small-tier models. The 1.2B Reasoning variant boasts an impressive Intelligence Index of 8.3 and punches far above its weight class for local logical processing. Excellent for users running Koharu on heavily constrained hardware.

| Property | Value |
|---|---|
| HuggingFace | `LGAI-EXAONE/EXAONE-4.0-1.2B-Reasoning` |
| GGUF | bartowski quants |
| Size options | **1.2B** (Reasoning & Non-reasoning variants) |
| Cost | Free for 1M tokens API / Open weights |
| Speed on M4 Air | Extremely fast (32+ tok/s) |

**Code change:** Add to `ModelId` enum in `koharu-llm/src/lib.rs`.

---

### 3c. Local Gemma 4 Support (E2B & E4B Variants)

**Why:** Google's newly released **Gemma 4** shifts the paradigm with a **128K native context window** for its small models and native vision/audio processing. Perfect for maintaining character voice across a full manhwa chapter. While Gemma 4 doesn't have a 9B model, its "E" (Effective) embedding architecture makes the **E4B** incredibly capable for its size. It's the highest-performing option that strictly fits under a 10B hardware footprint.

| Property | Value |
|---|---|
| HuggingFace | `google/gemma-4-E4B-it`, `google/gemma-4-E2B-it` |
| GGUF | Community GGUFs available (e.g., LiteRT-LM) |
| Size options | **E2B (<1.5GB RAM)**, **E4B (Recommended for <10B limits)** |
| Context window | **128K tokens** (massively expanded for small models) |
| Architecture | Native multimodal (vision/audio), Per-Layer Embeddings (PLE) |

**Code change:** Add `gemma-4-e2b` and `gemma-4-e4b` models to `ModelId` enum.

---

### 3d. Qwen 3.5 Local Support (4B & 9B Variants)

**Why:** Alibaba's Qwen 3.5 "Small" series delivers massive capabilities in highly practical sizes. The 9B model rivals 100B+ models on reasoning benchmarks, while the 4B brings robust native multimodal capability (early-fusion text/vision) to entry-level laptops. Both have a natively massive 262k token context window, outclassing virtually all legacy dense models in their weight class.

| Property | Value |
|---|---|
| HuggingFace | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` |
| GGUF | `unsloth` / `bartowski` Q4_K_M formats available |
| 4B Requirements | ~6 GB VRAM/RAM (Fits comfortably on most recent machines) |
| 9B Requirements | ~10-16 GB VRAM/RAM (Ideal for 16GB unified memory Macs) |
| Context window | **262,144 tokens** |
| Features | Early-fusion multimodal native, strong Korean support, multi-step agent reasoning |

**Code change:** Add `qwen3.5-4b` and `qwen3.5-9b` models to `ModelId` enum.

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 1 | Qwen 3.5 4B/9B (enum addition) | Very low | Top-tier performance for 6GB - 16GB RAM constraints |
| 2 | Gemma 4 E4B (enum addition) | Very low | Best local context window (128K) & multimodal support under 10B |
| 3 | OpenRouter API provider | Low — reuses OpenAI code | Access to Gemini 3.1 Pro / Claude 4.5 |
| 4 | GLM-OCR (repo ID swap) | Low — swap one constant | Fixes Korean OCR gap (now with Skill mode) |
| 5 | big-LaMa (second model package) | Low | Better inpainting quality |
| 6 | MAT inpainting | Medium | Best complex background inpainting |
| 7 | Qwen3-VL OCR | Low — same as GLM-OCR swap | Highest accuracy ceiling (32 languages, 256K vision context) |
| 8 | Apple Vision OCR | High — FFI bridge required | Zero-download macOS-native option |