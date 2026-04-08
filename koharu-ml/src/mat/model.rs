/// MAT (Mask-Aware Transformer) generator — StyleGAN2-based implementation.
///
/// Implements the generator from:
///   "MAT: Mask-Aware Transformer for Large Hole Image Inpainting" (2022)
///
/// Weight key conventions match the `Sanster/MAT` HuggingFace checkpoint
/// (`Places_512_FullData_G.pt`), which follows the original MAT paper repo:
///   https://github.com/fenglinglwb/MAT
///
/// Key layout:
///   mapping.fc{0..7}.weight / .bias           (8 FC layers, 512 → 512)
///   synthesis.b4.const                        ([1, 512, 4, 4] learned constant)
///   synthesis.b{4|8|…|512}.conv0.weight / .bias
///   synthesis.b{res}.conv0.affine.weight / .bias
///   synthesis.b{res}.conv0.noise_const        (buffer, flat or [1,H,W])
///   synthesis.b{res}.conv0.noise_strength     (scalar)
///   synthesis.b{res}.conv1.*                  (same sub-keys)
///   synthesis.b{res}.torgb.weight / .bias
///   synthesis.b{res}.torgb.affine.weight / .bias
///
/// Channel widths (StyleGAN2 standard, 512-cap):
///   res   4   8  16  32  64  128  256  512
///   ch  512 512 512 512 256  128   64   32
///
/// Input:  masked image (RGB, 3-ch) concatenated with mask (1-ch) = 4 channels.
/// Output: RGB image (3 channels), values in [0, 1].
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------

const Z_DIM: usize = 512;
const W_DIM: usize = 512;
const MAPPING_LAYERS: usize = 8;

/// Resolutions in the synthesis network (4 → 512).
const RESOLUTIONS: [usize; 8] = [4, 8, 16, 32, 64, 128, 256, 512];

fn ch(res: usize) -> usize {
    match res {
        4 | 8 | 16 | 32 => 512,
        64 => 256,
        128 => 128,
        256 => 64,
        512 => 32,
        _ => 32,
    }
}

// ---------------------------------------------------------------------------
// Leaky ReLU helper (candle 0.9 has no `.leaky_relu()` method on Tensor)
// ---------------------------------------------------------------------------

fn leaky_relu(x: &Tensor, neg_slope: f64) -> Result<Tensor> {
    // leaky_relu(x) = max(x, neg_slope * x)
    let scaled = (x * neg_slope)?;
    Ok(x.maximum(&scaled)?)
}

// ---------------------------------------------------------------------------
// Fully-connected layer
// ---------------------------------------------------------------------------

struct Linear {
    weight: Tensor, // [out, in]
    bias: Tensor,   // [out]
}

impl Linear {
    fn load(vb: &VarBuilder, in_features: usize, out_features: usize) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = vb.get(out_features, "bias")?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out.broadcast_add(&self.bias)?)
    }
}

// ---------------------------------------------------------------------------
// Mapping network: z (512) → w (512) via 8 FC layers with leaky ReLU
// ---------------------------------------------------------------------------

struct MappingNetwork {
    layers: Vec<Linear>,
}

impl MappingNetwork {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(MAPPING_LAYERS);
        for i in 0..MAPPING_LAYERS {
            layers.push(
                Linear::load(&vb.pp(format!("fc{i}")), Z_DIM, W_DIM)
                    .with_context(|| format!("mapping.fc{i}"))?,
            );
        }
        Ok(Self { layers })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = z.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
            x = leaky_relu(&x, 0.2)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Affine style layer: w (W_DIM) → style (out_features)
// ---------------------------------------------------------------------------

struct AffineLayer {
    layer: Linear,
}

impl AffineLayer {
    fn load(vb: &VarBuilder, out_features: usize) -> Result<Self> {
        Ok(Self {
            layer: Linear::load(vb, W_DIM, out_features)?,
        })
    }

    fn forward(&self, w: &Tensor) -> Result<Tensor> {
        self.layer.forward(w)
    }
}

// ---------------------------------------------------------------------------
// Modulated Conv2d
//
// Applies a per-sample affine style from w, then weight-demodulated conv.
// ---------------------------------------------------------------------------

struct ModulatedConv2d {
    weight: Tensor,         // [out_ch, in_ch, kH, kW]
    bias: Tensor,           // [out_ch]
    affine: AffineLayer,
    noise_const: Tensor,    // flat buffer, reshaped at runtime
    noise_strength: Tensor, // scalar
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
}

impl ModulatedConv2d {
    fn load(vb: &VarBuilder, in_ch: usize, out_ch: usize, kernel_size: usize) -> Result<Self> {
        let weight = vb.get((out_ch, in_ch, kernel_size, kernel_size), "weight")?;
        let bias = vb.get(out_ch, "bias")?;
        let affine = AffineLayer::load(&vb.pp("affine"), in_ch)?;
        // noise_const is stored as a registered buffer; its shape in the
        // checkpoint may be [res, res] or flat.  We flatten and reshape
        // at forward time to match the actual spatial dimensions.
        let noise_const = vb.get_with_hints(
            candle_core::Shape::from(()),
            "noise_const",
            candle_nn::Init::Const(0.0),
        )?;
        let noise_strength = vb.get_with_hints(
            candle_core::Shape::from(()),
            "noise_strength",
            candle_nn::Init::Const(0.0),
        )?;
        Ok(Self { weight, bias, affine, noise_const, noise_strength, in_ch, out_ch, kernel_size })
    }

    /// `x` – (batch, in_ch, H, W)
    /// `w` – (batch, W_DIM) style vector
    fn forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (batch, _c, h, wi) = x.dims4()?;

        // 1. Per-sample style: [batch, in_ch]
        let style = self.affine.forward(w)?;
        // Reshape to [batch, 1, in_ch, 1, 1]
        let style = style.reshape((batch, 1, self.in_ch, 1, 1))?;

        // 2. Modulate: weight' = weight * style
        //    weight: [out_ch, in_ch, kH, kW] → [1, out_ch, in_ch, kH, kW]
        let weight = self.weight.unsqueeze(0)?;
        let w_mod = weight.broadcast_mul(&style)?; // [batch, out_ch, in_ch, kH, kW]

        // 3. Demodulate: normalise over (in_ch, kH, kW) axes
        let denom = (w_mod.sqr()?.sum((2usize, 3usize, 4usize))? + 1e-8f64)?
            .sqrt()?
            .reshape((batch, self.out_ch, 1, 1, 1))?;
        let w_demod = w_mod.broadcast_div(&denom)?; // [batch, out_ch, in_ch, kH, kW]

        // 4. Group-conv: fold batch into groups so each sample uses its own kernel.
        //    x_folded: [1, batch*in_ch, H, W]
        //    w_folded: [batch*out_ch, in_ch, kH, kW]
        let x_folded = x.reshape((1, batch * self.in_ch, h, wi))?;
        let w_folded = w_demod
            .reshape((batch * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))?;

        let pad = self.kernel_size / 2;
        let out = x_folded.conv2d(&w_folded, pad, 1, 1, batch)?;
        let (_, _, h_out, w_out) = out.dims4()?;
        let out = out.reshape((batch, self.out_ch, h_out, w_out))?;

        // 5. Add bias
        let bias = self.bias.reshape((1, self.out_ch, 1, 1))?;
        let out = out.broadcast_add(&bias)?;

        // 6. Scaled noise
        let noise = self.make_noise(h_out, w_out, x.device())?;
        let strength = self.noise_strength.reshape((1, 1, 1, 1))?;
        let out = (out + noise.broadcast_mul(&strength)?)?;

        Ok(out)
    }

    fn make_noise(
        &self,
        h: usize,
        w: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let spatial = h * w;
        let flat = self.noise_const.flatten_all()?;
        let n = flat.elem_count();
        if n >= spatial {
            Ok(flat.narrow(0, 0, spatial)?.reshape((1, 1, h, w))?)
        } else {
            // Fallback: zeros (noise_const too small, shouldn't happen in practice)
            Ok(Tensor::zeros((1, 1, h, w), DType::F32, device)?)
        }
    }
}

// ---------------------------------------------------------------------------
// ToRGB layer: 1×1 modulated conv (no demodulation)
// ---------------------------------------------------------------------------

struct ToRgb {
    weight: Tensor,   // [3, in_ch, 1, 1]
    bias: Tensor,     // [3]
    affine: AffineLayer,
    in_ch: usize,
}

impl ToRgb {
    fn load(vb: &VarBuilder, in_ch: usize) -> Result<Self> {
        let weight = vb.get((3, in_ch, 1, 1), "weight")?;
        let bias = vb.get(3, "bias")?;
        let affine = AffineLayer::load(&vb.pp("affine"), in_ch)?;
        Ok(Self { weight, bias, affine, in_ch })
    }

    fn forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (batch, _c, h, wi) = x.dims4()?;

        // Modulate (no demodulation for ToRGB)
        let style = self.affine.forward(w)?;
        let style = style.reshape((batch, 1, self.in_ch, 1, 1))?;
        let weight = self.weight.unsqueeze(0)?;
        let w_mod = weight.broadcast_mul(&style)?; // [batch, 3, in_ch, 1, 1]
        let w_folded = w_mod.reshape((batch * 3, self.in_ch, 1, 1))?;

        let x_folded = x.reshape((1, batch * self.in_ch, h, wi))?;
        let out = x_folded.conv2d(&w_folded, 0, 1, 1, batch)?;
        let (_, _, h_out, w_out) = out.dims4()?;
        let out = out.reshape((batch, 3, h_out, w_out))?;

        let bias = self.bias.reshape((1, 3, 1, 1))?;
        Ok(out.broadcast_add(&bias)?)
    }
}

// ---------------------------------------------------------------------------
// Synthesis block: conv0 + conv1 + torgb, with skip RGB accumulation
// ---------------------------------------------------------------------------

struct SynthesisBlock {
    conv0: ModulatedConv2d,
    conv1: ModulatedConv2d,
    torgb: ToRgb,
    res: usize,
}

impl SynthesisBlock {
    fn load(vb: &VarBuilder, res: usize, in_ch: usize, out_ch: usize) -> Result<Self> {
        let block_vb = vb.pp(format!("b{res}"));
        let conv0 = ModulatedConv2d::load(&block_vb.pp("conv0"), in_ch, out_ch, 3)
            .with_context(|| format!("synthesis.b{res}.conv0"))?;
        let conv1 = ModulatedConv2d::load(&block_vb.pp("conv1"), out_ch, out_ch, 3)
            .with_context(|| format!("synthesis.b{res}.conv1"))?;
        let torgb = ToRgb::load(&block_vb.pp("torgb"), out_ch)
            .with_context(|| format!("synthesis.b{res}.torgb"))?;
        Ok(Self { conv0, conv1, torgb, res })
    }

    /// `x`    – (batch, in_ch, H, W)
    /// `w`    – (batch, W_DIM)
    /// `skip` – accumulated RGB from the previous block, or None for b4
    ///
    /// Returns `(feature_map, updated_skip_rgb)`.
    fn forward(&self, x: &Tensor, w: &Tensor, skip: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // Upsample ×2 for all blocks except the first (4×4)
        let x = if self.res > 4 {
            let (_, _, h, wi) = x.dims4()?;
            x.upsample_nearest2d(h * 2, wi * 2)?
        } else {
            x.clone()
        };

        let x = leaky_relu(&self.conv0.forward(&x, w)?, 0.2)?;
        let x = leaky_relu(&self.conv1.forward(&x, w)?, 0.2)?;

        let rgb = self.torgb.forward(&x, w)?;

        let new_skip = match skip {
            Some(prev) => {
                let (_, _, h, wi) = rgb.dims4()?;
                let prev_up = prev.upsample_nearest2d(h, wi)?;
                (prev_up + rgb)?
            }
            None => rgb,
        };

        Ok((x, new_skip))
    }
}

// ---------------------------------------------------------------------------
// Full MAT generator
// ---------------------------------------------------------------------------

pub struct Mat {
    mapping: MappingNetwork,
    /// Learned constant 4×4 input: [1, 512, 4, 4]
    const_input: Tensor,
    blocks: Vec<SynthesisBlock>,
}

impl Mat {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let mapping = MappingNetwork::load(&vb.pp("mapping")).context("mapping network")?;

        let const_input = vb
            .get((1, ch(4), 4, 4), "synthesis.b4.const")
            .context("synthesis.b4.const")?;

        let mut blocks = Vec::with_capacity(RESOLUTIONS.len());
        let synthesis_vb = vb.pp("synthesis");
        for (i, &res) in RESOLUTIONS.iter().enumerate() {
            let in_ch = if i == 0 { ch(4) } else { ch(RESOLUTIONS[i - 1]) };
            let out_ch = ch(res);
            blocks.push(
                SynthesisBlock::load(&synthesis_vb, res, in_ch, out_ch)
                    .with_context(|| format!("synthesis block b{res}"))?,
            );
        }

        Ok(Self { mapping, const_input, blocks })
    }

    /// Run forward inference.
    ///
    /// `image` – (1, 3, H, W) f32 normalised to [0, 1]
    /// `mask`  – (1, 1, H, W) f32, 1 = masked pixels (area to inpaint)
    ///
    /// z is set to zeros for a deterministic (mean-style) output.
    pub fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let device = image.device();
        let batch = image.dim(0)?;

        // Zero z → deterministic w via the mapping network
        let z = Tensor::zeros((batch, Z_DIM), DType::F32, device)?;
        let w = self.mapping.forward(&z)?; // [batch, W_DIM]

        // Expand the learned constant to the batch size
        let x = self.const_input.expand((batch, ch(4), 4, 4))?;

        let mut cur_x = x;
        let mut skip: Option<Tensor> = None;
        for block in &self.blocks {
            let (new_x, new_skip) = block.forward(&cur_x, &w, skip.as_ref())?;
            cur_x = new_x;
            skip = Some(new_skip);
        }

        // Final RGB: tanh → [−1,1] → rescale to [0,1]
        let rgb = skip.expect("synthesis must have at least one block");
        let rgb = rgb.tanh()?;
        let rgb = ((rgb + 1.0)? / 2.0)?;

        // Resize to input spatial dimensions if needed
        let (_, _, ih, iw) = image.dims4()?;
        let (_, _, oh, ow) = rgb.dims4()?;
        let rgb = if ih != oh || iw != ow {
            rgb.upsample_nearest2d(ih, iw)?
        } else {
            rgb
        };

        // Composite: paste inpainted output over original only in masked region
        let mask_inv = (mask.ones_like()? - mask)?;
        let out = (rgb.broadcast_mul(mask)? + image.broadcast_mul(&mask_inv)?)?;

        Ok(out)
    }
}
