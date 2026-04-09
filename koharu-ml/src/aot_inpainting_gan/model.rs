/// AOT-GAN generator — exact implementation matching `NimaBoscarino/aot-gan-places2`.
///
/// Architecture reference: https://github.com/researchmm/AOT-GAN-for-Inpainting
///
/// Weight key layout (flat state_dict, no prefix):
///   encoder.1.{weight,bias}              Conv2d(4→64,  k=7, pad=3 reflect)
///   encoder.3.{weight,bias}              Conv2d(64→128, k=4, stride=2, pad=1)
///   encoder.5.{weight,bias}              Conv2d(128→256, k=4, stride=2, pad=1)
///   middle.N.block{00..03}.1.{weight,bias}  Dilation branches (Conv2d 256→64, d=1/2/4/8)
///   middle.N.fuse.1.{weight,bias}        Fuse conv   (Conv2d 256→256, pad=1 reflect)
///   middle.N.gate.1.{weight,bias}        Gate conv   (Conv2d 256→256, pad=1 reflect)
///   decoder.0.conv.{weight,bias}         UpConv(256→128): bilinear ×2 + Conv2d k=3 pad=1
///   decoder.2.conv.{weight,bias}         UpConv(128→64):  bilinear ×2 + Conv2d k=3 pad=1
///   decoder.4.{weight,bias}              Conv2d(64→3, k=3, pad=1)
use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reflection padding for a 4-D tensor (B, C, H, W).
/// Mirrors the interior (not the edge pixel itself), matching PyTorch's
/// `nn.ReflectionPad2d` semantics.
fn reflect_pad2d(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let (_, _, h, w) = x.dims4()?;
    // Width: mirror columns [1..pad] left and [W-1-pad..W-1] right
    let left = x.narrow(3, 1, pad)?.flip(&[3usize])?;
    let right = x.narrow(3, w - 1 - pad, pad)?.flip(&[3usize])?;
    let x = Tensor::cat(&[&left, x, &right], 3)?;
    // Height: mirror rows [1..pad] top and [H-1-pad..H-1] bottom
    let top = x.narrow(2, 1, pad)?.flip(&[2usize])?;
    let bottom = x.narrow(2, h - 1 - pad, pad)?.flip(&[2usize])?;
    Ok(Tensor::cat(&[&top, &x, &bottom], 2)?)
}

/// Stateless spatial normalisation used inside AOTBlock.
/// Equivalent to PyTorch's `my_layer_norm`:
///   mean, std per-spatial-position → normalise → scale by 5.
fn spatial_norm(x: &Tensor) -> Result<Tensor> {
    // Mean and std over H and W axes (dims 2, 3)
    let mean = x.mean_keepdim((2usize, 3usize))?;
    let var = ((x - &mean)?.sqr()?.mean_keepdim((2usize, 3usize))? + 1e-9f64)?;
    let std = var.sqrt()?;
    let normalised = ((x - &mean)? / std)?;
    Ok((normalised * 5.0f64)?)
}

// ---------------------------------------------------------------------------
// Encoder conv layers
// ---------------------------------------------------------------------------

/// Conv2d with preceding reflection padding (replaces nn.ReflectionPad2d + Conv2d).
struct ReflectConv {
    conv: Conv2d,
    pad: usize,
}

impl ReflectConv {
    fn load(
        vb: &VarBuilder,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        pad: usize,
    ) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_ch,
            out_ch,
            kernel,
            Conv2dConfig {
                stride,
                padding: 0, // padding is done manually via reflection
                dilation,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.clone(),
        )?;
        Ok(Self { conv, pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = reflect_pad2d(x, self.pad)?;
        Ok(self.conv.forward(&padded)?)
    }
}

// ---------------------------------------------------------------------------
// AOT Block — core of the middle network
// ---------------------------------------------------------------------------

/// Single AOT block with 4 atrous branches + fuse + gate.
/// See `middle.N` in the weight key layout.
struct AotBlock {
    /// Dilation branches: block00 (d=1), block01 (d=2), block02 (d=4), block03 (d=8).
    /// Each: Conv2d(256 → 64, k=3, dilation=d) preceded by ReflectionPad2d(d).
    branches: Vec<ReflectConv>,
    /// Fuse: Conv2d(256 → 256, k=3) after cat of all branches. Preceded by pad=1.
    fuse: ReflectConv,
    /// Gate: Conv2d(256 → 256, k=3). Same shape as fuse. Preceded by pad=1.
    gate: ReflectConv,
}

impl AotBlock {
    fn load(vb: &VarBuilder, block_idx: usize) -> Result<Self> {
        let blk = vb.pp(format!("middle.{block_idx}"));
        let dilations = [1usize, 2, 4, 8];
        let branch_names = ["block00", "block01", "block02", "block03"];

        let mut branches = Vec::with_capacity(4);
        for (name, &d) in branch_names.iter().zip(dilations.iter()) {
            // Key pattern: middle.N.blockMM.1.{weight,bias}
            // The Conv2d is at sequential index 1 (index 0 is ReflectionPad2d).
            branches.push(
                ReflectConv::load(&blk.pp(name).pp("1"), 256, 64, 3, 1, d, d)
                    .with_context(|| format!("middle.{block_idx}.{name}"))?,
            );
        }

        let fuse = ReflectConv::load(&blk.pp("fuse").pp("1"), 256, 256, 3, 1, 1, 1)
            .with_context(|| format!("middle.{block_idx}.fuse"))?;
        let gate = ReflectConv::load(&blk.pp("gate").pp("1"), 256, 256, 3, 1, 1, 1)
            .with_context(|| format!("middle.{block_idx}.gate"))?;

        Ok(Self { branches, fuse, gate })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Run all 4 atrous branches on the input, concatenate.
        let outs: Vec<Tensor> = self
            .branches
            .iter()
            .map(|b| b.forward(x))
            .collect::<Result<_>>()?;
        let cat = Tensor::cat(&outs, 1)?; // [B, 256, H, W]

        // Apply spatial normalisation then fuse.
        let normed = spatial_norm(&cat)?;
        let fused = candle_nn::ops::leaky_relu(&self.fuse.forward(&normed)?, 0.2)?;

        // Gate: sigmoid-weighted residual blend.
        let gate = candle_nn::ops::sigmoid(&self.gate.forward(&fused)?)?;
        // out = x * (1 − gate) + fused * gate
        let one_minus_gate = (gate.ones_like()? - &gate)?;
        Ok((x.broadcast_mul(&one_minus_gate)? + fused.broadcast_mul(&gate)?)?)
    }
}

// ---------------------------------------------------------------------------
// UpConv decoder block
// ---------------------------------------------------------------------------

/// Bilinear ×2 upsampling followed by 3×3 same-padding conv.
/// Weight key: `decoder.N.conv.{weight,bias}`.
struct UpConv {
    conv: Conv2d,
}

impl UpConv {
    fn load(vb: &VarBuilder, in_ch: usize, out_ch: usize) -> Result<Self> {
        let conv = candle_nn::conv2d(
            in_ch,
            out_ch,
            3,
            Conv2dConfig {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        Ok(self.conv.forward(&up)?)
    }
}

// ---------------------------------------------------------------------------
// Full generator
// ---------------------------------------------------------------------------

pub struct AotGanGenerator {
    enc1: ReflectConv, // encoder.1  4→64,  k=7, pad=3
    enc3: Conv2d,      // encoder.3  64→128, k=4, stride=2, pad=1 (regular padding)
    enc5: Conv2d,      // encoder.5  128→256, k=4, stride=2, pad=1
    middle: Vec<AotBlock>,
    dec0: UpConv,      // decoder.0  256→128
    dec2: UpConv,      // decoder.2  128→64
    dec4: Conv2d,      // decoder.4  64→3,  k=3, pad=1
}

impl AotGanGenerator {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        // Encoder
        let enc1 = ReflectConv::load(&vb.pp("encoder").pp("1"), 4, 64, 7, 1, 1, 3)
            .context("encoder.1")?;
        let enc3 = candle_nn::conv2d(
            64,
            128,
            4,
            Conv2dConfig { stride: 2, padding: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None },
            vb.pp("encoder").pp("3"),
        )
        .context("encoder.3")?;
        let enc5 = candle_nn::conv2d(
            128,
            256,
            4,
            Conv2dConfig { stride: 2, padding: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None },
            vb.pp("encoder").pp("5"),
        )
        .context("encoder.5")?;

        // Middle — 8 AOT blocks
        let mut middle = Vec::with_capacity(8);
        for i in 0..8 {
            middle.push(AotBlock::load(vb, i).with_context(|| format!("AOT block {i}"))?);
        }

        // Decoder
        let dec0 = UpConv::load(&vb.pp("decoder").pp("0"), 256, 128).context("decoder.0")?;
        let dec2 = UpConv::load(&vb.pp("decoder").pp("2"), 128, 64).context("decoder.2")?;
        let dec4 = candle_nn::conv2d(
            64,
            3,
            3,
            Conv2dConfig { stride: 1, padding: 1, dilation: 1, groups: 1, cudnn_fwd_algo: None },
            vb.pp("decoder").pp("4"),
        )
        .context("decoder.4")?;

        Ok(Self { enc1, enc3, enc5, middle, dec0, dec2, dec4 })
    }

    /// `image` – (1, 3, H, W) f32 in [0, 1]
    /// `mask`  – (1, 1, H, W) f32, 1 = masked pixels
    pub fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Zero out masked pixels, then concatenate mask channel.
        let mask_inv = (mask.ones_like()? - mask)?;
        let masked = image.broadcast_mul(&mask_inv)?;
        let x = Tensor::cat(&[&masked, mask], 1)?; // [B, 4, H, W]

        // Encoder
        let x = self.enc1.forward(&x)?.relu()?;
        let x = self.enc3.forward(&x)?.relu()?;
        let x = self.enc5.forward(&x)?.relu()?;

        // Middle (8 AOT blocks)
        let mut x = x;
        for block in &self.middle {
            x = block.forward(&x)?;
        }

        // Decoder
        let x = self.dec0.forward(&x)?.relu()?;
        let x = self.dec2.forward(&x)?.relu()?;
        let x = self.dec4.forward(&x)?;

        // tanh → [−1, 1] → [0, 1]
        let out = ((x.tanh()? + 1.0f64)? / 2.0f64)?;

        // Composite: use network output only in masked region.
        let out = (out.broadcast_mul(mask)? + image.broadcast_mul(&mask_inv)?)?;
        Ok(out)
    }
}
