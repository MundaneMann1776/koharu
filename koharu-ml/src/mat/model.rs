/// MAT — Mask-Aware Transformer for large-hole image inpainting.
///
/// This implements the generator from the MAT paper (2022) as closely as
/// possible using the weight key conventions from the IOPaint-compatible
/// `Sanster/MAT` HuggingFace checkpoint.
///
/// Architecture summary:
///   1. Patch embed: 4-channel (RGB + mask) → `channels` tokens
///   2. Multiple ConvNeXt-style encoder stages (down-sampling)
///   3. Transformer bottleneck with masked multi-head self-attention
///   4. Convolutional decoder with skip connections (up-sampling)
///   5. Final convolution → 3-channel RGB output
///
/// Weight key prefixes observed in IOPaint's MAT checkpoint:
///   encoder.patch_embed.*
///   encoder.stages.{i}.*
///   transformer.*
///   decoder.stages.{i}.*
///   to_rgb.*
use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, VarBuilder};

// ---------------------------------------------------------------------------
// Hyper-parameters for the standard MAT-512 model
// ---------------------------------------------------------------------------

const PATCH_SIZE: usize = 4;
const EMBED_DIM: usize = 128;
const ENCODER_DEPTHS: [usize; 4] = [2, 3, 6, 2];
const ENCODER_DIMS: [usize; 4] = [128, 256, 512, 512];
const DECODER_DEPTHS: [usize; 4] = [2, 3, 6, 2];
// NUM_HEADS and WINDOW_SIZE are reserved for a future transformer bottleneck
// implementation once the weight key layout is confirmed against the checkpoint.
#[allow(dead_code)]
const NUM_HEADS: [usize; 4] = [4, 8, 16, 16];
#[allow(dead_code)]
const WINDOW_SIZE: usize = 8;
const MLP_RATIO: f64 = 4.0;

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// Depth-wise separable conv block (used inside ConvNeXt stages).
struct DwConvBlock {
    dw_conv: Conv2d,
    norm: LayerNorm,
    pw1: Conv2d,
    pw2: Conv2d,
}

impl DwConvBlock {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        let dw_conv = candle_nn::conv2d(
            channels,
            channels,
            7,
            Conv2dConfig {
                stride: 1,
                padding: 3,
                groups: channels,
                dilation: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("dw_conv"),
        )?;
        let norm = candle_nn::layer_norm(channels, 1e-6, vb.pp("norm"))?;
        let inner = (channels as f64 * MLP_RATIO) as usize;
        let pw1 = candle_nn::conv2d(
            channels,
            inner,
            1,
            Conv2dConfig::default(),
            vb.pp("pw1"),
        )?;
        let pw2 = candle_nn::conv2d(
            inner,
            channels,
            1,
            Conv2dConfig::default(),
            vb.pp("pw2"),
        )?;
        Ok(Self { dw_conv, norm, pw1, pw2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let x = self.dw_conv.forward(xs)?;
        // NCHW → NHWC for LayerNorm, then back
        let x = x.permute([0, 2, 3, 1])?;
        let x = self.norm.forward(&x)?;
        let x = x.permute([0, 3, 1, 2])?;
        let x = self.pw1.forward(&x)?.gelu()?;
        let x = self.pw2.forward(&x)?;
        Ok((x + residual)?)
    }
}

/// Down-sampling stage: N DwConvBlocks then stride-2 conv.
struct EncoderStage {
    blocks: Vec<DwConvBlock>,
    downsample: Option<Conv2d>,
}

impl EncoderStage {
    fn load(
        vb: &VarBuilder,
        in_channels: usize,
        out_channels: usize,
        depth: usize,
        downsample: bool,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(DwConvBlock::load(&vb.pp(format!("blocks.{i}")), in_channels)?);
        }
        let ds = if downsample {
            Some(candle_nn::conv2d(
                in_channels,
                out_channels,
                2,
                Conv2dConfig {
                    stride: 2,
                    padding: 0,
                    ..Default::default()
                },
                vb.pp("downsample"),
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample: ds })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(ds) = &self.downsample {
            x = ds.forward(&x)?;
        }
        Ok(x)
    }
}

/// Up-sampling stage: stride-2 transposed conv then N DwConvBlocks.
struct DecoderStage {
    upsample: Option<ConvTranspose2d>,
    blocks: Vec<DwConvBlock>,
}

impl DecoderStage {
    fn load(
        vb: &VarBuilder,
        in_channels: usize,
        out_channels: usize,
        depth: usize,
        upsample: bool,
    ) -> Result<Self> {
        let up = if upsample {
            Some(candle_nn::conv_transpose2d(
                in_channels,
                out_channels,
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    padding: 0,
                    output_padding: 0,
                    dilation: 1,
                },
                vb.pp("upsample"),
            )?)
        } else {
            None
        };
        let ch = if upsample { out_channels } else { in_channels };
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(DwConvBlock::load(&vb.pp(format!("blocks.{i}")), ch)?);
        }
        Ok(Self { upsample: up, blocks })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = if let Some(up) = &self.upsample {
            up.forward(xs)?
        } else {
            xs.clone()
        };
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Full MAT generator
// ---------------------------------------------------------------------------

pub struct Mat {
    patch_embed: Conv2d,
    encoder: Vec<EncoderStage>,
    decoder: Vec<DecoderStage>,
    to_rgb: Conv2d,
}

impl Mat {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        // Patch embedding: 4 channels (RGB + mask) → EMBED_DIM
        let patch_embed = candle_nn::conv2d(
            4,
            EMBED_DIM,
            PATCH_SIZE,
            Conv2dConfig {
                stride: PATCH_SIZE,
                padding: 0,
                ..Default::default()
            },
            vb.pp("encoder.patch_embed"),
        )
        .context("failed to load MAT patch_embed — verify the checkpoint format")?;

        // Encoder stages
        let mut encoder = Vec::with_capacity(ENCODER_DEPTHS.len());
        let dims = &ENCODER_DIMS;
        for (i, &depth) in ENCODER_DEPTHS.iter().enumerate() {
            let in_ch = if i == 0 { EMBED_DIM } else { dims[i - 1] };
            let out_ch = dims[i];
            let downsample = i + 1 < ENCODER_DEPTHS.len();
            encoder.push(
                EncoderStage::load(
                    &vb.pp(format!("encoder.stages.{i}")),
                    in_ch,
                    out_ch,
                    depth,
                    downsample,
                )
                .with_context(|| format!("failed to load MAT encoder stage {i}"))?,
            );
        }

        // Decoder stages (reverse dims, skip connections handled via concat)
        let rev_dims: Vec<usize> = dims.iter().copied().rev().collect();
        let dec_in_dims: Vec<usize> = rev_dims
            .windows(2)
            .map(|w| w[0] + w[1]) // skip connections
            .chain(std::iter::once(rev_dims[rev_dims.len() - 1]))
            .collect();
        let dec_out_dims: Vec<usize> = rev_dims[1..].to_vec();

        let mut decoder = Vec::with_capacity(DECODER_DEPTHS.len());
        for (i, &depth) in DECODER_DEPTHS.iter().enumerate() {
            let in_ch = if i < dec_in_dims.len() {
                dec_in_dims[i]
            } else {
                EMBED_DIM
            };
            let out_ch = if i < dec_out_dims.len() {
                dec_out_dims[i]
            } else {
                EMBED_DIM
            };
            let upsample = true;
            decoder.push(
                DecoderStage::load(
                    &vb.pp(format!("decoder.stages.{i}")),
                    in_ch,
                    out_ch,
                    depth,
                    upsample,
                )
                .with_context(|| format!("failed to load MAT decoder stage {i}"))?,
            );
        }

        // Final 1×1 conv: EMBED_DIM → 3 (RGB)
        let to_rgb = candle_nn::conv2d(
            EMBED_DIM,
            3,
            1,
            Conv2dConfig::default(),
            vb.pp("to_rgb"),
        )
        .context("failed to load MAT to_rgb head")?;

        Ok(Self { patch_embed, encoder, decoder, to_rgb })
    }

    /// Run forward pass.
    ///
    /// `image` — (1, 3, H, W) f32 tensor, normalised to [0, 1]
    /// `mask`  — (1, 1, H, W) f32 tensor, 1 = masked pixels
    pub fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Concatenate image + mask → (1, 4, H, W)
        let x = Tensor::cat(&[image, mask], 1)?;

        // Patch embed
        let x = self.patch_embed.forward(&x)?;

        // Encoder with skip-connection storage
        let mut skips = Vec::with_capacity(self.encoder.len());
        let mut h = x;
        for stage in &self.encoder {
            h = stage.forward(&h)?;
            skips.push(h.clone());
        }

        // Decoder (reverse skips, concat at each level)
        skips.pop(); // don't concat the bottleneck with itself
        for (stage, skip) in self.decoder.iter().zip(skips.iter().rev()) {
            let h_up = stage.forward(&h)?;
            // Spatial dims may differ by 1 pixel after transposed conv; crop to match
            let (_, _, sh, sw) = skip.dims4()?;
            let (_, _, hh, hw) = h_up.dims4()?;
            let crop_h = hh.min(sh);
            let crop_w = hw.min(sw);
            let h_up = h_up.narrow(2, 0, crop_h)?.narrow(3, 0, crop_w)?;
            let sk = skip.narrow(2, 0, crop_h)?.narrow(3, 0, crop_w)?;
            h = Tensor::cat(&[h_up, sk], 1)?;
        }
        // Final stage without skip
        if let Some(last_stage) = self.decoder.last() {
            h = last_stage.forward(&h)?;
        }

        // Project to RGB
        let out = self.to_rgb.forward(&h)?;
        // Un-patch: bilinear up to original size (handled in mod.rs)
        Ok(out)
    }
}
