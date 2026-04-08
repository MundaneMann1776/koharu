mod model;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use image::{
    DynamicImage, GenericImageView, RgbImage,
    imageops::{FilterType, resize},
};
use koharu_runtime::RuntimeManager;
use tracing::instrument;

use crate::{
    device,
    inpainting::{binarize_mask, extract_alpha, restore_alpha_channel},
    loading,
};

use self::model::Mat as MatModel;

const HF_REPO: &str = "Sanster/MAT";
// The upstream checkpoint is `Places_512_FullData_G.pt` (PyTorch pickle).
// This constant names a safetensors conversion that does not yet exist upstream;
// users must convert it manually (see Mat::load error message below).
const WEIGHTS_FILE: &str = "Places_512_FullData_G.safetensors";

/// Longest side to which the input image is scaled before MAT inference.
/// MAT-512 was trained at 512×512.
const MAX_SIDE: u32 = 512;

koharu_runtime::declare_hf_model_package!(
    id: "model:mat:weights",
    repo: HF_REPO,
    file: WEIGHTS_FILE,
    bootstrap: false,
    order: 140,
);

pub struct Mat {
    model: MatModel,
    device: Device,
}

impl Mat {
    pub async fn load(_runtime: &RuntimeManager, _cpu: bool) -> Result<Self> {
        // Sanster/MAT only ships `Places_512_FullData_G.pt` (PyTorch pickle).
        // koharu's model loader requires safetensors. Convert first:
        //   python -c "from safetensors.torch import save_file; import torch; \
        //   ckpt = torch.load('Places_512_FullData_G.pt', map_location='cpu'); \
        //   save_file(ckpt, 'Places_512_FullData_G.safetensors')"
        // then place the output in your Koharu data dir under hf-hub/Sanster/MAT/
        //
        // Additionally, the MAT architecture implementation in model.rs is an
        // approximation; weight key names may not match until tested against the
        // actual checkpoint. Expect a descriptive load error if they differ.
        bail!(
            "MAT inpainting is not yet ready for use.\n\
             The upstream checkpoint (Sanster/MAT) is a PyTorch .pt file; \
             koharu requires safetensors.\n\
             Steps to enable:\n\
             1. Download Places_512_FullData_G.pt from \
             https://huggingface.co/Sanster/MAT\n\
             2. Convert: python -c \"from safetensors.torch import save_file; \
             import torch; save_file(torch.load('Places_512_FullData_G.pt', \
             map_location='cpu'), 'Places_512_FullData_G.safetensors')\"\n\
             3. Place the .safetensors file in your Koharu data dir at \
             hf-hub/Sanster/MAT/\n\
             This message will be removed once the weight key layout is confirmed."
        );

        #[allow(unreachable_code)]
        {
        let device = device(_cpu)?;
        let weights_path = _runtime
            .downloads()
            .huggingface_model(HF_REPO, WEIGHTS_FILE)
            .await?;
        let model = loading::load_buffered_safetensors_path(&weights_path, &device, |vb| {
            MatModel::load(&vb)
        })
        .context(
            "failed to load MAT weights — the checkpoint format may differ from expectations; \
             see koharu-ml/src/mat/model.rs for the expected key layout",
        )?;

        Ok(Self { model, device })
        } // end #[allow(unreachable_code)]
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Result<DynamicImage> {
        if image.dimensions() != mask.dimensions() {
            bail!(
                "MAT: image and mask dimensions mismatch: image {:?}, mask {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }

        let binary_mask = binarize_mask(mask);
        let (orig_w, orig_h) = image.dimensions();

        // Scale to at most MAX_SIDE on the longest dimension
        let scale = (MAX_SIDE as f32 / orig_w.max(orig_h) as f32).min(1.0);
        let proc_w = ((orig_w as f32 * scale) as u32).max(1) & !3; // align to 4
        let proc_h = ((orig_h as f32 * scale) as u32).max(1) & !3;

        let img_resized = resize(
            &image.to_rgb8(),
            proc_w,
            proc_h,
            FilterType::Lanczos3,
        );
        let mask_resized = resize(
            &DynamicImage::ImageLuma8(binary_mask.clone()).to_luma8(),
            proc_w,
            proc_h,
            FilterType::Nearest,
        );

        // Build tensors: (1, 3, H, W) and (1, 1, H, W), normalised to [0,1]
        let img_tensor = image_to_tensor(&img_resized, &self.device)?;
        let mask_tensor = mask_to_tensor(&mask_resized, &self.device)?;

        let output = self.model.forward(&img_tensor, &mask_tensor)?;

        // Decode output tensor → RgbImage, then up-scale back to original dims
        let mut output_rgb = tensor_to_image(&output)?;
        if output_rgb.dimensions() != (orig_w, orig_h) {
            output_rgb = resize(&output_rgb, orig_w, orig_h, FilterType::Lanczos3);
        }
        let output_dyn = DynamicImage::ImageRgb8(output_rgb);

        if image.color().has_alpha() {
            let original_alpha = image.to_rgba8();
            let alpha = extract_alpha(&original_alpha);
            let output_rgba = restore_alpha_channel(
                &output_dyn.to_rgb8(),
                &alpha,
                &binary_mask,
            );
            Ok(DynamicImage::ImageRgba8(output_rgba))
        } else {
            Ok(output_dyn)
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn image_to_tensor(img: &RgbImage, device: &Device) -> Result<Tensor> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw(); // [r0, g0, b0, r1, g1, b1, ...]
    // Separate into plane-per-channel for CHW layout
    let r: Vec<f32> = raw.iter().step_by(3).map(|&v| v as f32 / 255.0).collect();
    let g: Vec<f32> = raw.iter().skip(1).step_by(3).map(|&v| v as f32 / 255.0).collect();
    let b: Vec<f32> = raw.iter().skip(2).step_by(3).map(|&v| v as f32 / 255.0).collect();
    let data: Vec<f32> = r.into_iter().chain(g).chain(b).collect();
    Tensor::from_vec(data, (1usize, 3, h as usize, w as usize), device)?
        .to_dtype(DType::F32)
        .context("image tensor conversion failed")
}

fn mask_to_tensor(mask: &image::GrayImage, device: &Device) -> Result<Tensor> {
    let (w, h) = mask.dimensions();
    let data: Vec<f32> = mask.pixels().map(|p| if p.0[0] > 127 { 1.0_f32 } else { 0.0 }).collect();
    Tensor::from_vec(data, (1, 1, h as usize, w as usize), device)?
        .to_dtype(DType::F32)
        .context("mask tensor conversion failed")
}

fn tensor_to_image(t: &Tensor) -> Result<RgbImage> {
    // t: (1, 3, H, W) — clamp to [0, 1] then convert to u8
    let t = t.squeeze(0)?.permute([1, 2, 0])?.clamp(0.0, 1.0)?;
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let h = t.dim(0)?;
    let w = t.dim(1)?;
    let mut img = RgbImage::new(w as u32, h as u32);
    for (i, pixel) in img.pixels_mut().enumerate() {
        let base = i * 3;
        pixel.0 = [
            (data[base] * 255.0).round() as u8,
            (data[base + 1] * 255.0).round() as u8,
            (data[base + 2] * 255.0).round() as u8,
        ];
    }
    Ok(img)
}
