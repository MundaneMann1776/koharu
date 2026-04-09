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
    inpainting::{binarize_mask, extract_alpha},
};

use self::model::AotGanGenerator;

/// AOT-GAN Places2 checkpoint (NimaBoscarino/aot-gan-places2).
/// Flat state_dict — no key prefix.  Trained at 512×512.
const HF_REPO: &str = "NimaBoscarino/aot-gan-places2";
const WEIGHTS_FILE: &str = "pytorch_model.bin";

/// AOT-GAN was trained at 512×512.
const MODEL_SIZE: u32 = 512;

koharu_runtime::declare_hf_model_package!(
    id: "model:aot-gan:weights",
    repo: HF_REPO,
    file: WEIGHTS_FILE,
    bootstrap: false,
    order: 141,
);

pub struct AotGan {
    model: AotGanGenerator,
    device: Device,
}

impl AotGan {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let device = device(cpu)?;
        let weights_path = runtime
            .downloads()
            .huggingface_model(HF_REPO, WEIGHTS_FILE)
            .await?;

        // pytorch_model.bin is a HuggingFace PreTrainedModel flat state_dict.
        // candle's from_pth handles this format directly; no key prefix needed.
        let vb = candle_nn::VarBuilder::from_pth(&weights_path, DType::F32, &device)
            .context("failed to open pytorch_model.bin")?;

        let model = AotGanGenerator::load(&vb)
            .context("failed to load AOT-GAN weights")?;

        Ok(Self { model, device })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Result<DynamicImage> {
        if image.dimensions() != mask.dimensions() {
            bail!(
                "AOT-GAN: image and mask dimensions mismatch: image {:?}, mask {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }

        let binary_mask = binarize_mask(mask);
        let (orig_w, orig_h) = image.dimensions();

        // Scale to MODEL_SIZE × MODEL_SIZE (square, as the model was trained).
        let img_resized = resize(&image.to_rgb8(), MODEL_SIZE, MODEL_SIZE, FilterType::Lanczos3);
        let mask_resized = resize(
            &DynamicImage::ImageLuma8(binary_mask.clone()).to_luma8(),
            MODEL_SIZE,
            MODEL_SIZE,
            FilterType::Nearest,
        );

        let img_tensor = image_to_tensor(&img_resized, &self.device)?;
        let mask_tensor = mask_to_tensor(&mask_resized, &self.device)?;

        let output = self.model.forward(&img_tensor, &mask_tensor)?;

        let mut output_rgb = tensor_to_image(&output)?;
        // Scale back to original dimensions.
        if output_rgb.dimensions() != (orig_w, orig_h) {
            output_rgb = resize(&output_rgb, orig_w, orig_h, FilterType::Lanczos3);
        }

        let output_dyn = DynamicImage::ImageRgb8(output_rgb);
        if image.color().has_alpha() {
            let original_alpha = image.to_rgba8();
            let alpha = extract_alpha(&original_alpha);
            let output_rgba = crate::inpainting::restore_alpha_channel(
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
    let raw = img.as_raw();
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
    let data: Vec<f32> = mask
        .pixels()
        .map(|p| if p.0[0] > 127 { 1.0_f32 } else { 0.0 })
        .collect();
    Tensor::from_vec(data, (1usize, 1, h as usize, w as usize), device)?
        .to_dtype(DType::F32)
        .context("mask tensor conversion failed")
}

fn tensor_to_image(t: &Tensor) -> Result<RgbImage> {
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
