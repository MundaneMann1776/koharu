/// MI-GAN inpainting via vision.cpp subprocess.
///
/// MI-GAN (Mobile Inpainting GAN) is a 7.37M-parameter, 14.8 MB model
/// (GGUF fp16) that runs via Acly's vision.cpp CLI tool.
///
/// # Platform notes
/// - vision.cpp does NOT use Metal on macOS — inference runs on CPU.
///   Expect ~0.5–2s per image on M4 at 512×512.
/// - On Linux/Windows with Vulkan SDK, GPU acceleration is available.
///
/// # Setup (one-time)
/// 1. Build vision.cpp: `cd vision-cpp-helper && ./build.sh`
/// 2. Install: `sudo cp vision-cpp/build/bin/vision-cli /usr/local/bin/`
/// 3. The GGUF model is downloaded automatically on first use (14.8 MB).
///
/// # CLI contract (vision.cpp v0.3)
///   vision-cli migan -m <model.gguf> -i <image.png> <mask.png> -o <out.png>
///   Mask convention: white (255) = inpaint, black (0) = keep.
use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use koharu_runtime::RuntimeManager;
use std::process::Command;
use tracing::instrument;

use crate::inpainting::binarize_mask;

const HF_REPO: &str = "Acly/MIGAN-GGUF";
const MODEL_FILE: &str = "MIGAN-512-places2-F16.gguf";

koharu_runtime::declare_hf_model_package!(
    id: "model:migan:weights",
    repo: HF_REPO,
    file: MODEL_FILE,
    bootstrap: false,
    order: 142,
);

pub struct MiGan {
    model_path: std::path::PathBuf,
}

impl MiGan {
    pub async fn load(runtime: &RuntimeManager) -> Result<Self> {
        let model_path = runtime
            .downloads()
            .huggingface_model(HF_REPO, MODEL_FILE)
            .await?;
        Ok(Self { model_path })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Result<DynamicImage> {
        if image.dimensions() != mask.dimensions() {
            bail!(
                "MI-GAN: image and mask dimensions mismatch: image {:?}, mask {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }

        // Find vision-cli binary in the same locations as apple-vision-ocr-helper.
        let cli_path = find_vision_cli()?;

        // Write inputs to named temp files — vision-cli uses file paths, not stdin.
        // Use a random u64 (from address of a stack variable) combined with nanos
        // to avoid collisions between concurrent calls on the same process.
        let tmp_dir = std::env::temp_dir();
        let nonce = {
            let addr = &tmp_dir as *const _ as u64;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64)
                .unwrap_or(0);
            addr ^ ts
        };
        let img_path = tmp_dir.join(format!("koharu_migan_img_{nonce}.png"));
        let mask_path = tmp_dir.join(format!("koharu_migan_mask_{nonce}.png"));
        let out_path = tmp_dir.join(format!("koharu_migan_out_{nonce}.png"));

        // Save image as PNG. On mask-save failure, clean up the image file.
        image
            .to_rgb8()
            .save(&img_path)
            .context("failed to write temp image for MI-GAN")?;

        let binary = binarize_mask(mask);
        if let Err(e) = save_mask(&binary, &mask_path) {
            let _ = std::fs::remove_file(&img_path);
            return Err(e);
        }

        // Invoke vision-cli.
        let status = Command::new(&cli_path)
            .args([
                "migan",
                "-m",
                self.model_path.to_str().context("model path is not valid UTF-8")?,
                "-i",
                img_path.to_str().context("image path is not valid UTF-8")?,
                mask_path.to_str().context("mask path is not valid UTF-8")?,
                "-o",
                out_path.to_str().context("output path is not valid UTF-8")?,
            ])
            .env(
                "PATH",
                "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            )
            .status()
            .context("failed to spawn vision-cli; build it from vision-cpp-helper/build.sh and copy to /usr/local/bin/")?;

        // Clean up inputs regardless of success.
        let _ = std::fs::remove_file(&img_path);
        let _ = std::fs::remove_file(&mask_path);

        if !status.success() {
            let _ = std::fs::remove_file(&out_path);
            bail!(
                "vision-cli migan exited with {status}; \
                 ensure vision-cli is built and the model file is accessible"
            );
        }

        // Read the composited output PNG.
        let result = image::open(&out_path)
            .with_context(|| format!("failed to read MI-GAN output from {}", out_path.display()))?;
        let _ = std::fs::remove_file(&out_path);

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find `vision-cli` binary in standard locations.
fn find_vision_cli() -> Result<std::ffi::OsString> {
    let name = "vision-cli";
    let absolute_paths = [
        "/usr/local/bin/vision-cli",
        "/opt/homebrew/bin/vision-cli",
        "/opt/local/bin/vision-cli",
    ];

    for path in &absolute_paths {
        if std::path::Path::new(path).exists() {
            return Ok(std::ffi::OsString::from(path));
        }
    }

    // Walk up from current executable (covers target/debug/ and app bundle).
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        for _ in 0..8 {
            if let Some(d) = dir {
                let candidate = d.join(name);
                if candidate.exists() {
                    return Ok(candidate.into_os_string());
                }
                dir = d.parent();
            }
        }
    }

    bail!(
        "vision-cli not found. Build it:\n\
         cd vision-cpp-helper && ./build.sh\n\
         sudo cp vision-cpp/build/bin/vision-cli /usr/local/bin/"
    )
}

/// Save a binary mask as a greyscale PNG.
/// White pixels (255) mark the area to inpaint.
fn save_mask(mask: &GrayImage, path: &std::path::Path) -> Result<()> {
    // Ensure values are strictly 0 or 255 (binarize_mask may already do this).
    let clean: GrayImage = image::ImageBuffer::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([if mask.get_pixel(x, y).0[0] > 127 { 255u8 } else { 0 }])
    });
    clean.save(path).context("failed to write temp mask for MI-GAN")
}
