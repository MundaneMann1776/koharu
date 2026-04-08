/// Apple Vision OCR — macOS-only, zero-download, Neural Engine accelerated.
///
/// Delegates to the `apple-vision-ocr-helper` Swift CLI binary via subprocess.
/// The binary must be compiled and placed in the same directory as the Koharu
/// executable, or on `PATH`.  It is built from
/// `apple-vision-ocr-helper/` in the repository root.
///
/// Supported languages: Korean (ko-KR), Japanese (ja-JP), Chinese (zh-Hans,
/// zh-Hant), English, and any other language supported by the installed
/// macOS version's Vision framework.
///
/// Platform gate: this module is only compiled on macOS (`target_os = "macos"`).
/// On other platforms, `AppleVisionOcr::available()` returns false and
/// `load()` returns an error.
use anyhow::{Context, Result, bail};
use image::DynamicImage;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Output JSON format from the Swift helper
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HelperOutput {
    text: Option<String>,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct AppleVisionOcr {
    /// Requested language codes passed to VNRecognizeTextRequest.
    /// Empty = let Vision auto-detect.
    pub languages: Vec<String>,
}

impl AppleVisionOcr {
    /// Returns `true` only on macOS.
    pub fn available() -> bool {
        cfg!(target_os = "macos")
    }

    pub fn new(languages: Vec<String>) -> Result<Self> {
        if !Self::available() {
            bail!("Apple Vision OCR is only available on macOS");
        }
        Ok(Self { languages })
    }

    /// Run OCR on a single image.  Returns the recognised text.
    pub fn recognize(&self, image: &DynamicImage) -> Result<String> {
        #[cfg(not(target_os = "macos"))]
        {
            bail!("Apple Vision OCR is only available on macOS");
        }

        #[cfg(target_os = "macos")]
        {
            self.recognize_macos(image)
        }
    }

    #[cfg(target_os = "macos")]
    fn recognize_macos(&self, image: &DynamicImage) -> Result<String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Encode image as PNG in memory
        let mut png_bytes: Vec<u8> = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .context("failed to encode image as PNG for Apple Vision OCR")?;

        // Build subprocess command
        let mut cmd = Command::new("apple-vision-ocr-helper");
        for lang in &self.languages {
            cmd.arg("--lang").arg(lang);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = cmd
            .spawn()
            .context("failed to spawn apple-vision-ocr-helper; ensure it is on PATH or in the same directory as koharu")?;

        // Write PNG to helper stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&png_bytes)
                .context("failed to write image to apple-vision-ocr-helper stdin")?;
        }

        let output = child
            .wait_with_output()
            .context("apple-vision-ocr-helper process failed")?;

        let json = std::str::from_utf8(&output.stdout)
            .context("apple-vision-ocr-helper produced non-UTF-8 output")?;

        let result: HelperOutput = serde_json::from_str(json.trim())
            .with_context(|| format!("failed to parse apple-vision-ocr-helper output: {json}"))?;

        if let Some(err) = result.error {
            bail!("Apple Vision OCR error: {err}");
        }

        Ok(result.text.unwrap_or_default())
    }
}
