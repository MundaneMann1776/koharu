mod hf_hub;
mod inpainting;

pub mod aot_inpainting;
pub mod comic_text_bubble_detector;
pub mod comic_text_detector;
pub mod font_detector;
pub mod lama;
pub mod loading;
pub mod manga_ocr;
pub mod manga_text_segmentation_2025;
pub mod mit48px_ocr;
pub mod paddleocr_vl;
pub mod pp_doclayout_v3;
pub mod probability_map;
pub mod speech_bubble_segmentation;

use std::{future::Future, path::PathBuf};

use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};

pub use candle_core::Device;
use koharu_runtime::{GpuBackend, RuntimeManager};

pub fn device(runtime: &RuntimeManager, cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }

    match runtime.gpu_backend() {
        GpuBackend::CudaNvidia | GpuBackend::CudaZluda => Ok(Device::new_cuda(0)?),
        GpuBackend::Metal => Ok(Device::new_metal(0)?),
        GpuBackend::Cpu => Ok(Device::Cpu),
    }
}

pub fn device_from_env(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() && koharu_runtime::check_cuda_driver_support() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn resolve_device(runtime: Option<&RuntimeManager>, cpu: bool) -> Result<Device> {
    match runtime {
        Some(runtime) => device(runtime, cpu),
        None => device_from_env(cpu),
    }
}

pub fn load_with_device<T>(
    runtime: Option<&RuntimeManager>,
    cpu: bool,
    load: impl FnOnce(Device) -> Result<T>,
) -> Result<T> {
    let device = resolve_device(runtime, cpu)?;
    load(device)
}

pub async fn load_runtime_model<T, Files, Resolve, Load>(
    runtime: &RuntimeManager,
    cpu: bool,
    resolve: Resolve,
    load: Load,
) -> Result<T>
where
    Resolve: Future<Output = Result<Files>>,
    Load: FnOnce(Files, Device) -> Result<T>,
{
    let device = resolve_device(Some(runtime), cpu)?;
    let files = resolve.await?;
    load(files, device)
}

pub async fn download_huggingface_files<const N: usize>(
    runtime: &RuntimeManager,
    repo: &str,
    files: [&str; N],
) -> Result<[PathBuf; N]> {
    let downloads = runtime.downloads();
    let mut paths = Vec::with_capacity(N);
    for file in files {
        paths.push(downloads.huggingface_model(repo, file).await?);
    }
    Ok(paths.try_into().unwrap_or_else(|_| unreachable!()))
}
