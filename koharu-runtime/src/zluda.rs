use std::ffi::c_void;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::{Result, anyhow, bail};

use crate::GpuBackend;
use crate::Runtime;
use crate::archive::{self, ArchiveKind, ExtractPolicy};
use crate::install::InstallState;
use crate::loader::load_library_by_path;
use crate::loader::{add_runtime_search_path, preload_library};

const RELEASE_BASE_URL: &str = "https://github.com/vosen/ZLUDA/releases/download";
const RELEASE_TAG: &str = "v6-preview.63";
const ZLUDA_ASSET_NAME: &str = "zluda-windows-e070320.zip";
const ZLUDA_DLLS: &[&str] = &[
    "nvcudart_hybrid64.dll",
    "nvcuda.dll",
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cufft64_12.dll",
];
const ZLUDA_ARCHIVE_ENTRIES: &[&str] = &[
    "zluda/nvcudart_hybrid64.dll",
    "zluda/nvcuda.dll",
    "zluda/cublasLt64_13.dll",
    "zluda/cublas64_13.dll",
    "zluda/cufft64_12.dll",
];
const HIP_SDK_DOWNLOAD_URL: &str =
    "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html";

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationProfile {
    CandleInference,
    Fft,
}

impl ValidationProfile {
    const fn requires_cufft(self) -> bool {
        matches!(self, Self::Fft)
    }
}

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CublasCreate = unsafe extern "C" fn(handle: *mut *mut c_void) -> i32;
type CublasDestroy = unsafe extern "C" fn(handle: *mut c_void) -> i32;
type CublasLtCreate = unsafe extern "C" fn(handle: *mut *mut c_void) -> i32;
type CublasLtDestroy = unsafe extern "C" fn(handle: *mut c_void) -> i32;
type CufftCreate = unsafe extern "C" fn(handle: *mut i32) -> i32;
type CufftDestroy = unsafe extern "C" fn(handle: i32) -> i32;

pub(crate) fn package_enabled(runtime: &Runtime) -> bool {
    runtime.wants_gpu() && runtime.gpu_backend() == GpuBackend::CudaZluda
}

pub(crate) fn package_present(runtime: &Runtime) -> Result<bool> {
    let install_dir = install_dir(runtime);
    let source_id = source_id();
    let install = InstallState::new(&install_dir, &source_id);
    if !install.is_current() {
        return Ok(false);
    }

    Ok(ZLUDA_DLLS
        .iter()
        .all(|library| install_dir.join(library).exists()))
}

pub(crate) async fn package_prepare(runtime: &Runtime) -> Result<()> {
    if let Err(err) = ensure_ready(runtime).await {
        tracing::warn!(
            "ZLUDA runtime is unavailable: {err:#}; falling back to CPU for unsupported Candle models."
        );
    }
    Ok(())
}

pub(crate) async fn ensure_ready(runtime: &Runtime) -> Result<()> {
    ensure_ready_with_profile(runtime, ValidationProfile::CandleInference).await
}

async fn ensure_ready_with_profile(runtime: &Runtime, profile: ValidationProfile) -> Result<()> {
    let install_dir = install_dir(runtime);
    let source_id = source_id();
    let install = InstallState::new(&install_dir, &source_id);

    if !install.is_current() {
        install.reset()?;

        let url = format!("{RELEASE_BASE_URL}/{RELEASE_TAG}/{ZLUDA_ASSET_NAME}");
        let archive = runtime
            .downloads()
            .cached_download(&url, ZLUDA_ASSET_NAME)
            .await
            .with_context(|| format!("failed to download `{url}`"))?;
        archive::extract(
            &archive,
            &install_dir,
            ArchiveKind::Zip,
            ExtractPolicy::SelectedPaths(ZLUDA_ARCHIVE_ENTRIES),
        )?;
        install.commit()?;
    }

    let hip_bin = detect_hip_bin_dir().ok_or_else(hip_sdk_missing_error)?;
    export_hip_path(&hip_bin);
    add_runtime_search_path(&hip_bin)?;
    add_runtime_search_path(&install_dir)?;
    validate_runtime(&install_dir, profile)?;

    for library in ZLUDA_DLLS {
        preload_library(&install_dir.join(library))?;
    }

    Ok(())
}

pub(crate) fn backend_status(runtime: &Runtime) -> Result<()> {
    backend_status_with_profile(runtime, ValidationProfile::CandleInference)
}

pub(crate) fn candidate_status(runtime: &Runtime) -> Result<()> {
    if !runtime.wants_gpu() {
        bail!("GPU acceleration is disabled by compute policy");
    }
    detect_hip_bin_dir().ok_or_else(hip_sdk_missing_error)?;
    Ok(())
}

fn backend_status_with_profile(runtime: &Runtime, profile: ValidationProfile) -> Result<()> {
    candidate_status(runtime)?;

    let hip_bin = detect_hip_bin_dir().ok_or_else(hip_sdk_missing_error)?;
    export_hip_path(&hip_bin);
    if !package_present(runtime)? {
        bail!(
            "managed ZLUDA runtime is not installed in `{}`",
            install_dir(runtime).display()
        );
    }

    add_runtime_search_path(&hip_bin)?;
    add_runtime_search_path(&install_dir(runtime))?;
    validate_runtime(&install_dir(runtime), profile)
}

fn validate_runtime(install_dir: &Path, profile: ValidationProfile) -> Result<()> {
    validate_cuda_driver(&install_dir.join("nvcuda.dll"))?;
    validate_cublas(&install_dir.join("cublas64_13.dll"))?;
    validate_cublas_lt(&install_dir.join("cublasLt64_13.dll"))?;
    if profile.requires_cufft() {
        validate_cufft(&install_dir.join("cufft64_12.dll"))?;
    }
    Ok(())
}

fn validate_cuda_driver(path: &Path) -> Result<()> {
    let library = load_library_by_path(path)?;
    let cu_init = unsafe { *library.get::<CuInit>(b"cuInit\0")? };
    let status = unsafe { cu_init(0) };
    if status != 0 {
        bail!(
            "`{}` failed cuInit with error code {status}",
            path.display()
        );
    }
    Ok(())
}

fn validate_cublas(path: &Path) -> Result<()> {
    let library = load_library_by_path(path)?;
    let create = unsafe { *library.get::<CublasCreate>(b"cublasCreate_v2\0")? };
    let destroy = unsafe { *library.get::<CublasDestroy>(b"cublasDestroy_v2\0")? };
    let mut handle = std::ptr::null_mut();
    let status = unsafe { create(&mut handle) };
    if status != 0 {
        bail!(
            "`{}` failed cublasCreate_v2 with error code {status}",
            path.display()
        );
    }
    let status = unsafe { destroy(handle) };
    if status != 0 {
        bail!(
            "`{}` failed cublasDestroy_v2 with error code {status}",
            path.display()
        );
    }
    Ok(())
}

fn validate_cublas_lt(path: &Path) -> Result<()> {
    let library = load_library_by_path(path)?;
    let create = unsafe { *library.get::<CublasLtCreate>(b"cublasLtCreate\0")? };
    let destroy = unsafe { *library.get::<CublasLtDestroy>(b"cublasLtDestroy\0")? };
    let mut handle = std::ptr::null_mut();
    let status = unsafe { create(&mut handle) };
    if status != 0 {
        bail!(
            "`{}` failed cublasLtCreate with error code {status}",
            path.display()
        );
    }
    let status = unsafe { destroy(handle) };
    if status != 0 {
        bail!(
            "`{}` failed cublasLtDestroy with error code {status}",
            path.display()
        );
    }
    Ok(())
}

fn validate_cufft(path: &Path) -> Result<()> {
    let library = load_library_by_path(path)?;
    let create = unsafe { *library.get::<CufftCreate>(b"cufftCreate\0")? };
    let destroy = unsafe { *library.get::<CufftDestroy>(b"cufftDestroy\0")? };
    let mut handle = 0_i32;
    let status = unsafe { create(&mut handle) };
    if status != 0 {
        bail!(
            "`{}` failed cufftCreate with error code {status}",
            path.display()
        );
    }
    let status = unsafe { destroy(handle) };
    if status != 0 {
        bail!(
            "`{}` failed cufftDestroy with error code {status}",
            path.display()
        );
    }
    Ok(())
}

fn detect_hip_bin_dir() -> Option<PathBuf> {
    hip_bin_candidates().into_iter().find(|candidate| {
        candidate.join("amdhip64_7.dll").exists() || candidate.join("amdhip64_6.dll").exists()
    })
}

fn hip_sdk_missing_error() -> anyhow::Error {
    anyhow!(
        "HIP SDK not found. Set `HIP_PATH` or install the AMD HIP SDK from `{HIP_SDK_DOWNLOAD_URL}`."
    )
}

fn export_hip_path(hip_bin: &Path) {
    if let Some(root) = hip_bin.parent() {
        unsafe {
            std::env::set_var("HIP_PATH", root);
        }
    }
}

fn hip_bin_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(path) = std::env::var_os("HIP_PATH") {
        candidates.push(PathBuf::from(path).join("bin"));
    }

    candidates.push(PathBuf::from(r"C:\hip_sdk\bin"));

    let rocm_root = PathBuf::from(r"C:\Program Files\AMD\ROCm");
    candidates.push(rocm_root.join("bin"));
    if let Ok(entries) = std::fs::read_dir(&rocm_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                candidates.push(path.join("bin"));
            }
        }
    }

    candidates
}

fn install_dir(runtime: &Runtime) -> PathBuf {
    runtime.root().join("runtime").join("zluda")
}

fn source_id() -> String {
    format!("zluda;tag={RELEASE_TAG};asset={ZLUDA_ASSET_NAME}")
}

crate::declare_native_package!(
    id: "runtime:zluda",
    bootstrap: true,
    order: 11,
    enabled: crate::zluda::package_enabled,
    present: crate::zluda::package_present,
    prepare: crate::zluda::package_prepare,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_id_mentions_release_asset() {
        let id = source_id();
        assert!(id.contains("zluda"));
        assert!(id.contains(ZLUDA_ASSET_NAME));
    }

    #[test]
    fn required_runtime_dlls_cover_zluda_cuda_entrypoints() {
        assert!(ZLUDA_DLLS.contains(&"nvcuda.dll"));
        assert!(ZLUDA_DLLS.contains(&"cublas64_13.dll"));
        assert!(ZLUDA_DLLS.contains(&"cublasLt64_13.dll"));
        assert!(ZLUDA_DLLS.contains(&"cufft64_12.dll"));
    }

    #[test]
    fn preload_order_keeps_driver_before_math_libraries() {
        let nvcuda_index = ZLUDA_DLLS
            .iter()
            .position(|dll| *dll == "nvcuda.dll")
            .unwrap();
        let cublas_index = ZLUDA_DLLS
            .iter()
            .position(|dll| *dll == "cublas64_13.dll")
            .unwrap();
        let cufft_index = ZLUDA_DLLS
            .iter()
            .position(|dll| *dll == "cufft64_12.dll")
            .unwrap();
        assert!(nvcuda_index < cublas_index);
        assert!(nvcuda_index < cufft_index);
    }

    #[test]
    fn candle_inference_profile_does_not_require_fft() {
        assert!(!ValidationProfile::CandleInference.requires_cufft());
    }

    #[test]
    fn fft_profile_requires_fft() {
        assert!(ValidationProfile::Fft.requires_cufft());
    }

    #[test]
    fn archive_entries_only_extract_primary_runtime_dlls() {
        assert_eq!(ZLUDA_ARCHIVE_ENTRIES.len(), ZLUDA_DLLS.len());
        assert!(
            ZLUDA_ARCHIVE_ENTRIES
                .iter()
                .all(|entry| entry.starts_with("zluda/"))
        );
        assert!(
            ZLUDA_ARCHIVE_ENTRIES
                .iter()
                .all(|entry| !entry.contains("/trace/"))
        );
    }
}
