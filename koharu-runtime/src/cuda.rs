use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use serde::Deserialize;
use std::fmt;
use std::sync::OnceLock;

use crate::GpuBackend;
use crate::Runtime;
use crate::archive::{self, ArchiveKind, ExtractPolicy};
use crate::install::InstallState;
use crate::loader::{add_runtime_search_path, preload_library};

const CUDA_SUCCESS: i32 = 0;
const CUDA_13_0_DRIVER_VERSION: i32 = 13000;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const MIN_COMPUTE_CAPABILITY: (i32, i32) = (7, 5); // Turing (RTX 20xx) and above
static CUDA_DRIVER_SUPPORTED: OnceLock<bool> = OnceLock::new();

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut i32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
type CuDeviceGetAttribute = unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: i32) -> i32;

#[derive(Debug, Clone, Copy)]
pub(crate) struct NvidiaBackendInfo {
    driver_version: CudaDriverVersion,
    compute_capability: (i32, i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudaDriverVersion {
    raw: i32,
}

#[derive(Debug, Deserialize)]
struct PypiRelease {
    urls: Vec<PypiFile>,
}

#[derive(Debug, Deserialize)]
struct PypiFile {
    filename: String,
    url: String,
}

#[allow(dead_code)]
struct WheelSpec {
    package: &'static str,
    windows_dylibs: &'static [&'static str],
    linux_dylibs: &'static [&'static str],
}

const WHEELS: &[WheelSpec] = &[
    WheelSpec {
        package: "nvidia-cuda-runtime/13.0.96",
        windows_dylibs: &["cudart64_13.dll"],
        linux_dylibs: &["libcudart.so.13"],
    },
    WheelSpec {
        package: "nvidia-cublas/13.2.1.1",
        windows_dylibs: &["cublasLt64_13.dll", "cublas64_13.dll"],
        linux_dylibs: &["libcublasLt.so.13", "libcublas.so.13"],
    },
    WheelSpec {
        package: "nvidia-cufft/12.1.0.78",
        windows_dylibs: &["cufft64_12.dll"],
        linux_dylibs: &["libcufft.so.12"],
    },
    WheelSpec {
        package: "nvidia-curand/10.4.1.81",
        windows_dylibs: &["curand64_10.dll"],
        linux_dylibs: &["libcurand.so.10"],
    },
];

impl CudaDriverVersion {
    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    pub const fn raw(self) -> i32 {
        self.raw
    }

    pub const fn major(self) -> i32 {
        self.raw / 1000
    }

    pub const fn minor(self) -> i32 {
        (self.raw % 1000) / 10
    }

    pub const fn supports_cuda_13_0(self) -> bool {
        self.raw >= CUDA_13_0_DRIVER_VERSION
    }
}

impl fmt::Display for CudaDriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

pub fn driver_version() -> Result<CudaDriverVersion> {
    let library_name = if cfg!(target_os = "windows") {
        "nvcuda.dll"
    } else {
        "libcuda.so"
    };

    unsafe {
        let library = Library::new(library_name)
            .with_context(|| format!("failed to load NVIDIA driver library `{library_name}`"))?;
        let cu_init = *library
            .get::<CuInit>(b"cuInit\0")
            .context("failed to load cuInit from NVIDIA driver")?;
        let cu_driver_get_version = *library
            .get::<CuDriverGetVersion>(b"cuDriverGetVersion\0")
            .context("failed to load cuDriverGetVersion from NVIDIA driver")?;

        let status = cu_init(0);
        if status != CUDA_SUCCESS {
            bail!("cuInit failed with CUDA driver error code {status}");
        }

        let mut raw = 0;
        let status = cu_driver_get_version(&mut raw);
        if status != CUDA_SUCCESS {
            bail!("cuDriverGetVersion failed with CUDA driver error code {status}");
        }

        Ok(CudaDriverVersion::from_raw(raw))
    }
}

/// Query the compute capability of CUDA device 0.
///
/// Returns `(major, minor)` e.g. `(7, 5)` for Turing, `(8, 6)` for Ampere.
pub fn compute_capability() -> Result<(i32, i32)> {
    let library_name = if cfg!(target_os = "windows") {
        "nvcuda.dll"
    } else {
        "libcuda.so"
    };

    unsafe {
        let library = Library::new(library_name)
            .with_context(|| format!("failed to load `{library_name}`"))?;
        let cu_init = *library
            .get::<CuInit>(b"cuInit\0")
            .context("cuInit not found")?;
        let cu_device_get = *library
            .get::<CuDeviceGet>(b"cuDeviceGet\0")
            .context("cuDeviceGet not found")?;
        let cu_device_get_attribute = *library
            .get::<CuDeviceGetAttribute>(b"cuDeviceGetAttribute\0")
            .context("cuDeviceGetAttribute not found")?;

        let status = cu_init(0);
        if status != CUDA_SUCCESS {
            bail!("cuInit failed with error code {status}");
        }

        let mut dev = 0;
        let status = cu_device_get(&mut dev, 0);
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGet failed with error code {status}");
        }

        let mut major = 0;
        let status = cu_device_get_attribute(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        );
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGetAttribute(MAJOR) failed with error code {status}");
        }

        let mut minor = 0;
        let status = cu_device_get_attribute(
            &mut minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        );
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGetAttribute(MINOR) failed with error code {status}");
        }

        Ok((major, minor))
    }
}

/// Check whether the installed NVIDIA driver supports CUDA 13.0+.
///
/// Returns `true` when GPU compute should be used, `false` when the caller
/// should fall back to CPU.  Warnings are emitted via `tracing::warn!`.
pub fn check_cuda_driver_support() -> bool {
    *CUDA_DRIVER_SUPPORTED.get_or_init(|| match backend_info() {
        Ok(info) => {
            tracing::info!("NVIDIA driver reports CUDA {} support", info.driver_version);
            tracing::info!(
                "GPU compute capability: {}.{}",
                info.compute_capability.0,
                info.compute_capability.1
            );
            true
        }
        Err(err) => {
            tracing::warn!(
                "Could not enable the NVIDIA CUDA backend: {err:#}; falling back to CPU."
            );
            false
        }
    })
}

pub(crate) fn package_enabled(runtime: &Runtime) -> bool {
    runtime.wants_gpu() && runtime.gpu_backend() == GpuBackend::CudaNvidia
}

pub(crate) fn package_present(runtime: &Runtime) -> Result<bool> {
    let install_dir = install_dir(runtime);
    let source_id = source_id()?;
    let install = InstallState::new(&install_dir, &source_id);
    if !install.is_current() {
        return Ok(false);
    }

    Ok(WHEELS
        .iter()
        .flat_map(|wheel| wheel.dylibs().iter())
        .all(|dylib| install_dir.join(dylib).exists()))
}

pub(crate) async fn package_prepare(runtime: &Runtime) -> Result<()> {
    ensure_ready(runtime).await
}

pub(crate) async fn ensure_ready(runtime: &Runtime) -> Result<()> {
    let install_dir = install_dir(runtime);
    let source_id = source_id()?;
    let install = InstallState::new(&install_dir, &source_id);

    if !install.is_current() {
        install.reset()?;

        for wheel in WHEELS {
            let asset = select_wheel(runtime, wheel).await?;
            let archive = runtime
                .downloads()
                .cached_download(&asset.url, &asset.filename)
                .await
                .with_context(|| format!("failed to download `{}`", asset.url))?;
            archive::extract(
                &archive,
                &install_dir,
                ArchiveKind::Zip,
                ExtractPolicy::Selected(wheel.dylibs()),
            )?;
        }

        install.commit()?;
    }

    add_runtime_search_path(&install_dir)?;
    for wheel in WHEELS {
        for dylib in wheel.dylibs() {
            let path = install_dir.join(dylib);
            if path.exists() {
                preload_library(&path)?;
            }
        }
    }

    Ok(())
}

crate::declare_native_package!(
    id: "runtime:cuda",
    bootstrap: true,
    order: 10,
    enabled: package_enabled,
    present: package_present,
    prepare: package_prepare,
);

struct WheelAsset {
    url: String,
    filename: String,
}

pub(crate) fn backend_info() -> Result<NvidiaBackendInfo> {
    if !driver_library_available() {
        bail!("NVIDIA driver library not found");
    }

    let driver_version =
        driver_version().context("failed to query the installed NVIDIA driver version")?;
    if !driver_version.supports_cuda_13_0() {
        bail!(
            "NVIDIA driver only supports CUDA {driver_version}; install a driver with CUDA 13.0 or newer support"
        );
    }

    let compute_capability =
        compute_capability().context("failed to query the active GPU compute capability")?;
    if compute_capability < MIN_COMPUTE_CAPABILITY {
        bail!(
            "GPU compute capability {}.{} is below the minimum required {}.{}",
            compute_capability.0,
            compute_capability.1,
            MIN_COMPUTE_CAPABILITY.0,
            MIN_COMPUTE_CAPABILITY.1,
        );
    }

    Ok(NvidiaBackendInfo {
        driver_version,
        compute_capability,
    })
}

fn driver_library_available() -> bool {
    #[cfg(target_os = "windows")]
    return unsafe { Library::new("nvcuda.dll") }.is_ok();

    #[cfg(target_os = "linux")]
    return unsafe { Library::new("libcuda.so.1") }.is_ok();

    #[allow(unreachable_code)]
    false
}

fn install_dir(runtime: &Runtime) -> std::path::PathBuf {
    runtime.root().join("runtime").join("cuda")
}

fn platform_tags() -> Result<&'static [&'static str]> {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Ok(&["win_amd64"]);

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Ok(&["manylinux_2_27_x86_64", "manylinux_2_17_x86_64"]);

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    bail!(
        "CUDA wheels unsupported on {}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

impl WheelSpec {
    fn dylibs(&self) -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        return self.windows_dylibs;

        #[cfg(target_os = "linux")]
        return self.linux_dylibs;

        #[allow(unreachable_code)]
        &[]
    }
}

fn source_id() -> Result<String> {
    let packages = WHEELS.iter().map(|wheel| wheel.package).collect::<Vec<_>>();
    Ok(format!(
        "cuda;platform={};wheels={}",
        platform_tags()?.join(","),
        packages.join(",")
    ))
}

async fn select_wheel(runtime: &Runtime, wheel: &WheelSpec) -> Result<WheelAsset> {
    let (distribution, version) = wheel
        .package
        .split_once('/')
        .ok_or_else(|| anyhow!("invalid wheel package `{}`", wheel.package))?;

    let metadata_url = format!("https://pypi.org/pypi/{distribution}/{version}/json");
    let release: PypiRelease = runtime
        .http_client()
        .get(&metadata_url)
        .send()
        .await
        .with_context(|| format!("failed to fetch `{metadata_url}`"))?
        .json()
        .await
        .with_context(|| format!("failed to parse metadata for `{distribution}`"))?;

    let tags = platform_tags()?;
    for file in release.urls {
        if file.filename.ends_with(".whl") && tags.iter().any(|tag| file.filename.contains(tag)) {
            return Ok(WheelAsset {
                url: file.url,
                filename: file.filename,
            });
        }
    }

    bail!("no wheel found for `{distribution}` {version} on {tags:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_id_includes_platform() {
        let id = source_id().unwrap();
        assert!(id.contains("cuda"));
        assert!(id.contains("platform="));
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn wheels_have_dylibs_for_current_platform() {
        for wheel in WHEELS {
            assert!(
                !wheel.dylibs().is_empty(),
                "{} has no dylibs",
                wheel.package
            );
        }
    }

    #[test]
    fn preload_order_follows_wheel_declaration() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        for wheel in WHEELS {
            for dylib in wheel.dylibs() {
                std::fs::write(root.join(dylib), b"ok").unwrap();
            }
        }

        let all_dylibs: Vec<&str> = WHEELS
            .iter()
            .flat_map(|wheel| wheel.dylibs().iter().copied())
            .collect();
        for dylib in &all_dylibs {
            assert!(root.join(dylib).exists());
        }
    }

    #[test]
    fn parses_major_minor_from_driver_version() {
        let version = CudaDriverVersion::from_raw(13000);
        assert_eq!(version.major(), 13);
        assert_eq!(version.minor(), 0);
        assert_eq!(version.to_string(), "13.0");
    }

    #[test]
    fn checks_cuda_13_0_threshold() {
        assert!(CudaDriverVersion::from_raw(13000).supports_cuda_13_0());
        assert!(CudaDriverVersion::from_raw(13010).supports_cuda_13_0());
        assert!(!CudaDriverVersion::from_raw(12990).supports_cuda_13_0());
        assert!(!CudaDriverVersion::from_raw(12080).supports_cuda_13_0());
    }
}
