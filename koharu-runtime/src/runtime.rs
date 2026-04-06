use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::{Context, Result, bail};
use camino::Utf8PathBuf;
use reqwest_middleware::ClientWithMiddleware;
use strum::{EnumMessage, IntoStaticStr};
use tokio::sync::broadcast;

use crate::downloads::Downloads;
use crate::packages::PackageCatalog;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePolicy {
    PreferGpu,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoStaticStr, EnumMessage)]
pub enum GpuBackend {
    #[strum(serialize = "cpu", message = "CPU")]
    Cpu,
    #[strum(serialize = "metal", message = "Metal")]
    Metal,
    #[strum(serialize = "cuda-nvidia", message = "CUDA (NVIDIA)")]
    CudaNvidia,
    #[strum(serialize = "cuda-zluda", message = "CUDA (ZLUDA)")]
    CudaZluda,
}

impl GpuBackend {
    pub fn as_str(self) -> &'static str {
        self.into()
    }

    pub fn display_name(self) -> &'static str {
        self.get_message().expect("gpu backend display name")
    }

    pub const fn uses_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeHttpConfig {
    pub connect_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for RuntimeHttpConfig {
    fn default() -> Self {
        Self {
            connect_timeout_secs: 20,
            read_timeout_secs: 300,
            max_retries: 3,
        }
    }
}

pub fn default_app_data_root() -> Utf8PathBuf {
    let root = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("Koharu");
    Utf8PathBuf::from_path_buf(root)
        .unwrap_or_else(|path| Utf8PathBuf::from(path.to_string_lossy().into_owned()))
}

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    root: PathBuf,
    compute: ComputePolicy,
    downloads: Downloads,
    packages: PackageCatalog,
    gpu_backend: RwLock<Option<GpuBackend>>,
}

impl Runtime {
    pub fn new(root: impl Into<PathBuf>, compute: ComputePolicy) -> Result<Self> {
        Self::new_with_http(root, compute, RuntimeHttpConfig::default())
    }

    pub fn new_with_http(
        root: impl Into<PathBuf>,
        compute: ComputePolicy,
        http: RuntimeHttpConfig,
    ) -> Result<Self> {
        let root = root.into();
        let downloads = Downloads::new(
            root.join("runtime").join(".downloads"),
            root.join("models").join("huggingface"),
            &http,
        )?;

        Ok(Self {
            inner: Arc::new(RuntimeInner {
                root,
                compute,
                downloads,
                packages: PackageCatalog::discover(),
                gpu_backend: RwLock::new(None),
            }),
        })
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    pub fn wants_gpu(&self) -> bool {
        matches!(self.inner.compute, ComputePolicy::PreferGpu)
    }

    pub fn http_client(&self) -> Arc<ClientWithMiddleware> {
        self.inner.downloads.client()
    }

    pub fn subscribe_downloads(&self) -> broadcast::Receiver<koharu_core::DownloadProgress> {
        self.inner.downloads.subscribe()
    }

    pub fn downloads(&self) -> Downloads {
        self.inner.downloads.clone()
    }

    pub fn gpu_backend(&self) -> GpuBackend {
        self.cached_gpu_backend()
            .unwrap_or_else(|| self.resolve_gpu_backend(false))
    }

    pub async fn prepare(&self) -> Result<()> {
        let dirs = [
            self.root().join("runtime"),
            self.root().join("runtime").join(".downloads"),
            self.root().join("models"),
            self.root().join("models").join("huggingface"),
        ];
        for dir in dirs {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("failed to create `{}`", dir.display()))?;
        }

        let planned_backend = self.resolve_gpu_backend(true);
        *self
            .inner
            .gpu_backend
            .write()
            .expect("gpu backend lock poisoned") = Some(planned_backend);

        self.inner.packages.prepare_bootstrap(self).await?;

        let backend = self.finalize_gpu_backend(planned_backend, true);
        *self
            .inner
            .gpu_backend
            .write()
            .expect("gpu backend lock poisoned") = Some(backend);

        Ok(())
    }

    pub fn require_gpu_backend(&self, required: GpuBackend) -> Result<()> {
        if let Some(resolved) = self.cached_gpu_backend() {
            if resolved != required {
                return mismatch_backend(required, resolved);
            }
            return verify_selected_backend(self, required).map_err(|err| {
                err.context(format!("required ML backend {}", required.display_name()))
            });
        }

        BackendProbe::detect(self).require(required)
    }

    pub fn llama_directory(&self) -> Result<PathBuf> {
        crate::llama::runtime_dir(self)
    }

    fn resolve_gpu_backend(&self, diagnostics: bool) -> GpuBackend {
        let probe = BackendProbe::detect(self);
        if diagnostics {
            probe.log_resolution();
        }
        probe.resolved()
    }

    fn finalize_gpu_backend(&self, planned: GpuBackend, diagnostics: bool) -> GpuBackend {
        if let Err(err) = verify_selected_backend(self, planned) {
            if diagnostics && planned.uses_gpu() {
                tracing::warn!(
                    "Resolved ML backend {} could not be activated: {err:#}; falling back to CPU.",
                    planned.display_name()
                );
            }
            GpuBackend::Cpu
        } else {
            planned
        }
    }

    fn cached_gpu_backend(&self) -> Option<GpuBackend> {
        *self
            .inner
            .gpu_backend
            .read()
            .expect("gpu backend lock poisoned")
    }
}

pub type RuntimeManager = Runtime;

#[derive(Debug, Clone, Copy)]
struct BackendAvailability {
    wants_gpu: bool,
    metal: bool,
    cuda_nvidia: bool,
    cuda_zluda: bool,
}

#[derive(Debug)]
struct BackendProbe {
    wants_gpu: bool,
    nvidia: Result<(), anyhow::Error>,
    zluda: Result<(), anyhow::Error>,
    resolved: GpuBackend,
}

impl BackendProbe {
    fn detect(runtime: &Runtime) -> Self {
        let wants_gpu = runtime.wants_gpu();
        let nvidia = if wants_gpu {
            crate::cuda::backend_info().map(|_| ())
        } else {
            Ok(())
        };
        let zluda = if wants_gpu {
            crate::zluda::candidate_status(runtime)
        } else {
            Ok(())
        };
        let resolved = resolve_backend(BackendAvailability {
            wants_gpu,
            metal: cfg!(all(target_os = "macos", target_arch = "aarch64")),
            cuda_nvidia: nvidia.is_ok(),
            cuda_zluda: zluda.is_ok(),
        });

        Self {
            wants_gpu,
            nvidia,
            zluda,
            resolved,
        }
    }

    fn resolved(&self) -> GpuBackend {
        self.resolved
    }

    fn require(self, required: GpuBackend) -> Result<()> {
        let Self {
            wants_gpu,
            nvidia,
            zluda,
            resolved,
        } = self;

        if resolved == required {
            return Ok(());
        }

        match required {
            GpuBackend::Cpu => mismatch_backend(required, resolved),
            GpuBackend::Metal => {
                require_gpu_enabled(required, wants_gpu)?;
                if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    bail!(
                        "required ML backend {}, but this platform does not support Metal",
                        required.display_name()
                    );
                }
                mismatch_backend(required, resolved)
            }
            GpuBackend::CudaNvidia => {
                require_gpu_enabled(required, wants_gpu)?;
                nvidia.map_err(|err| {
                    err.context(format!("required ML backend {}", required.display_name()))
                })?;
                mismatch_backend(required, resolved)
            }
            GpuBackend::CudaZluda => {
                require_gpu_enabled(required, wants_gpu)?;
                zluda.map_err(|err| {
                    err.context(format!("required ML backend {}", required.display_name()))
                })?;
                mismatch_backend(required, resolved)
            }
        }
    }

    fn log_resolution(&self) {
        if self.resolved == GpuBackend::Cpu && self.wants_gpu {
            if let Err(err) = &self.nvidia {
                tracing::info!("NVIDIA CUDA backend unavailable: {err:#}");
            }
            if cfg!(target_os = "windows")
                && let Err(err) = &self.zluda
            {
                tracing::warn!(
                    "ZLUDA backend unavailable: {err:#}; falling back to CPU for unsupported Candle models."
                );
            }
        }

        tracing::info!("Resolved ML backend: {}", self.resolved.display_name());
    }
}

fn verify_selected_backend(runtime: &Runtime, selected: GpuBackend) -> Result<()> {
    match selected {
        GpuBackend::Cpu => Ok(()),
        GpuBackend::Metal => {
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                Ok(())
            } else {
                bail!("this platform does not support Metal")
            }
        }
        GpuBackend::CudaNvidia => crate::cuda::backend_info().map(|_| ()),
        GpuBackend::CudaZluda => crate::zluda::backend_status(runtime),
    }
}

fn require_gpu_enabled(required: GpuBackend, wants_gpu: bool) -> Result<()> {
    if !wants_gpu {
        bail!(
            "required ML backend {}, but compute policy is CPU-only",
            required.display_name()
        );
    }
    Ok(())
}

fn mismatch_backend(required: GpuBackend, resolved: GpuBackend) -> Result<()> {
    bail!(
        "required ML backend {}, but resolved {}",
        required.display_name(),
        resolved.display_name()
    )
}

fn resolve_backend(availability: BackendAvailability) -> GpuBackend {
    if !availability.wants_gpu {
        GpuBackend::Cpu
    } else if availability.cuda_nvidia {
        GpuBackend::CudaNvidia
    } else if availability.cuda_zluda {
        GpuBackend::CudaZluda
    } else if availability.metal {
        GpuBackend::Metal
    } else {
        GpuBackend::Cpu
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;

    use super::*;

    #[test]
    fn resolver_prefers_nvidia_when_available() {
        let backend = resolve_backend(BackendAvailability {
            wants_gpu: true,
            metal: true,
            cuda_nvidia: true,
            cuda_zluda: true,
        });
        assert_eq!(backend, GpuBackend::CudaNvidia);
    }

    #[test]
    fn resolver_selects_zluda_when_nvidia_is_unavailable() {
        let backend = resolve_backend(BackendAvailability {
            wants_gpu: true,
            metal: false,
            cuda_nvidia: false,
            cuda_zluda: true,
        });
        assert_eq!(backend, GpuBackend::CudaZluda);
    }

    #[test]
    fn resolver_falls_back_to_cpu_when_gpu_backends_are_unavailable() {
        let backend = resolve_backend(BackendAvailability {
            wants_gpu: true,
            metal: false,
            cuda_nvidia: false,
            cuda_zluda: false,
        });
        assert_eq!(backend, GpuBackend::Cpu);
    }

    #[tokio::test]
    #[ignore]
    async fn prepares_llama_runtime_into_configured_root() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        let runtime = Runtime::new(tempdir.path(), ComputePolicy::CpuOnly)?;
        runtime.prepare().await?;
        assert!(runtime.llama_directory()?.exists());
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn repeated_basename_loads_succeed_after_prepare() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        let runtime = Runtime::new(tempdir.path(), ComputePolicy::CpuOnly)?;
        runtime.prepare().await?;
        let dir = runtime.llama_directory()?;

        let lib_name = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.contains("llama").then_some(name)
            })
            .next()
            .ok_or_else(|| anyhow::anyhow!("no llama library found"))?;

        let _first = crate::load_library_by_name(&lib_name)?;
        let _second = crate::load_library_by_name(&lib_name)?;
        Ok(())
    }
}
