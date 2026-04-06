use anyhow::{Result, bail};
use koharu_runtime::{ComputePolicy, GpuBackend, RuntimeManager, default_app_data_root};
use tracing_subscriber::fmt::format::FmtSpan;

const ZLUDA_CPU_ENGINES: &[&str] = &["lama-manga", "paddleocr-vl-candle"];

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .try_init();
}

pub async fn prepare_runtime(cpu: bool) -> Result<RuntimeManager> {
    let runtime = RuntimeManager::new(
        default_app_data_root(),
        if cpu {
            ComputePolicy::CpuOnly
        } else {
            ComputePolicy::PreferGpu
        },
    )?;
    runtime.prepare().await?;
    if force_zluda() {
        runtime.require_gpu_backend(GpuBackend::CudaZluda)?;
    }
    Ok(runtime)
}

pub fn effective_cpu(runtime: &RuntimeManager, cpu: bool, engine_id: &str) -> Result<bool> {
    if cpu {
        if force_zluda() {
            bail!("`--cpu` cannot be used when `KOHARU_FORCE_ZLUDA=1`.");
        }
        return Ok(true);
    }

    if matches!(runtime.gpu_backend(), GpuBackend::CudaZluda) && !zluda_supports_engine(engine_id) {
        if force_zluda() {
            bail!(
                "engine `{engine_id}` is not accelerated on ZLUDA; refusing CPU fallback because `KOHARU_FORCE_ZLUDA=1`."
            );
        }
        tracing::info!(
            engine_id,
            "Falling back to CPU because this Candle binary is not accelerated on ZLUDA."
        );
        return Ok(true);
    }

    Ok(false)
}

fn zluda_supports_engine(engine_id: &str) -> bool {
    !ZLUDA_CPU_ENGINES.contains(&engine_id)
}

fn force_zluda() -> bool {
    std::env::var("KOHARU_FORCE_ZLUDA")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}
