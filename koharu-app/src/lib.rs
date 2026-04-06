pub mod acceleration;
pub mod config;
pub mod edit;
pub mod engine;
pub mod google_fonts;
pub mod io;
pub mod llm;
pub mod pipeline;
pub mod renderer;
pub mod storage;
pub mod utils;

use std::sync::Arc;

use koharu_runtime::{GpuBackend, RuntimeManager};
use tokio::sync::RwLock;

use crate::acceleration::ModelAccelerationPolicy;
use crate::config::AppConfig;
use crate::engine::Registry;
use crate::storage::Storage;

#[derive(Clone)]
pub struct AppResources {
    pub runtime: RuntimeManager,
    pub storage: Arc<Storage>,
    pub registry: Arc<Registry>,
    pub config: Arc<RwLock<AppConfig>>,
    pub llm: Arc<llm::Model>,
    pub gpu_backend: GpuBackend,
    pub model_acceleration: ModelAccelerationPolicy,
    pub pipeline: Arc<RwLock<Option<pipeline::PipelineHandle>>>,
    pub version: &'static str,
}

impl AppResources {
    pub fn candle_uses_cpu(&self, engine_id: &str) -> bool {
        self.model_acceleration.force_cpu(engine_id)
    }

    pub fn llm_uses_cpu(&self) -> bool {
        !self.gpu_backend.uses_gpu()
    }
}
