use koharu_core::{ZLUDA_GPU_ENGINES, zluda_supports_engine};
use koharu_runtime::GpuBackend;

const CANDLE_GPU_ENGINES: &[&str] = &[
    "pp-doclayout-v3",
    "comic-text-detector",
    "comic-text-detector-seg",
    "comic-text-bubble-detector",
    "speech-bubble-segmentation",
    "yuzumarker-font-detection",
    "manga-ocr",
    "mit48px-ocr",
    "lama-manga",
    "aot-inpainting",
];

#[derive(Debug, Clone)]
pub struct ModelAccelerationPolicy {
    backend: GpuBackend,
}

impl ModelAccelerationPolicy {
    pub fn new(backend: GpuBackend) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    pub fn force_cpu(&self, engine_id: &str) -> bool {
        !self.accelerates(engine_id)
    }

    pub fn accelerates(&self, engine_id: &str) -> bool {
        match self.backend {
            GpuBackend::Cpu => false,
            GpuBackend::Metal | GpuBackend::CudaNvidia => CANDLE_GPU_ENGINES.contains(&engine_id),
            GpuBackend::CudaZluda => zluda_supports_engine(engine_id),
        }
    }

    pub fn accelerated_engines(&self) -> Vec<String> {
        match self.backend {
            GpuBackend::Cpu => Vec::new(),
            GpuBackend::Metal | GpuBackend::CudaNvidia => CANDLE_GPU_ENGINES
                .iter()
                .map(|engine| (*engine).to_string())
                .collect(),
            GpuBackend::CudaZluda => ZLUDA_GPU_ENGINES
                .iter()
                .map(|engine| (*engine).to_string())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zluda_accelerates_only_allowlisted_engines() {
        let policy = ModelAccelerationPolicy::new(GpuBackend::CudaZluda);
        assert!(policy.accelerates("comic-text-bubble-detector"));
        assert!(policy.accelerates("comic-text-detector"));
        assert!(policy.accelerates("comic-text-detector-seg"));
        assert!(policy.accelerates("aot-inpainting"));
        assert!(policy.force_cpu("pp-doclayout-v3"));
        assert!(policy.force_cpu("speech-bubble-segmentation"));
        assert!(policy.force_cpu("yuzumarker-font-detection"));
        assert!(policy.force_cpu("manga-ocr"));
        assert!(policy.force_cpu("mit48px-ocr"));
        assert!(policy.force_cpu("lama-manga"));
        assert!(policy.force_cpu("paddle-ocr-vl-1.5"));
    }

    #[test]
    fn nvidia_policy_keeps_existing_candle_gpu_allowlist() {
        let policy = ModelAccelerationPolicy::new(GpuBackend::CudaNvidia);
        assert!(policy.accelerates("pp-doclayout-v3"));
        assert!(policy.accelerates("lama-manga"));
        assert!(policy.accelerates("comic-text-detector-seg"));
        assert!(policy.force_cpu("paddle-ocr-vl-1.5"));
    }
}
