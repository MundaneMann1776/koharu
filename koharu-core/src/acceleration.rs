pub const ZLUDA_GPU_ENGINES: &[&str] = &[
    "comic-text-bubble-detector",
    "comic-text-detector",
    "comic-text-detector-seg",
    "aot-inpainting",
];

#[must_use]
pub fn zluda_supports_engine(engine_id: &str) -> bool {
    ZLUDA_GPU_ENGINES.contains(&engine_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zluda_allowlist_matches_current_supported_engines() {
        assert!(zluda_supports_engine("comic-text-bubble-detector"));
        assert!(zluda_supports_engine("comic-text-detector"));
        assert!(zluda_supports_engine("comic-text-detector-seg"));
        assert!(zluda_supports_engine("aot-inpainting"));
        assert!(!zluda_supports_engine("pp-doclayout-v3"));
        assert!(!zluda_supports_engine("speech-bubble-segmentation"));
        assert!(!zluda_supports_engine("yuzumarker-font-detection"));
        assert!(!zluda_supports_engine("manga-ocr"));
        assert!(!zluda_supports_engine("mit48px-ocr"));
        assert!(!zluda_supports_engine("lama-manga"));
        assert!(!zluda_supports_engine("paddleocr-vl-candle"));
    }
}
