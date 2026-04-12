use std::collections::HashMap;

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use koharu_core::{GoogleFontCatalog, GoogleFontEntry};
use tokio::sync::Mutex;
use tracing::debug;

const CATALOG_JSON: &str = include_str!("../data/google-fonts-catalog.json");

const RECOMMENDED_FAMILIES: &[&str] = &[
    "Comic Neue",
    "Bangers",
    "Patrick Hand",
    "Caveat",
    "Pangolin",
];

const ALLOWED_SUBSETS: &[&str] = &[
    "latin",
    "latin-ext",
    "vietnamese",
    "cyrillic",
    "cyrillic-ext",
    "greek",
    "greek-ext",
    "japanese",
    "korean",
];

const CURATED_FAMILIES: &[&str] = &[
    "Roboto",
    "Roboto Condensed",
    "Roboto Mono",
    "Open Sans",
    "Lato",
    "Montserrat",
    "Poppins",
    "Nunito",
    "Nunito Sans",
    "Inter",
    "Work Sans",
    "Source Sans 3",
    "Source Serif 4",
    "Merriweather",
    "Merriweather Sans",
    "IBM Plex Sans",
    "IBM Plex Serif",
    "IBM Plex Mono",
    "Noto Sans",
    "Noto Serif",
    "Noto Sans JP",
    "Noto Serif JP",
    "Noto Sans KR",
    "Noto Serif KR",
    "M PLUS 1",
    "M PLUS 1p",
    "M PLUS Rounded 1c",
    "BIZ UDPGothic",
    "BIZ UDMincho",
    "Zen Kaku Gothic New",
    "Zen Maru Gothic",
    "Zen Old Mincho",
    "Shippori Mincho",
    "Yuji Syuku",
    "DotGothic16",
    "Hina Mincho",
    "Nanum Gothic",
    "Nanum Myeongjo",
    "Stylish",
    "Gamja Flower",
    "Do Hyeon",
    "Gowun Dodum",
    "Gowun Batang",
    "Black Han Sans",
    "Jua",
    "Comic Neue",
    "Bangers",
    "Patrick Hand",
    "Caveat",
    "Pangolin",
    "Kalam",
    "Indie Flower",
    "Architects Daughter",
    "Shadows Into Light Two",
    "Handlee",
    "Baloo 2",
    "Fredoka",
    "Bebas Neue",
    "Oswald",
    "Raleway",
    "Quicksand",
    "Manrope",
    "Barlow",
    "Barlow Condensed",
    "Rubik",
    "Mukta",
    "Fira Sans",
    "Inconsolata",
    "JetBrains Mono",
    "Fira Code",
    "DM Sans",
    "DM Serif Text",
    "Plus Jakarta Sans",
    "Cabin",
    "Playfair Display",
    "Lora",
    "Crimson Text",
    "Libre Baskerville",
    "EB Garamond",
    "Cardo",
    "Cormorant Garamond",
    "Space Grotesk",
    "Outfit",
    "Sora",
    "Heebo",
    "Titillium Web",
];

fn has_only_allowed_subsets(entry: &GoogleFontEntry) -> bool {
    !entry.subsets.is_empty()
        && entry
            .subsets
            .iter()
            .all(|subset| ALLOWED_SUBSETS.contains(&subset.as_str()))
}

fn is_curated_family(family: &str) -> bool {
    CURATED_FAMILIES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(family))
}

/// On-demand Google Fonts service with persistent disk caching.
pub struct GoogleFontService {
    catalog: GoogleFontCatalog,
    cache_dir: Utf8PathBuf,
    /// Tracks which families have been downloaded to disk.
    cached_families: Mutex<HashMap<String, Vec<Utf8PathBuf>>>,
}

impl GoogleFontService {
    pub fn new(app_data_root: &Utf8Path) -> Result<Self> {
        let catalog: GoogleFontCatalog =
            serde_json::from_str(CATALOG_JSON).context("failed to parse Google Fonts catalog")?;
        let cache_dir = app_data_root.join("fonts").join("google");
        std::fs::create_dir_all(cache_dir.as_std_path())
            .context("failed to create Google Fonts cache dir")?;

        // Scan existing cache to populate known cached families
        let mut cached_families = HashMap::new();
        for entry in &catalog.fonts {
            let family_dir = cache_dir.join(normalize_family_dir(&entry.family));
            if family_dir.exists() {
                let paths: Vec<Utf8PathBuf> = entry
                    .variants
                    .iter()
                    .map(|v| family_dir.join(&v.filename))
                    .filter(|p| p.exists())
                    .collect();
                if !paths.is_empty() {
                    cached_families.insert(entry.family.clone(), paths);
                }
            }
        }

        Ok(Self {
            catalog,
            cache_dir,
            cached_families: Mutex::new(cached_families),
        })
    }

    /// Returns the full catalog for browsing.
    pub fn catalog(&self) -> &GoogleFontCatalog {
        &self.catalog
    }

    /// Returns the list of recommended font family names.
    pub fn recommended_families(&self) -> &[&str] {
        RECOMMENDED_FAMILIES
    }

    /// Restricts the browser list to a curated set of practical fonts and
    /// supported subsets (Latin/European + Korean/Japanese).
    pub fn is_entry_browsable(&self, entry: &GoogleFontEntry) -> bool {
        has_only_allowed_subsets(entry) && is_curated_family(&entry.family)
    }

    /// Checks if a family has been cached to disk.
    pub async fn is_cached(&self, family: &str) -> bool {
        self.cached_families.lock().await.contains_key(family)
    }

    /// Downloads a font family's regular variant to disk cache.
    /// Returns the path to the cached .ttf file.
    /// No-op if already cached.
    pub async fn fetch_family(
        &self,
        family: &str,
        http: &reqwest_middleware::ClientWithMiddleware,
    ) -> Result<Utf8PathBuf> {
        // Check cache first
        {
            let cached = self.cached_families.lock().await;
            if let Some(first) = cached.get(family).and_then(|p| p.first()) {
                return Ok(first.clone());
            }
        }

        let entry = self
            .catalog
            .fonts
            .iter()
            .find(|e| e.family == family)
            .with_context(|| format!("font family not found in catalog: {family}"))?;

        // Prefer regular weight, fallback to first variant
        let variant = entry
            .variants
            .iter()
            .find(|v| v.weight == 400 && v.style == "normal")
            .or_else(|| entry.variants.first())
            .context("font has no variants")?;

        let family_dir_name = normalize_family_dir(&entry.family);
        let url = format!(
            "https://raw.githubusercontent.com/google/fonts/main/ofl/{}/{}",
            family_dir_name, variant.filename
        );

        debug!(%family, %url, "downloading Google Font");
        let response = http
            .get(&url)
            .send()
            .await
            .context("failed to fetch Google Font")?
            .error_for_status()
            .context("Google Fonts CDN returned an error")?;
        let bytes = response
            .bytes()
            .await
            .context("failed to read font bytes")?;

        // Write to disk cache
        let family_dir = self.cache_dir.join(&family_dir_name);
        std::fs::create_dir_all(family_dir.as_std_path())?;
        let file_path = family_dir.join(&variant.filename);
        std::fs::write(file_path.as_std_path(), &bytes)
            .with_context(|| format!("failed to write cached font to {file_path}"))?;

        // Update in-memory cache tracking
        self.cached_families
            .lock()
            .await
            .insert(family.to_string(), vec![file_path.clone()]);

        Ok(file_path)
    }

    /// Reads the cached font file bytes. Returns None if not cached.
    pub fn read_cached_file(&self, family: &str) -> Result<Option<Vec<u8>>> {
        let entry = self.catalog.fonts.iter().find(|e| e.family == family);
        let Some(entry) = entry else {
            return Ok(None);
        };
        let variant = entry
            .variants
            .iter()
            .find(|v| v.weight == 400 && v.style == "normal")
            .or_else(|| entry.variants.first());
        let Some(variant) = variant else {
            return Ok(None);
        };
        let file_path = self
            .cache_dir
            .join(normalize_family_dir(&entry.family))
            .join(&variant.filename);
        if !file_path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(file_path.as_std_path()).context("failed to read cached font")?;
        Ok(Some(data))
    }

    /// Find catalog entry by family name.
    pub fn find_entry(&self, family: &str) -> Option<&GoogleFontEntry> {
        self.catalog.fonts.iter().find(|e| e.family == family)
    }
}

/// Converts family name to directory name (lowercase, spaces to empty).
/// e.g. "Comic Neue" -> "comicneue"
fn normalize_family_dir(family: &str) -> String {
    family.to_lowercase().replace(' ', "")
}
