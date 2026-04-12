use std::collections::HashMap;

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use koharu_core::{GoogleFontCatalog, GoogleFontEntry};
use tokio::sync::Mutex;
use tracing::debug;

const CATALOG_JSON: &str = include_str!("../data/google-fonts-catalog.json");

/// Fonts shown at the top of the selector as quick picks.
const RECOMMENDED_FAMILIES: &[&str] = &[
    "Bangers",
    "Comic Neue",
    "Bebas Neue",
    "Patrick Hand",
    "Caveat",
];

/// Subsets considered "useful" — a font passes if it has AT LEAST ONE of these.
/// Using any() rather than all() so that popular fonts like Roboto and Open Sans
/// (which Google has since expanded with math/symbols/devanagari subsets) are
/// not silently excluded.
const USEFUL_SUBSETS: &[&str] = &[
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

/// Curated for manga/manhwa scanlation. Organised by role so it's easy to
/// review at a glance. Coding fonts, academic serifs, and niche decorative
/// faces have been removed.
const CURATED_FAMILIES: &[&str] = &[
    // ── Dialogue & Handwritten ──────────────────────────────────────────────
    // These are the fonts scanlation groups reach for first.
    "Comic Neue",           // Primary dialogue font (Wild Words alternative)
    "Patrick Hand",         // Clean, legible dialogue
    "Caveat",               // Handwritten action / inner-monologue text
    "Pangolin",             // Slightly quirky dialogue alternative
    "Kalam",                // Handwritten, natural feel
    "Indie Flower",         // Loose handwritten
    "Architects Daughter",  // Popular in webtoons for casual dialogue
    "Shadows Into Light Two", // Expressive handwritten
    "Handlee",              // Clean informal script

    // ── Impact, SFX & Chapter Titles ────────────────────────────────────────
    "Bangers",              // #1 SFX font — used by nearly every group
    "Bebas Neue",           // Condensed all-caps, chapter headers & titles
    "Anton",                // Heavy bold, common for impact text in manhwa
    "Russo One",            // Rounded bold titles
    "Black Ops One",        // Action / military manhwa titles
    "Oswald",               // Condensed, versatile title font
    "Barlow Condensed",     // Compact titles that need to fit tight spaces
    "Lilita One",           // Bold rounded display, popular in manhwa covers
    "Righteous",            // Clean display, good for fantasy/action titles
    "Cinzel",               // Roman-carved look, perfect for isekai/fantasy

    // ── General Purpose (clean, modern) ────────────────────────────────────
    // Used for translated narration boxes, UI-style text, and when a neutral
    // readable face is needed.
    "Roboto",
    "Roboto Condensed",
    "Open Sans",
    "Lato",
    "Montserrat",
    "Poppins",
    "Nunito",
    "Nunito Sans",
    "Inter",
    "Work Sans",
    "DM Sans",
    "Plus Jakarta Sans",
    "Barlow",
    "Cabin",
    "Quicksand",
    "Raleway",
    "Manrope",
    "Rubik",               // Clean rounded sans, popular in Webtoon English TLs
    "Fredoka",             // Playful rounded, lighter slice-of-life manhwa
    "Source Sans 3",
    "Exo 2",               // Sci-fi / tech manhwa

    // ── Japanese ────────────────────────────────────────────────────────────
    "M PLUS 1",
    "M PLUS 1p",
    "M PLUS Rounded 1c",
    "BIZ UDPGothic",
    "BIZ UDMincho",
    "Zen Kaku Gothic New",
    "Zen Maru Gothic",
    "Zen Old Mincho",
    "Shippori Mincho",
    "Hina Mincho",
    "DotGothic16",
    "Yuji Syuku",

    // ── Korean ──────────────────────────────────────────────────────────────
    "Nanum Gothic",
    "Nanum Myeongjo",
    "Nanum Pen Script",
    "Nanum Brush Script",
    "Black Han Sans",
    "Do Hyeon",
    "Gowun Dodum",
    "Gowun Batang",
    "Jua",
    "Sunflower",
    "Dokdo",
];

/// Returns true if the font supports at least one of our useful script subsets.
/// Uses any() so that fonts like Roboto or Open Sans — which Google has since
/// expanded with math/symbols/devanagari subsets — are not silently excluded.
fn has_any_useful_subset(entry: &GoogleFontEntry) -> bool {
    entry
        .subsets
        .iter()
        .any(|subset| USEFUL_SUBSETS.contains(&subset.as_str()))
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
        has_any_useful_subset(entry) && is_curated_family(&entry.family)
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
