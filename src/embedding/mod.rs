// Embedding module - CLEANED UP: Essential implementations only
// BRUTAL CLEANUP: Removed 3+ duplicate embedding implementations

// KEPT: Only essential embedding implementations  
pub mod nomic;      // Main Nomic embedding implementation (ONLY ONE)
pub mod cache;      // Embedding cache system

// Re-export commonly used types (CLEANED UP)
#[cfg(feature = "ml")]
pub use nomic::{NomicEmbedder, EmbeddingConfig};

#[cfg(not(feature = "ml"))]
pub struct EmbeddingConfig;

#[cfg(not(feature = "ml"))]
pub struct NomicEmbedder;
pub use cache::{EmbeddingCache, CacheEntry, CacheStats};

// Common embedding traits and types
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    pub total_embeddings: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub average_time_ms: f64,
}

pub trait Embedder {
    fn embed(&self, text: &str) -> crate::Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
    fn get_stats(&self) -> EmbeddingStats;
}