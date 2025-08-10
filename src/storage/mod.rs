// Storage module - CLEANED UP: Essential implementations only
// BRUTAL CLEANUP: Removed 5+ duplicate storage implementations

// KEPT: Only essential storage implementations
pub mod safe_vectordb;       // Thread-safe main vector storage
pub mod simple_vectordb;     // Simple fallback storage

// Re-export commonly used types (CLEANED UP)
pub use safe_vectordb::{VectorStorage, StorageConfig};
pub use simple_vectordb::{StorageError, EmbeddingRecord, VectorSchema};

// Common storage traits and types
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub path: String,
    pub timestamp: u64,
    pub chunk_index: usize,
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub use_reranking: bool,
}

pub trait VectorStore {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: VectorMetadata) -> crate::Result<()>;
    fn search(&self, query_vector: &[f32], config: &SearchConfig) -> crate::Result<Vec<(String, f32, VectorMetadata)>>;
    fn remove(&mut self, id: &str) -> crate::Result<bool>;
    fn size(&self) -> usize;
    fn clear(&mut self) -> crate::Result<()>;
}