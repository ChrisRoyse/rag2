// Storage module - CLEANED UP: Essential implementations only
// BRUTAL CLEANUP: Removed 5+ duplicate storage implementations

// KEPT: Only essential storage implementations
pub mod safe_vectordb;       // Thread-safe main vector storage
pub mod simple_vectordb;     // Simple fallback storage

// NEW: Real LanceDB implementation for production vector storage
#[cfg(feature = "vectordb")]
pub mod lancedb_storage;     // Production LanceDB vector storage

// Re-export commonly used types (CLEANED UP)
pub use safe_vectordb::{VectorStorage, StorageConfig};

// Export from simple_vectordb with feature flag check
#[cfg(feature = "vectordb")]
pub use simple_vectordb::{StorageError, EmbeddingRecord, VectorSchema};

// Provide fallback types when vectordb feature is not enabled
#[cfg(not(feature = "vectordb"))]
pub use self::fallback_types::{StorageError, EmbeddingRecord, VectorSchema};

#[cfg(not(feature = "vectordb"))]
mod fallback_types {
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone)]
    pub struct StorageError {
        pub message: String,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingRecord {
        pub id: String,
        pub content: String,
        pub embedding: Vec<f32>,
        pub metadata: Option<serde_json::Value>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VectorSchema {
        pub dimensions: usize,
        pub distance_metric: String,
    }
}

#[cfg(feature = "vectordb")]
pub use lancedb_storage::{LanceDBStorage, LanceDBConfig, LanceDBRecord, LanceDBError, LanceDBStats};

// NEW: Integrated Nomic + LanceDB system for production use
#[cfg(all(feature = "vectordb", feature = "ml"))]
pub mod nomic_lancedb_integration;

#[cfg(all(feature = "vectordb", feature = "ml"))]
pub use nomic_lancedb_integration::{
    NomicLanceDBSystem, SystemConfig, SystemStats, create_nomic_lancedb_system
};

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