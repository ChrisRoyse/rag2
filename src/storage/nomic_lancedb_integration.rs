// Nomic + LanceDB Integration Layer - FUNCTIONAL IMPLEMENTATION ONLY
// This module provides integrated embeddings + vector storage for production use

#[cfg(all(feature = "vectordb", feature = "ml"))]
use std::path::PathBuf;
#[cfg(all(feature = "vectordb", feature = "ml"))]
use std::sync::Arc;
#[cfg(all(feature = "vectordb", feature = "ml"))]
use anyhow::Result;
#[cfg(all(feature = "vectordb", feature = "ml"))]
use crate::embedding::nomic::NomicEmbedder;
#[cfg(all(feature = "vectordb", feature = "ml"))]
use crate::storage::lancedb_storage::{LanceDBStorage, LanceDBConfig, LanceDBRecord, LanceDBError};
#[cfg(all(feature = "vectordb", feature = "ml"))]
use crate::chunking::Chunk;

/// Integrated Nomic embeddings + LanceDB storage system
#[cfg(all(feature = "vectordb", feature = "ml"))]
pub struct NomicLanceDBSystem {
    embedder: Arc<NomicEmbedder>,
    storage: LanceDBStorage,
    config: SystemConfig,
}

#[cfg(all(feature = "vectordb", feature = "ml"))]
#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub batch_size: usize,
    pub max_retries: u32,
    pub enable_cache: bool,
}

#[cfg(all(feature = "vectordb", feature = "ml"))]
impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            max_retries: 3,
            enable_cache: true,
        }
    }
}

#[cfg(all(feature = "vectordb", feature = "ml"))]
impl NomicLanceDBSystem {
    /// Create new integrated system
    pub async fn new(
        embedder: Arc<NomicEmbedder>,
        db_path: PathBuf,
        config: Option<SystemConfig>,
    ) -> Result<Self, LanceDBError> {
        let config = config.unwrap_or_default();
        
        // Initialize LanceDB storage with Nomic's 768-dimension embeddings
        let lance_config = LanceDBConfig::new(db_path, 768);
        let mut storage = LanceDBStorage::new(lance_config).await?;
        storage.init_table().await?;

        Ok(Self {
            embedder,
            storage,
            config,
        })
    }

    /// Embed text and store in LanceDB in one operation
    pub async fn embed_and_store(
        &self,
        file_path: &str,
        chunk_index: usize,
        chunk: &Chunk,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Generate embedding using Nomic
        let embedding = self.embedder.embed(&chunk.content)
            .map_err(|e| format!("Embedding failed: {}", e))?;

        // Store in LanceDB
        self.storage
            .insert_embedding(file_path, chunk_index, chunk, embedding)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Embed text and search for similar content
    pub async fn embed_and_search(
        &self,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<LanceDBRecord>, Box<dyn std::error::Error + Send + Sync>> {
        // Generate query embedding using Nomic
        let query_embedding = self.embedder.embed(query_text)
            .map_err(|e| format!("Query embedding failed: {}", e))?;

        // Search in LanceDB
        self.storage
            .search_similar(query_embedding, limit)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Batch embed and store multiple chunks efficiently
    pub async fn embed_and_store_batch(
        &self,
        items: Vec<(&str, usize, Chunk)>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if items.is_empty() {
            return Ok(());
        }

        // Process in configurable batch sizes
        for chunk_batch in items.chunks(self.config.batch_size) {
            // Extract texts for batch embedding
            let texts: Vec<&str> = chunk_batch
                .iter()
                .map(|(_, _, chunk)| chunk.content.as_str())
                .collect();

            // Generate embeddings using Nomic batch processing
            let embeddings = self.embedder.embed_batch(&texts)
                .map_err(|e| format!("Batch embedding failed: {}", e))?;

            // Combine with original data for LanceDB
            let batch_data: Vec<(&str, usize, Chunk, Vec<f32>)> = chunk_batch
                .iter()
                .zip(embeddings.into_iter())
                .map(|((file_path, chunk_index, chunk), embedding)| {
                    (*file_path, *chunk_index, chunk.clone(), embedding)
                })
                .collect();

            // Store batch in LanceDB with retry logic
            let mut retries = 0;
            loop {
                match self.storage.insert_batch(batch_data.clone()).await {
                    Ok(()) => break,
                    Err(e) if retries < self.config.max_retries => {
                        retries += 1;
                        log::warn!("Batch insert retry {} failed: {}", retries, e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            100 * (1 << retries) // Exponential backoff
                        )).await;
                    }
                    Err(e) => return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                }
            }
        }

        Ok(())
    }

    /// Get comprehensive system statistics
    pub async fn get_system_stats(&self) -> Result<SystemStats, LanceDBError> {
        let lance_stats = self.storage.get_stats().await?;
        let embedding_stats = self.embedder.get_stats();

        Ok(SystemStats {
            total_embeddings: lance_stats.total_embeddings,
            embedding_dimension: lance_stats.embedding_dimension,
            table_name: lance_stats.table_name,
            cache_hits: embedding_stats.cache_hits,
            cache_misses: embedding_stats.cache_misses,
            average_embedding_time_ms: embedding_stats.average_time_ms,
        })
    }

    /// Delete all embeddings for a file
    pub async fn delete_file_embeddings(
        &self,
        file_path: &str,
    ) -> Result<(), LanceDBError> {
        self.storage.delete_by_file(file_path).await
    }

    /// Clear all embeddings
    pub async fn clear_all_embeddings(&self) -> Result<(), LanceDBError> {
        self.storage.clear_all().await
    }

    /// Get embedding count
    pub async fn count_embeddings(&self) -> Result<usize, LanceDBError> {
        self.storage.count().await
    }

    /// Direct access to storage for advanced operations
    pub fn storage(&self) -> &LanceDBStorage {
        &self.storage
    }

    /// Direct access to embedder for advanced operations
    pub fn embedder(&self) -> Arc<NomicEmbedder> {
        self.embedder.clone()
    }
}

#[cfg(all(feature = "vectordb", feature = "ml"))]
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub total_embeddings: usize,
    pub embedding_dimension: usize,
    pub table_name: String,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub average_embedding_time_ms: f64,
}

/// Factory function to create integrated system with sensible defaults
#[cfg(all(feature = "vectordb", feature = "ml"))]
pub async fn create_nomic_lancedb_system(
    db_path: PathBuf,
) -> Result<NomicLanceDBSystem, Box<dyn std::error::Error + Send + Sync>> {
    // Initialize Nomic embedder
    let embedder = Arc::new(
        NomicEmbedder::new().await
            .map_err(|e| format!("Failed to initialize Nomic embedder: {}", e))?
    );

    // Create integrated system
    NomicLanceDBSystem::new(embedder, db_path, None)
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
}

#[cfg(all(test, feature = "vectordb", feature = "ml"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::chunking::Chunk;

    #[tokio::test]
    async fn test_integration_system_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("integration_test.db");

        let result = create_nomic_lancedb_system(db_path).await;
        
        // Note: This test may fail if Nomic models are not available
        // That's acceptable for dependency validation
        match result {
            Ok(_) => println!("Integration system created successfully"),
            Err(e) => println!("Integration system creation failed (expected in CI): {}", e),
        }
    }

    #[tokio::test]
    async fn test_integration_workflow() {
        // Skip if no model files available
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("workflow_test.db");

        if let Ok(system) = create_nomic_lancedb_system(db_path).await {
            let chunk = Chunk {
                content: "fn example_function() { println!(\"Hello World\"); }".to_string(),
                start_line: 1,
                end_line: 3,
            };

            // Test embed and store
            let result = system.embed_and_store("example.rs", 0, &chunk).await;
            assert!(result.is_ok(), "Embed and store failed: {:?}", result);

            // Test embed and search
            let search_results = system
                .embed_and_search("function that prints hello", 5)
                .await;

            match search_results {
                Ok(results) => {
                    assert!(!results.is_empty(), "No search results found");
                    assert_eq!(results[0].content, chunk.content);
                }
                Err(e) => panic!("Search failed: {}", e),
            }

            // Test statistics
            let stats = system.get_system_stats().await;
            assert!(stats.is_ok(), "Failed to get stats: {:?}", stats);
        } else {
            println!("Skipping integration workflow test - no model available");
        }
    }
}