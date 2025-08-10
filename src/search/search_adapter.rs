//! Unified search adapter for BM25 + semantic embeddings
//! 
//! This module provides a unified interface that combines BM25 statistical search
//! with semantic embeddings from Nomic + LanceDB when features are enabled.

use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use async_trait::async_trait;

use crate::search::{BM25Searcher, BM25Match};

#[cfg(all(feature = "ml", feature = "vectordb"))]
use crate::storage::nomic_lancedb_integration::NomicLanceDBSystem;
#[cfg(all(feature = "ml", feature = "vectordb"))]
use crate::chunking::Chunk;

/// Unified search adapter that combines BM25 + semantic embeddings
pub struct UnifiedSearchAdapter {
    bm25_searcher: Arc<RwLock<BM25Searcher>>,
    #[cfg(all(feature = "ml", feature = "vectordb"))]
    embeddings_system: Option<Arc<NomicLanceDBSystem>>,
}

/// Combined search result with multiple match types
#[derive(Debug, Clone)]
pub struct UnifiedMatch {
    pub doc_id: String,
    pub score: f32,
    pub match_type: String,
    pub matched_terms: Vec<String>,
    pub content: Option<String>,
}

impl From<BM25Match> for UnifiedMatch {
    fn from(bm25_match: BM25Match) -> Self {
        Self {
            doc_id: bm25_match.doc_id,
            score: bm25_match.score,
            match_type: "Statistical".to_string(),
            matched_terms: bm25_match.matched_terms,
            content: None,
        }
    }
}

#[cfg(all(feature = "ml", feature = "vectordb"))]
impl From<crate::storage::lancedb_storage::LanceDBRecord> for UnifiedMatch {
    fn from(record: crate::storage::lancedb_storage::LanceDBRecord) -> Self {
        Self {
            doc_id: record.file_path,
            score: 0.8, // Default semantic score - could be computed from embedding similarity
            match_type: "Semantic".to_string(),
            matched_terms: vec![], // Semantic matches don't have explicit terms
            content: Some(record.content),
        }
    }
}

impl UnifiedSearchAdapter {
    /// Create new unified search adapter with BM25 base
    pub fn new(bm25_searcher: Arc<RwLock<BM25Searcher>>) -> Self {
        Self {
            bm25_searcher,
            #[cfg(all(feature = "ml", feature = "vectordb"))]
            embeddings_system: None,
        }
    }

    /// Initialize with both BM25 and embeddings (when features available)
    #[cfg(all(feature = "ml", feature = "vectordb"))]
    pub async fn with_embeddings(
        bm25_searcher: Arc<RwLock<BM25Searcher>>, 
        db_path: std::path::PathBuf
    ) -> Result<Self> {
        let embeddings_system = Arc::new(
            crate::storage::nomic_lancedb_integration::create_nomic_lancedb_system(db_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize embeddings: {}", e))?
        );

        Ok(Self {
            bm25_searcher,
            embeddings_system: Some(embeddings_system),
        })
    }

    /// Unified search that combines BM25 + semantic results
    pub async fn search(
        &self, 
        query: &str, 
        max_results: usize
    ) -> Result<Vec<UnifiedMatch>> {
        let mut all_results = Vec::new();

        // Always perform BM25 search (core functionality)
        let bm25_guard = self.bm25_searcher.read().await;
        let bm25_results = bm25_guard.search(query, max_results / 2)?;
        drop(bm25_guard);
        
        all_results.extend(
            bm25_results.into_iter().map(UnifiedMatch::from)
        );
        
        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            if let Some(ref embeddings_system) = self.embeddings_system {
                match embeddings_system.embed_and_search(query, max_results / 2).await {
                    Ok(semantic_results) => {
                        all_results.extend(
                            semantic_results.into_iter().map(UnifiedMatch::from)
                        );
                    }
                    Err(e) => {
                        log::warn!("Semantic search failed: {}, using BM25 only", e);
                    }
                }
            }
        }

        // Sort by score and limit results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(max_results);
        
        Ok(all_results)
    }

    /// Index a file with both BM25 and embeddings
    pub async fn index_file(&self, file_path: &Path) -> Result<()> {
        // BM25 indexing (always available)
        {
            let mut bm25_guard = self.bm25_searcher.write().await;
            // Read file content and create BM25Document
            let content = std::fs::read_to_string(file_path)?;
            let tokens = crate::search::bm25::tokenize_content(&content);
            let doc = crate::search::BM25Document {
                id: file_path.to_string_lossy().to_string(),
                file_path: file_path.to_string_lossy().to_string(),
                chunk_index: 0,
                tokens,
                start_line: 1,
                end_line: content.lines().count(),
                language: Some("text".to_string()),
            };
            bm25_guard.add_document(doc)?;
        }

        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            if let Some(ref embeddings_system) = self.embeddings_system {
                // Read file and chunk it for embeddings
                match std::fs::read_to_string(file_path) {
                    Ok(content) => {
                        // Simple chunk creation for now
                        let chunks = vec![crate::chunking::Chunk {
                            content: content.clone(),
                            start_line: 1,
                            end_line: content.lines().count(),
                        }];
                        let batch_data: Vec<(&str, usize, Chunk)> = chunks
                            .into_iter()
                            .enumerate()
                            .map(|(idx, chunk)| {
                                (file_path.to_string_lossy().as_ref(), idx, chunk)
                            })
                            .collect();
                        
                        if let Err(e) = embeddings_system.embed_and_store_batch(batch_data).await {
                            log::warn!("Failed to store embeddings for {}: {}", file_path.display(), e);
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to read file {} for embedding: {}", file_path.display(), e);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Clear all indexes
    pub async fn clear_all(&self) -> Result<()> {
        // Clear BM25 index
        {
            let mut bm25_guard = self.bm25_searcher.write().await;
            bm25_guard.clear();
        }

        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            if let Some(ref embeddings_system) = self.embeddings_system {
                embeddings_system.clear_all_embeddings().await
                    .map_err(|e| anyhow::anyhow!("Failed to clear embeddings: {}", e))?;
            }
        }
        
        Ok(())
    }

    /// Get statistics about both search backends
    pub async fn get_stats(&self) -> Result<SearchStats> {
        let bm25_guard = self.bm25_searcher.read().await;
        let bm25_doc_count = bm25_guard.document_lengths.len();
        drop(bm25_guard);

        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            if let Some(ref embeddings_system) = self.embeddings_system {
                let embedding_stats = embeddings_system.get_system_stats().await
                    .map_err(|e| anyhow::anyhow!("Failed to get embedding stats: {}", e))?;
                
                return Ok(SearchStats {
                    bm25_documents: bm25_doc_count,
                    embedding_count: Some(embedding_stats.total_embeddings),
                    backends_available: vec!["BM25".to_string(), "Semantic".to_string()],
                });
            }
        }

        Ok(SearchStats {
            bm25_documents: bm25_doc_count,
            embedding_count: None,
            backends_available: vec!["BM25".to_string()],
        })
    }

    /// Check if embeddings are available
    pub fn has_embeddings(&self) -> bool {
        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            self.embeddings_system.is_some()
        }
        #[cfg(not(all(feature = "ml", feature = "vectordb")))]
        {
            false
        }
    }
}

/// Search statistics across all backends
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub bm25_documents: usize,
    pub embedding_count: Option<usize>,
    pub backends_available: Vec<String>,
}

/// Legacy trait for compatibility (now implemented by UnifiedSearchAdapter)
#[async_trait]
pub trait TextSearcher {
    /// Search for text and return matches
    async fn search(&self, query: &str) -> Result<Vec<crate::search::ExactMatch>>;
    
    /// Index a single file
    async fn index_file(&mut self, file_path: &Path) -> Result<()>;
    
    /// Clear the entire index
    async fn clear_index(&mut self) -> Result<()>;
    
    /// Update a document in the index
    async fn update_document(&mut self, file_path: &Path) -> Result<()>;
    
    /// Remove a document from the index
    async fn remove_document(&mut self, file_path: &Path) -> Result<()>;
    
    /// Reload the search index reader
    async fn reload_reader(&self) -> Result<()>;
}