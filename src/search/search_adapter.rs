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

    /// Advanced search using reciprocal rank fusion to intelligently combine BM25 and semantic results
    /// 
    /// This method performs separate BM25 and semantic searches, then uses reciprocal rank fusion
    /// with proper score normalization to create a unified ranking that leverages both approaches.
    /// 
    /// # Arguments
    /// * `query` - The search query string
    /// * `max_results` - Maximum number of results to return
    /// * `k` - RRF parameter (default 60) - higher values reduce the impact of rank differences
    /// * `alpha` - Weight balance between BM25 (1.0) and semantic (0.0) results
    /// 
    /// # Returns
    /// * `Result<Vec<UnifiedMatch>>` - Fused results ranked by combined score
    pub async fn intelligent_fusion(
        &self, 
        query: &str, 
        max_results: usize,
        k: f32,
        alpha: f32
    ) -> Result<Vec<UnifiedMatch>> {
        let mut bm25_results = Vec::new();
        let mut semantic_results = Vec::new();

        // Perform BM25 search
        {
            let bm25_guard = self.bm25_searcher.read().await;
            let raw_bm25_results = bm25_guard.search(query, max_results * 2)?; // Get more for better fusion
            drop(bm25_guard);
            
            bm25_results = raw_bm25_results.into_iter().map(UnifiedMatch::from).collect();
        }

        // Perform semantic search (if available)
        #[cfg(all(feature = "ml", feature = "vectordb"))]
        {
            if let Some(ref embeddings_system) = self.embeddings_system {
                match embeddings_system.embed_and_search(query, max_results * 2).await {
                    Ok(raw_semantic_results) => {
                        semantic_results = raw_semantic_results.into_iter().map(UnifiedMatch::from).collect();
                    }
                    Err(e) => {
                        log::warn!("Semantic search failed during fusion: {}, using BM25 only", e);
                    }
                }
            }
        }

        // Apply reciprocal rank fusion
        self.apply_reciprocal_rank_fusion(bm25_results, semantic_results, max_results, k, alpha)
    }

    /// Apply reciprocal rank fusion to combine two result sets
    /// 
    /// RRF formula: score = α * (1/(k + rank_bm25)) + (1-α) * (1/(k + rank_semantic))
    /// where rank starts at 1 for the top result
    fn apply_reciprocal_rank_fusion(
        &self,
        mut bm25_results: Vec<UnifiedMatch>,
        mut semantic_results: Vec<UnifiedMatch>,
        max_results: usize,
        k: f32,
        alpha: f32
    ) -> Result<Vec<UnifiedMatch>> {
        use std::collections::HashMap;

        // Normalize scores within each result set
        self.normalize_scores(&mut bm25_results);
        self.normalize_scores(&mut semantic_results);

        // Create ranking maps (doc_id -> rank)
        let bm25_ranks: HashMap<String, usize> = bm25_results
            .iter()
            .enumerate()
            .map(|(idx, result)| (result.doc_id.clone(), idx + 1))
            .collect();

        let semantic_ranks: HashMap<String, usize> = semantic_results
            .iter()
            .enumerate()
            .map(|(idx, result)| (result.doc_id.clone(), idx + 1))
            .collect();

        // Collect all unique document IDs
        let mut all_doc_ids = std::collections::HashSet::new();
        all_doc_ids.extend(bm25_ranks.keys().cloned());
        all_doc_ids.extend(semantic_ranks.keys().cloned());

        // Create document lookup maps for getting original results
        let bm25_lookup: HashMap<String, &UnifiedMatch> = bm25_results
            .iter()
            .map(|result| (result.doc_id.clone(), result))
            .collect();

        let semantic_lookup: HashMap<String, &UnifiedMatch> = semantic_results
            .iter()
            .map(|result| (result.doc_id.clone(), result))
            .collect();

        // Calculate RRF scores for each document
        let mut fused_results: Vec<UnifiedMatch> = all_doc_ids
            .into_iter()
            .filter_map(|doc_id| {
                let bm25_rank = bm25_ranks.get(&doc_id);
                let semantic_rank = semantic_ranks.get(&doc_id);

                // Calculate RRF score
                let bm25_score = bm25_rank.map_or(0.0, |rank| 1.0 / (k + *rank as f32));
                let semantic_score = semantic_rank.map_or(0.0, |rank| 1.0 / (k + *rank as f32));
                
                let combined_score = alpha * bm25_score + (1.0 - alpha) * semantic_score;

                // Skip documents with zero combined score
                if combined_score <= 0.0 {
                    return None;
                }

                // Create unified match with the best available information
                let base_match = bm25_lookup.get(&doc_id)
                    .or_else(|| semantic_lookup.get(&doc_id))?;

                // Combine matched terms from both sources
                let mut matched_terms = Vec::new();
                if let Some(bm25_match) = bm25_lookup.get(&doc_id) {
                    matched_terms.extend_from_slice(&bm25_match.matched_terms);
                }

                // Determine match type based on which sources contributed
                let match_type = match (bm25_rank.is_some(), semantic_rank.is_some()) {
                    (true, true) => "Hybrid".to_string(),
                    (true, false) => "Statistical".to_string(),
                    (false, true) => "Semantic".to_string(),
                    (false, false) => unreachable!(), // This case is filtered out above
                };

                Some(UnifiedMatch {
                    doc_id: doc_id.clone(),
                    score: combined_score,
                    match_type,
                    matched_terms,
                    content: base_match.content.clone(),
                })
            })
            .collect();

        // Sort by combined RRF score (descending)
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        fused_results.truncate(max_results);

        Ok(fused_results)
    }

    /// Normalize scores within a result set to [0, 1] range using min-max normalization
    /// 
    /// This ensures fair comparison between BM25 and semantic scores in the fusion process
    fn normalize_scores(&self, results: &mut [UnifiedMatch]) {
        if results.is_empty() {
            return;
        }

        let min_score = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
        let max_score = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);

        // Avoid division by zero
        if (max_score - min_score).abs() < f32::EPSILON {
            // All scores are the same, set them all to 1.0
            for result in results {
                result.score = 1.0;
            }
        } else {
            // Apply min-max normalization: (score - min) / (max - min)
            for result in results {
                result.score = (result.score - min_score) / (max_score - min_score);
            }
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