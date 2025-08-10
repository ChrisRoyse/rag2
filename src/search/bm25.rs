use rustc_hash::FxHashMap;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

/// High-performance BM25 implementation optimized for code search
#[derive(Debug, Clone)]
pub struct BM25Engine {
    /// Term frequency saturation parameter (default: 1.2)
    k1: f32,
    /// Length normalization parameter (default: 0.75)
    b: f32,
    
    /// Document collection statistics
    total_docs: usize,
    avg_doc_length: f32,
    
    /// Term statistics: term -> (document_frequency, total_frequency)
    term_frequencies: FxHashMap<String, TermStats>,
    /// Document lengths: doc_id -> length
    pub document_lengths: FxHashMap<String, usize>,
    
    /// Inverted index for fast lookups: term -> documents containing it
    inverted_index: FxHashMap<String, Vec<DocumentTerm>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStats {
    /// How many documents contain this term
    pub document_frequency: usize,
    /// Total occurrences across all documents
    pub total_frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentTerm {
    /// Document identifier
    pub doc_id: String,
    /// Occurrences in this document
    pub term_frequency: usize,
    /// Word positions for phrase queries
    pub positions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct BM25Match {
    pub doc_id: String,
    pub score: f32,
    /// Individual term contributions for debugging
    pub term_scores: FxHashMap<String, f32>,
    pub matched_terms: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BM25Document {
    pub id: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub tokens: Vec<Token>,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub position: usize,
    pub importance_weight: f32,
}

// BM25Engine must be explicitly configured with parameters - no default fallback allowed
// Use BM25Engine::with_params(k1, b) or BM25Engine::new() explicitly

impl BM25Engine {
    pub fn new() -> Self {
        Self::with_params(1.2, 0.75)
    }
    
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            total_docs: 0,
            avg_doc_length: 0.0,
            term_frequencies: FxHashMap::default(),
            document_lengths: FxHashMap::default(),
            inverted_index: FxHashMap::default(),
        }
    }
    
    /// Add a document to the BM25 index from a file path
    pub fn add_document_from_file(&mut self, file_path: &str) -> Result<()> {
        // Read file content
        let content = std::fs::read_to_string(file_path)
            .context(format!("Failed to read file: {}", file_path))?;
        
        // Tokenize content into Tokens
        let tokens: Vec<Token> = content
            .split_whitespace()
            .enumerate()
            .map(|(pos, word)| Token {
                text: word.to_lowercase(),
                position: pos,
                importance_weight: 1.0,
            })
            .collect();
        
        // Create BM25Document from file
        let doc = BM25Document {
            id: file_path.to_string(),
            file_path: file_path.to_string(),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: content.lines().count(),
            language: None,
        };
        
        self.add_document(doc)
    }
    
    /// Add a document to the BM25 index
    pub fn add_document(&mut self, doc: BM25Document) -> Result<()> {
        let doc_id = doc.id.clone();
        let doc_length = doc.tokens.len();
        
        // Update document count and average length
        let total_length = (self.avg_doc_length * self.total_docs as f32) + doc_length as f32;
        self.total_docs += 1;
        self.avg_doc_length = if self.total_docs > 0 {
            total_length / self.total_docs as f32
        } else {
            0.0
        };
        
        // Store document length
        self.document_lengths.insert(doc_id.clone(), doc_length);
        
        // Process tokens and update inverted index
        let mut term_positions: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut term_counts: FxHashMap<String, usize> = FxHashMap::default();
        
        for (pos, token) in doc.tokens.iter().enumerate() {
            // Always normalize terms to lowercase for consistency
            let term = token.text.to_lowercase();
            
            // Track positions for this term
            term_positions.entry(term.clone())
                .or_insert_with(Vec::new)
                .push(pos);
            
            // Count term frequency
            *term_counts.entry(term.clone()).or_insert(0) += 1;
        }
        
        // Update inverted index and term statistics
        for (term, positions) in term_positions {
            let freq = term_counts[&term];
            
            // Update term statistics
            let stats = self.term_frequencies.entry(term.clone())
                .or_insert(TermStats {
                    document_frequency: 0,
                    total_frequency: 0,
                });
            stats.document_frequency += 1;
            stats.total_frequency += freq;
            
            // Add to inverted index
            let doc_term = DocumentTerm {
                doc_id: doc_id.clone(),
                term_frequency: freq,
                positions,
            };
            
            self.inverted_index.entry(term)
                .or_insert_with(Vec::new)
                .push(doc_term);
        }
        
        Ok(())
    }
    
    /// Calculate IDF (Inverse Document Frequency) for a term
    pub fn calculate_idf(&self, term: &str) -> f32 {
        let term_lower = term.to_lowercase();
        
        if let Some(stats) = self.term_frequencies.get(&term_lower) {
            let n = self.total_docs as f32;
            let df = stats.document_frequency as f32;
            // BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
            let raw_idf = ((n - df + 0.5) / (df + 0.5)).ln();
            
            // FIXED: Proper IDF handling - preserve relative ordering
            // Raw IDF can be negative for very common terms (df > N/2)
            // We need to ensure rare terms (lower df) get higher IDF than common terms (higher df)
            let idf = if raw_idf < 0.0 {
                // Map negative IDF to small positive values, preserving order
                // More negative (more common) should give smaller positive values
                0.001 * (1.0 / (raw_idf.abs() + 1.0))
            } else {
                raw_idf.max(0.001)  // Ensure minimum positive value
            };
            idf
        } else {
            // Term not in any document, return high IDF
            (self.total_docs as f32 + 1.0).ln()
        }
    }
    
    /// Calculate BM25 score for a document given query terms
    pub fn calculate_bm25_score(&self, query_terms: &[String], doc_id: &str) -> Result<f32, anyhow::Error> {
        let doc_length = *self.document_lengths.get(doc_id)
            .ok_or_else(|| anyhow::anyhow!("Document '{}' not found in BM25 index. Document must be indexed before scoring.", doc_id))? as f32;
        
        let mut score = 0.0;
        
        for term in query_terms {
            let term_lower = term.to_lowercase();
            let idf = self.calculate_idf(&term_lower);
            
            // Find term frequency in this document
            // If term doesn't exist in index OR document doesn't contain term, tf = 0.0 (legitimate BM25 behavior)
            let tf = match self.inverted_index.get(&term_lower) {
                Some(docs) => {
                    // Term exists in index, check if this specific document contains it
                    match docs.iter().find(|dt| dt.doc_id == doc_id) {
                        Some(doc_term) => doc_term.term_frequency as f32,
                        None => 0.0, // Document doesn't contain this term - legitimate 0.0 contribution for BM25
                    }
                }
                None => {
                    // Term not found in any document - legitimate 0.0 contribution for BM25
                    0.0
                }
            };
            
            if tf > 0.0 {
                // BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                let norm_factor = 1.0 - self.b + self.b * (doc_length / self.avg_doc_length);
                let term_score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm_factor);
                score += term_score;
            }
        }
        
        Ok(score)
    }
    
    /// Search for documents matching the query
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<BM25Match>, anyhow::Error> {
        // Tokenize query (simple whitespace split for now)
        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        
        if query_terms.is_empty() {
            return Err(anyhow::anyhow!("Empty query provided to BM25 search. Query must contain valid search terms."));
        }
        
        // Find all documents that contain at least one query term
        let mut candidate_docs: FxHashMap<String, Vec<String>> = FxHashMap::default();
        
        for term in &query_terms {
            if let Some(doc_terms) = self.inverted_index.get(term) {
                for doc_term in doc_terms {
                    candidate_docs.entry(doc_term.doc_id.clone())
                        .or_insert_with(Vec::new)
                        .push(term.clone());
                }
            }
        }
        
        // Calculate BM25 scores for candidate documents
        let mut matches: Vec<BM25Match> = Vec::new();
        
        for (doc_id, matched_terms) in &candidate_docs {
            let score = self.calculate_bm25_score(&query_terms, doc_id)
                .with_context(|| format!("BM25 calculation failed for document '{}' - mathematical integrity compromised", doc_id))?;
                
            // Calculate individual term contributions for debugging
            let mut term_scores = FxHashMap::default();
            for term in &query_terms {
                let single_term_score = self.calculate_bm25_score(&[term.clone()], &doc_id)
                    .with_context(|| format!("Single term BM25 calculation failed for term '{}' in document '{}' - mathematical integrity compromised", term, doc_id))?;
                
                if single_term_score != 0.0 && single_term_score.is_finite() {
                    term_scores.insert(term.clone(), single_term_score);
                }
                // Note: Zero or negative scores are mathematically valid results, not errors
            }
            
            if score != 0.0 && score.is_finite() {
                matches.push(BM25Match {
                    doc_id: doc_id.to_string(),
                    score,
                    term_scores,
                    matched_terms: matched_terms.clone(),
                });
            }
        }
        
        // Validate all scores are finite before sorting - PRINCIPLE 0: No NaN fallbacks
        for (idx, match_result) in matches.iter().enumerate() {
            if !match_result.score.is_finite() {
                return Err(anyhow::anyhow!(
                    "BM25 score calculation produced invalid result (NaN or infinite) for document '{}' (index {}). Score: {}. This indicates mathematical corruption in BM25 computation and cannot be recovered from.",
                    match_result.doc_id, idx, match_result.score
                ));
            }
        }
        
        // Sort by score descending - safe after validation
        matches.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap() // Safe after validation
        });
        
        // Return top results
        matches.truncate(limit);
        Ok(matches)
    }
    
    /// Get statistics about the index with memory usage estimation
    pub fn get_stats(&self) -> IndexStats {
        let estimated_memory = self.estimate_memory_usage();
        
        IndexStats {
            total_documents: self.total_docs,
            total_terms: self.term_frequencies.len(),
            avg_document_length: self.avg_doc_length,
            k1: self.k1,
            b: self.b,
            estimated_memory_bytes: estimated_memory,
            performance_metrics: None, // Set by indexing operations
        }
    }
    
    /// Estimate memory usage of the index
    fn estimate_memory_usage(&self) -> usize {
        let mut total_bytes = 0;
        
        // Document lengths map
        total_bytes += self.document_lengths.len() * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
        for (key, _) in &self.document_lengths {
            total_bytes += key.len();
        }
        
        // Term frequencies map
        total_bytes += self.term_frequencies.len() * (std::mem::size_of::<String>() + std::mem::size_of::<TermStats>());
        for (key, _) in &self.term_frequencies {
            total_bytes += key.len();
        }
        
        // Inverted index
        for (term, doc_terms) in &self.inverted_index {
            total_bytes += term.len();
            total_bytes += doc_terms.len() * std::mem::size_of::<DocumentTerm>();
            for doc_term in doc_terms {
                total_bytes += doc_term.doc_id.len();
                total_bytes += doc_term.positions.len() * std::mem::size_of::<usize>();
            }
        }
        
        total_bytes
    }
    
    /// Index all files in a directory with performance monitoring and async processing
    pub async fn index_directory(&mut self, dir_path: &std::path::Path) -> Result<IndexStats> {
        self.index_directory_with_progress::<fn(usize, usize)>(dir_path, None).await
    }
    
    /// Index directory with optional progress callback
    pub async fn index_directory_with_progress<F>(
        &mut self, 
        dir_path: &std::path::Path,
        progress_callback: Option<F>
    ) -> Result<IndexStats> 
    where 
        F: Fn(usize, usize) + Send + Sync
    {
        use std::fs;
        use std::path::Path;
        use tokio::task;
        use std::sync::{Arc, Mutex};
        
        if !dir_path.exists() {
            return Err(anyhow::anyhow!("Directory does not exist: {}", dir_path.display()));
        }
        
        if !dir_path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory: {}", dir_path.display()));
        }
        
        let start_time = std::time::Instant::now();
        
        // First pass: collect all files to process
        let mut all_files = Vec::new();
        self.collect_files_recursive(dir_path, &mut all_files)?;
        
        let total_files = all_files.len();
        let processed_count = Arc::new(Mutex::new(0));
        
        println!("📁 Found {} files to index", total_files);
        
        // Process files in batches for better memory management
        let batch_size = 50;
        let mut batch_start = 0;
        
        while batch_start < all_files.len() {
            let batch_end = (batch_start + batch_size).min(all_files.len());
            let batch_files = &all_files[batch_start..batch_end];
            
            // Process batch
            for file_path in batch_files {
                match self.process_single_file(file_path).await {
                    Ok(_) => {
                        let mut count = processed_count.lock().unwrap();
                        *count += 1;
                        
                        if let Some(ref callback) = progress_callback {
                            callback(*count, total_files);
                        }
                        
                        // Progress reporting every 10 files
                        if *count % 10 == 0 {
                            let elapsed = start_time.elapsed();
                            let rate = *count as f64 / elapsed.as_secs_f64();
                            println!("⚡ Processed {}/{} files ({:.1} files/sec)", 
                                   *count, total_files, rate);
                        }
                    }
                    Err(e) => {
                        eprintln!("⚠️  Failed to process {}: {}", file_path.display(), e);
                        // Continue with other files
                    }
                }
            }
            
            batch_start = batch_end;
            
            // Yield control to allow other tasks to run
            task::yield_now().await;
        }
        
        // Update statistics after batch operation
        self.update_statistics()?;
        
        let final_elapsed = start_time.elapsed();
        let processed = *processed_count.lock().unwrap();
        let rate = processed as f64 / final_elapsed.as_secs_f64();
        
        println!("🎉 Indexing complete: {} files in {:.2}s ({:.1} files/sec)", 
                processed, final_elapsed.as_secs_f64(), rate);
        
        Ok(self.get_stats())
    }
    
    /// Collect all indexable files recursively
    fn collect_files_recursive(&self, dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
        use std::fs;
        
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                // Skip common non-source directories
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if matches!(dir_name, "target" | "node_modules" | ".git" | "__pycache__" | "build" | "dist") {
                        continue;
                    }
                }
                self.collect_files_recursive(&path, files)?;
            } else if path.is_file() {
                if self.is_indexable_file(&path) {
                    files.push(path);
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if file should be indexed based on extension and size
    fn is_indexable_file(&self, path: &std::path::Path) -> bool {
        // Check file extension
        if let Some(extension) = path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            
            let supported_extensions = [
                // Rust
                "rs",
                // Python
                "py", "pyx", "pyi",
                // JavaScript/TypeScript
                "js", "jsx", "ts", "tsx", "vue",
                // Java
                "java", "scala", "kt", "gradle",
                // C/C++
                "c", "cpp", "cxx", "cc", "h", "hpp", "hxx",
                // C#
                "cs",
                // Go
                "go",
                // Ruby
                "rb",
                // PHP
                "php",
                // Swift
                "swift",
                // Objective-C
                "m", "mm",
                // Shell
                "sh", "bash", "zsh", "fish",
                // Config files
                "json", "yaml", "yml", "toml", "xml", "ini", "cfg",
                // Documentation
                "md", "txt", "rst",
                // SQL
                "sql",
                // Dockerfile
                "dockerfile",
            ];
            
            if !supported_extensions.contains(&ext_str.as_str()) {
                return false;
            }
        } else {
            return false;
        }
        
        // Check file size (skip very large files)
        if let Ok(metadata) = path.metadata() {
            let size_mb = metadata.len() as f64 / 1_048_576.0;
            if size_mb > 5.0 { // Skip files larger than 5MB
                return false;
            }
        }
        
        true
    }
    
    /// Process a single file and add it to the index
    pub async fn process_single_file(&mut self, path: &std::path::Path) -> Result<()> {
        use tokio::fs;
        
        let content = fs::read_to_string(path).await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        
        // Skip empty files
        if content.trim().is_empty() {
            return Ok(());
        }
        
        let doc_id = format!("{}_{}", 
                           path.to_string_lossy().replace(['/', '\\'], "_"),
                           content.len()); // Use content length as uniqueness factor
        
        let tokens = tokenize_content(&content);
        
        // Skip files with too few meaningful tokens
        if tokens.len() < 5 {
            return Ok(());
        }
        
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let doc = BM25Document {
            id: doc_id,
            file_path: path.to_string_lossy().to_string(),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: content.lines().count(),
            language: if extension.is_empty() { None } else { Some(extension) },
        };
        
        self.add_document(doc)
    }
    
    /// Update an existing document in the BM25 index
    /// This removes the old document and adds the new one in a thread-safe manner
    pub fn update_document(&mut self, doc: BM25Document) -> Result<()> {
        // Remove existing document if it exists
        if let Err(_) = self.remove_document(&doc.id) {
            // Document didn't exist, proceed with addition
        }
        
        // Add the updated document
        self.add_document(doc)
    }
    
    /// Remove a document from the BM25 index
    pub fn remove_document(&mut self, doc_id: &str) -> Result<()> {
        // Get document length before removal
        let doc_length = self.document_lengths.get(doc_id)
            .ok_or_else(|| anyhow::anyhow!("Document '{}' not found in BM25 index", doc_id))?;
        let doc_length = *doc_length;
        
        // Update document count and average length
        if self.total_docs > 0 {
            let total_length = (self.avg_doc_length * self.total_docs as f32) - doc_length as f32;
            self.total_docs -= 1;
            self.avg_doc_length = if self.total_docs > 0 {
                total_length / self.total_docs as f32
            } else {
                0.0
            };
        }
        
        // Remove document length entry
        self.document_lengths.remove(doc_id);
        
        // Remove document from inverted index and update term frequencies
        let mut terms_to_remove = Vec::new();
        for (term, doc_terms) in self.inverted_index.iter_mut() {
            // Find entries for this document and calculate total frequency to subtract
            let total_freq_to_subtract: usize = doc_terms.iter()
                .filter(|dt| dt.doc_id == doc_id)
                .map(|dt| dt.term_frequency)
                .sum();
            
            // Find and remove entries for this document
            let original_len = doc_terms.len();
            doc_terms.retain(|dt| dt.doc_id != doc_id);
            
            // If we removed entries, update term statistics
            if doc_terms.len() != original_len {
                let removed_count = original_len - doc_terms.len();
                
                if let Some(stats) = self.term_frequencies.get_mut(term) {
                    stats.document_frequency = stats.document_frequency.saturating_sub(removed_count);
                    stats.total_frequency = stats.total_frequency.saturating_sub(total_freq_to_subtract);
                    
                    // Mark term for removal if no documents contain it
                    if stats.document_frequency == 0 {
                        terms_to_remove.push(term.clone());
                    }
                }
                
                // Mark empty document term vectors for removal
                if doc_terms.is_empty() {
                    terms_to_remove.push(term.clone());
                }
            }
        }
        
        // Remove terms that no longer have any documents
        for term in terms_to_remove {
            self.inverted_index.remove(&term);
            self.term_frequencies.remove(&term);
        }
        
        Ok(())
    }
    
    /// Remove all documents associated with a specific file path
    pub fn remove_documents_by_path(&mut self, file_path: &std::path::Path) -> Result<usize> {
        let path_prefix = file_path.to_string_lossy().replace(['/', '\\'], "_");
        let mut removed_count = 0;
        
        // Find all document IDs that start with this path prefix
        let docs_to_remove: Vec<String> = self.document_lengths.keys()
            .filter(|doc_id| doc_id.starts_with(&path_prefix))
            .cloned()
            .collect();
        
        // Remove each document
        for doc_id in docs_to_remove {
            if let Err(e) = self.remove_document(&doc_id) {
                eprintln!("Warning: Failed to remove document {}: {}", doc_id, e);
            } else {
                removed_count += 1;
            }
        }
        
        if removed_count > 0 {
            // Update statistics after removals
            self.update_statistics()?;
        }
        
        Ok(removed_count)
    }
    
    /// Update index statistics after batch operations
    /// This recalculates average document length and validates consistency
    pub fn update_statistics(&mut self) -> Result<()> {
        // Recalculate total documents
        let actual_doc_count = self.document_lengths.len();
        if actual_doc_count != self.total_docs {
            println!("⚠️ Document count mismatch detected: expected {}, found {}. Correcting...", 
                     self.total_docs, actual_doc_count);
            self.total_docs = actual_doc_count;
        }
        
        // Recalculate average document length
        if self.total_docs > 0 {
            let total_length: usize = self.document_lengths.values().sum();
            self.avg_doc_length = total_length as f32 / self.total_docs as f32;
        } else {
            self.avg_doc_length = 0.0;
        }
        
        // Validate term frequencies consistency
        let mut inconsistencies = 0;
        for (term, stats) in self.term_frequencies.iter_mut() {
            if let Some(doc_terms) = self.inverted_index.get(term) {
                let actual_doc_freq = doc_terms.len();
                let actual_total_freq: usize = doc_terms.iter().map(|dt| dt.term_frequency).sum();
                
                if actual_doc_freq != stats.document_frequency {
                    stats.document_frequency = actual_doc_freq;
                    inconsistencies += 1;
                }
                
                if actual_total_freq != stats.total_frequency {
                    stats.total_frequency = actual_total_freq;
                    inconsistencies += 1;
                }
            } else {
                // Term exists in statistics but not in index - remove it
                stats.document_frequency = 0;
                stats.total_frequency = 0;
                inconsistencies += 1;
            }
        }
        
        if inconsistencies > 0 {
            println!("⚠️ Fixed {} term frequency inconsistencies during statistics update", inconsistencies);
        }
        
        Ok(())
    }
    
    /// Clear the entire index
    pub fn clear(&mut self) {
        self.total_docs = 0;
        self.avg_doc_length = 0.0;
        self.term_frequencies.clear();
        self.document_lengths.clear();
        self.inverted_index.clear();
    }
}

/// Enhanced tokenization function for code search
/// Extracts identifiers, function names, and meaningful terms while filtering noise
pub fn tokenize_content(content: &str) -> Vec<Token> {
    use regex::Regex;
    use std::collections::HashSet;
    
    let mut tokens = Vec::new();
    let mut position = 0;
    
    // Common stop words to filter out
    let stop_words: HashSet<&str> = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those"
    ].iter().copied().collect();
    
    // Regex patterns for different code elements
    let identifier_pattern = Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*").expect("Valid regex");
    let string_literal_pattern = Regex::new(r#""[^"]*"|'[^']*'"#).expect("Valid regex");
    let comment_pattern = Regex::new(r"//.*?$|/\*.*?\*/").expect("Valid regex");
    
    // Remove comments first
    let content_no_comments = comment_pattern.replace_all(content, " ");
    
    // Extract string literals (for context)
    for string_match in string_literal_pattern.find_iter(&content_no_comments) {
        let string_content = string_match.as_str();
        // Extract words from string literals
        for word in string_content.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| c.is_ascii_punctuation())
                             .to_lowercase();
            if cleaned.len() > 2 && !stop_words.contains(cleaned.as_str()) {
                tokens.push(Token {
                    text: cleaned,
                    position,
                    importance_weight: 0.5, // Lower weight for string content
                });
                position += 1;
            }
        }
    }
    
    // Extract identifiers (function names, variable names, types)
    for identifier_match in identifier_pattern.find_iter(&content_no_comments) {
        let identifier = identifier_match.as_str().to_lowercase();
        
        // Skip very short identifiers and common keywords
        if identifier.len() < 2 {
            continue;
        }
        
        // Skip common programming keywords
        if matches!(identifier.as_str(), 
            "if" | "else" | "for" | "while" | "do" | "try" | "catch" | "finally" |
            "int" | "str" | "bool" | "let" | "var" | "const" | "fn" | "def" | "class" |
            "new" | "return" | "break" | "continue" | "true" | "false" | "null" | "none") {
            continue;
        }
        
        if !stop_words.contains(identifier.as_str()) {
            // Higher weight for longer identifiers (likely more meaningful)
            let importance = if identifier.len() > 6 {
                1.5
            } else if identifier.contains('_') || identifier.chars().any(|c| c.is_uppercase()) {
                1.2 // Function names, constants
            } else {
                1.0
            };
            
            tokens.push(Token {
                text: identifier,
                position,
                importance_weight: importance,
            });
            position += 1;
        }
    }
    
    // Also process remaining non-identifier words
    let words_pattern = Regex::new(r"\b\w+\b").expect("Valid regex");
    for word_match in words_pattern.find_iter(&content_no_comments) {
        let word = word_match.as_str().to_lowercase();
        
        // Skip if already processed as identifier or too short
        if word.len() < 3 || identifier_pattern.is_match(&word) {
            continue;
        }
        
        if !stop_words.contains(word.as_str()) {
            tokens.push(Token {
                text: word,
                position,
                importance_weight: 0.8,
            });
            position += 1;
        }
    }
    
    // Remove duplicates while preserving order and combining weights
    let mut unique_tokens = Vec::new();
    let mut seen = HashSet::new();
    
    for token in tokens {
        if !seen.contains(&token.text) {
            seen.insert(token.text.clone());
            unique_tokens.push(token);
        }
    }
    
    unique_tokens
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_documents: usize,
    pub total_terms: usize,
    pub avg_document_length: f32,
    pub k1: f32,
    pub b: f32,
    /// Memory usage estimation in bytes
    pub estimated_memory_bytes: usize,
    /// Indexing performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub indexing_time_seconds: f64,
    pub files_per_second: f64,
    pub terms_per_second: f64,
    pub peak_memory_mb: f64,
}

// Incremental update tests would go here when implemented

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bm25_basic() {
        let mut engine = BM25Engine::new();
        
        // Add test documents
        let doc1 = BM25Document {
            id: "doc1".to_string(),
            file_path: "test1.rs".to_string(),
            chunk_index: 0,
            tokens: vec![
                Token { text: "quick".to_string(), position: 0, importance_weight: 1.0 },
                Token { text: "brown".to_string(), position: 1, importance_weight: 1.0 },
                Token { text: "fox".to_string(), position: 2, importance_weight: 1.0 },
            ],
            start_line: 0,
            end_line: 10,
            language: Some("rust".to_string()),
        };
        
        let doc2 = BM25Document {
            id: "doc2".to_string(),
            file_path: "test2.rs".to_string(),
            chunk_index: 0,
            tokens: vec![
                Token { text: "quick".to_string(), position: 0, importance_weight: 1.0 },
                Token { text: "quick".to_string(), position: 1, importance_weight: 1.0 },
                Token { text: "dog".to_string(), position: 2, importance_weight: 1.0 },
            ],
            start_line: 0,
            end_line: 10,
            language: Some("rust".to_string()),
        };
        
        engine.add_document(doc1).unwrap();
        engine.add_document(doc2).unwrap();
        
        // Search for "quick"
        let results = engine.search("quick", 10).expect("Search must succeed in test");
        assert_eq!(results.len(), 2);
        
        // doc2 should score higher due to higher term frequency
        assert!(results[0].doc_id == "doc2");
        assert!(results[0].score > results[1].score);
        
        // Search for "fox"
        let results = engine.search("fox", 10).expect("Search must succeed in test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc1");
    }
    
    #[test]
    fn test_idf_calculation() {
        let mut engine = BM25Engine::new();
        
        // Add documents with varying term frequencies
        for i in 0..10 {
            let mut tokens = vec![
                Token { text: "common".to_string(), position: 0, importance_weight: 1.0 },
            ];
            
            if i < 2 {
                tokens.push(Token { text: "rare".to_string(), position: 1, importance_weight: 1.0 });
            }
            
            let doc = BM25Document {
                id: format!("doc{}", i),
                file_path: format!("test{}.rs", i),
                chunk_index: 0,
                tokens,
                start_line: 0,
                end_line: 10,
                language: Some("rust".to_string()),
            };
            
            engine.add_document(doc).unwrap();
        }
        
        // IDF of common term should be lower than rare term
        let idf_common = engine.calculate_idf("common");
        let idf_rare = engine.calculate_idf("rare");
        
        assert!(idf_rare > idf_common);
    }
}