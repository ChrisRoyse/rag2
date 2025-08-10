use serde::{Deserialize, Serialize};

/// A search match result containing file information and match details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactMatch {
    pub file_path: String,
    pub line_number: usize,
    pub content: String,
    pub line_content: String,
}

// CLEANED UP: Essential modules only - duplicates removed
pub mod preprocessing;     // ✅ Text preprocessing utilities
pub mod bm25;              // ✅ Main BM25 implementation (ONLY ONE)
pub mod text_processor;    // ✅ Text processing utilities
pub mod inverted_index;    // ✅ Inverted index data structure
pub mod cache;             // ✅ Search result caching
pub mod config;            // ✅ Search configuration
pub mod symbol_index;      // ✅ Symbol indexing for code search
#[cfg(feature = "tantivy")]
pub mod tantivy_search;    // ✅ Tantivy full-text search (ONLY ONE)

// Re-export essential types only - duplicates removed
pub use preprocessing::QueryPreprocessor;
pub use bm25::{BM25Engine, BM25Engine as BM25Searcher, BM25Match, BM25Document, Token as BM25Token};
pub use text_processor::{CodeTextProcessor, ProcessedToken, TokenType};
pub use inverted_index::{InvertedIndex, DocumentMetadata};
pub use config::SearchConfig;
// REMOVED: pub use symbol_index::{SymbolIndex, SymbolInfo}; // tree-sitter dependencies missing
pub use cache::SearchResult;
#[cfg(feature = "tantivy")]
pub use tantivy_search::TantivySearcher;