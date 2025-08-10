//! Index directory tool implementation for MCP server
//!
//! Provides directory indexing functionality with comprehensive error handling
//! and integration with BM25Searcher's indexing capabilities.

use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use serde::Deserialize;

use crate::mcp::{McpError, McpResult};
use crate::mcp::protocol::JsonRpcResponse;
use crate::mcp::types::IndexResponse;
use crate::search::{BM25Searcher, UnifiedSearchAdapter};

/// Parameters for index_directory tool
#[derive(Debug, Deserialize)]
struct IndexDirectoryParams {
    /// Directory path to index
    directory_path: String,
    /// Whether to include test files (optional)
    include_test_files: Option<bool>,
    /// File patterns to include (optional)
    include_patterns: Option<Vec<String>>,
    /// File patterns to exclude (optional)
    exclude_patterns: Option<Vec<String>>,
}

/// Execute index_directory tool
/// Indexes specified directory with all files matching criteria
pub async fn execute_index_directory(
    searcher: &Arc<RwLock<BM25Searcher>>,
    params: &serde_json::Value,
    id: Option<serde_json::Value>
) -> McpResult<JsonRpcResponse> {
    // Parse and validate parameters
    let index_params: IndexDirectoryParams = serde_json::from_value(params.clone())
        .map_err(|e| McpError::InvalidParams {
            message: format!("Invalid index_directory parameters: {}. Required: directory_path (string). Optional: include_test_files (bool), include_patterns (array), exclude_patterns (array)", e)
        })?;
    
    // Validate directory path exists
    let dir_path = PathBuf::from(&index_params.directory_path);
    if !dir_path.exists() {
        return Err(McpError::InvalidParams {
            message: format!("Directory does not exist: {}", index_params.directory_path)
        });
    }
    
    if !dir_path.is_dir() {
        return Err(McpError::InvalidParams {
            message: format!("Path is not a directory: {}", index_params.directory_path)
        });
    }
    
    println!("📂 MCP Tool: Indexing directory {:?} (include_test_files: {:?})", 
             dir_path, index_params.include_test_files);
    
    // Execute indexing operation
    let start_time = std::time::Instant::now();
    
    // Discover files in directory
    let files = discover_files(&dir_path, &index_params)?;
    println!("📁 Found {} files to index", files.len());
    
    let mut files_indexed = 0;
    let mut chunks_created = 0; 
    let mut errors = 0;
    
    // Index each file
    for file_path in files {
        match index_single_file(searcher, &file_path).await {
            Ok((file_chunks, _)) => {
                files_indexed += 1;
                chunks_created += file_chunks;
            }
            Err(e) => {
                errors += 1;
                log::warn!("Failed to index {}: {}", file_path.display(), e);
            }
        }
    }
    
    let stats = crate::mcp::types::IndexResponse {
        files_indexed,
        chunks_created,
        symbols_extracted: 0, // Not implemented yet
        errors,
        index_time_ms: 0, // Will be set below
        index_stats: None,
    };
    
    let index_time = start_time.elapsed();
    
    // Create response
    let response = IndexResponse {
        files_indexed: stats.files_indexed,
        chunks_created: stats.chunks_created,
        symbols_extracted: 0, // Not tracked in current implementation
        errors: stats.errors,
        index_time_ms: index_time.as_millis() as u64,
        index_stats: None, // Could be extended if needed
    };
    
    println!("✅ MCP Tool: Successfully indexed {} files, created {} chunks in {:?}",
             response.files_indexed, response.chunks_created, index_time);
    
    JsonRpcResponse::success(response, id)
}

/// Execute unified indexing with BM25 + embeddings when available  
pub async fn execute_unified_index_directory(
    unified_adapter: &Arc<UnifiedSearchAdapter>,
    params: &serde_json::Value,
    id: Option<serde_json::Value>
) -> McpResult<JsonRpcResponse> {
    // Parse and validate parameters
    let index_params: IndexDirectoryParams = serde_json::from_value(params.clone())
        .map_err(|e| McpError::InvalidParams {
            message: format!("Invalid index_directory parameters: {}. Required: directory_path (string). Optional: include_test_files (bool), include_patterns (array), exclude_patterns (array)", e)
        })?;
    
    // Validate directory path exists
    let dir_path = PathBuf::from(&index_params.directory_path);
    if !dir_path.exists() {
        return Err(McpError::InvalidParams {
            message: format!("Directory does not exist: {}", index_params.directory_path)
        });
    }
    
    if !dir_path.is_dir() {
        return Err(McpError::InvalidParams {
            message: format!("Path is not a directory: {}", index_params.directory_path)
        });
    }
    
    println!("📂 MCP Tool: Unified indexing directory {:?} (include_test_files: {:?}, backends: {})", 
             dir_path, index_params.include_test_files,
             if unified_adapter.has_embeddings() { "BM25 + Semantic" } else { "BM25 only" });
    
    // Execute indexing operation
    let start_time = std::time::Instant::now();
    
    // Discover files in directory
    let files = discover_files(&dir_path, &index_params)?;
    println!("📁 Found {} files to index with unified adapter", files.len());
    
    let mut files_indexed = 0;
    let mut chunks_created = 0; 
    let mut errors = 0;
    
    // Index each file with unified adapter
    for file_path in files {
        match unified_adapter.index_file(&file_path).await {
            Ok(()) => {
                files_indexed += 1;
                chunks_created += 1; // Estimate - unified adapter doesn't return chunk count
            }
            Err(e) => {
                errors += 1;
                log::warn!("Failed to index {} with unified adapter: {}", file_path.display(), e);
            }
        }
    }
    
    let index_time = start_time.elapsed();
    
    // Create response
    let response = IndexResponse {
        files_indexed,
        chunks_created,
        symbols_extracted: 0, // Not tracked in current implementation
        errors,
        index_time_ms: index_time.as_millis() as u64,
        index_stats: None, // Could be extended if needed
    };
    
    println!("✅ MCP Tool: Successfully indexed {} files using unified adapter in {:?}",
             response.files_indexed, index_time);
    
    JsonRpcResponse::success(response, id)
}

/// Discover files in directory based on parameters
fn discover_files(dir_path: &PathBuf, params: &IndexDirectoryParams) -> Result<Vec<PathBuf>, McpError> {
    use walkdir::WalkDir;
    
    let mut files = Vec::new();
    let include_test_files = params.include_test_files.unwrap_or(true);
    
    for entry in WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        
        let path = entry.path();
        let path_str = path.to_string_lossy();
        
        // Skip test files if not requested
        if !include_test_files && (path_str.contains("test") || path_str.contains("spec")) {
            continue;
        }
        
        // Check file extension for supported types
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            if matches!(ext_str.as_ref(), "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | "h" | "hpp" | "md" | "txt") {
                files.push(path.to_path_buf());
            }
        }
    }
    
    Ok(files)
}

/// Index a single file with BM25Searcher (legacy function)
async fn index_single_file(
    searcher: &Arc<RwLock<BM25Searcher>>, 
    file_path: &PathBuf
) -> Result<(usize, usize), McpError> {
    let mut searcher_guard = searcher.write().await;
    
    match searcher_guard.add_document_from_file(&file_path.to_string_lossy()) {
        Ok(()) => Ok((1, 0)), // 1 file, 0 symbols (not tracked)
        Err(e) => Err(McpError::InternalError {
            message: format!("Failed to index file: {}", e)
        })
    }
}
