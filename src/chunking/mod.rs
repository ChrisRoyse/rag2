pub mod regex_chunker;
pub mod line_validator;
pub mod three_chunk;

pub use regex_chunker::{SimpleRegexChunker, Chunk};
pub use line_validator::{LineValidator, ValidationError};
pub use three_chunk::{ThreeChunkExpander, ChunkContext, ExpansionError};

/// Utility function to chunk code content into smaller pieces
/// This is a convenience function for use in the MCP search integration
pub fn chunk_code_content(content: &str, chunk_size_target: usize, overlap: usize) -> anyhow::Result<Vec<Chunk>> {
    let chunker = SimpleRegexChunker::with_chunk_size(chunk_size_target)
        .map_err(|e| anyhow::anyhow!("Failed to create chunker: {}", e))?;
    Ok(chunker.chunk_file(content))
}