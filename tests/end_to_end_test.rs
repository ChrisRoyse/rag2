/// End-to-end test to verify the RAG system actually works
/// Truth: This is the first real test that actually validates functionality

use anyhow::Result;
use std::path::PathBuf;

#[cfg(feature = "ml")]
#[tokio::test]
async fn test_embeddings_actually_work() -> Result<()> {
    use embed_search::config::Config;
    use embed_search::embedding::{NomicEmbedder, EmbedderTrait};
    
    // Initialize config pointing to the actual model file
    std::env::set_var("EMBED_MODEL_PATH", "./model/nomic-embed-code.Q4_K_M.gguf");
    Config::init_test()?;
    
    // Create embedder
    let embedder = NomicEmbedder::new()?;
    
    // Test with actual code
    let test_code = "fn main() { println!(\"Hello, world!\"); }";
    
    // Generate embedding
    let embedding = embedder.embed(test_code).await?;
    
    // Verify embedding has correct dimensions
    assert_eq!(embedding.len(), 768, "Nomic embeddings should be 768-dimensional");
    
    // Verify embedding has reasonable values (not all zeros or NaN)
    let non_zero_count = embedding.iter().filter(|&&x| x != 0.0).count();
    assert!(non_zero_count > 700, "Embedding should have mostly non-zero values");
    
    let finite_count = embedding.iter().filter(|&&x| x.is_finite()).count();
    assert_eq!(finite_count, 768, "All embedding values should be finite");
    
    println!("✅ Embeddings work! First 10 values: {:?}", &embedding[..10]);
    
    Ok(())
}

#[tokio::test] 
async fn test_lightweight_storage() -> Result<()> {
    use embed_search::storage::lightweight_storage::{LightweightStorage, LightweightRecord};
    
    let storage = LightweightStorage::new();
    
    // Create test records
    let records = vec![
        LightweightRecord {
            id: "1".to_string(),
            file_path: "test.rs".to_string(),
            chunk_index: 0,
            content: "fn main() {}".to_string(),
            embedding: vec![0.1; 768], // Mock embedding
            start_line: 1,
            end_line: 1,
        },
        LightweightRecord {
            id: "2".to_string(),
            file_path: "test.rs".to_string(),
            chunk_index: 1,
            content: "println!(\"hello\");".to_string(),
            embedding: vec![0.2; 768], // Different mock embedding
            start_line: 2,
            end_line: 2,
        },
    ];
    
    // Insert records
    storage.insert_batch(records).await?;
    
    // Search for similar
    let query_embedding = vec![0.15; 768]; // Between the two embeddings
    let results = storage.search_similar(query_embedding, 2).await?;
    
    assert_eq!(results.len(), 2, "Should return both records");
    assert_eq!(results[0].id, "2", "Closer embedding should be first");
    
    // Delete by file
    storage.delete_by_file("test.rs").await?;
    let count = storage.count().await?;
    assert_eq!(count, 0, "All records should be deleted");
    
    println!("✅ Lightweight storage works!");
    
    Ok(())
}

#[tokio::test]
async fn test_basic_search_without_ml() -> Result<()> {
    use embed_search::search::simple_searcher::SimpleSearcher;
    use embed_search::search::SearchConfig;
    use std::path::Path;
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new()?;
    let config = SearchConfig::default();
    
    // Create test file
    let test_file = temp_dir.path().join("test.rs");
    std::fs::write(&test_file, "fn hello_world() { println!(\"Hello!\"); }")?;
    
    let searcher = SimpleSearcher::new(temp_dir.path().to_path_buf(), config)?;
    searcher.index_file(&test_file)?;
    
    // Search for function
    let results = searcher.search("hello_world")?;
    assert!(!results.is_empty(), "Should find the function");
    assert!(results[0].content.contains("hello_world"));
    
    println!("✅ Basic search works!");
    
    Ok(())
}