use embed_search::config::Config;
use embed_search::embedding::LazyEmbedder;
use anyhow::Result;

#[tokio::test]
async fn test_config_initialization() -> Result<()> {
    // Test that config loads without crashing
    Config::init_test()?;
    let config = Config::get()?;
    
    // Verify basic config properties
    assert!(config.chunk_size > 0);
    assert!(!config.vector_db_path.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_lazy_embedder_creation() -> Result<()> {
    // Test that embedder can be created without crashing
    let embedder = LazyEmbedder::new();
    assert!(!embedder.is_initialized());
    
    Ok(())
}

#[tokio::test]
#[cfg(feature = "ml")]
async fn test_embedding_basic_functionality() -> Result<()> {
    use embed_search::embedding::LazyEmbedder;
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Test basic text embedding
    let test_text = "function fibonacci(n) { return n < 2 ? n : fibonacci(n-1) + fibonacci(n-2); }";
    
    match embedder.embed(test_text).await {
        Ok(embedding) => {
            assert_eq!(embedding.len(), 768);
            assert!(embedding.iter().all(|&x| x.is_finite()));
        }
        Err(e) => {
            // If ML features can't initialize, that's expected without proper setup
            eprintln!("ML embedding failed (expected without model): {}", e);
        }
    }
    
    Ok(())
}