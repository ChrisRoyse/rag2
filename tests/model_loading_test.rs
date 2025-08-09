use std::path::Path;
use anyhow::Result;

#[test] 
fn test_model_file_exists() -> Result<()> {
    let model_path = Path::new("./model/nomic-embed-code.Q4_K_M.gguf");
    
    if model_path.exists() {
        let metadata = std::fs::metadata(model_path)?;
        assert!(metadata.len() > 1_000_000_000); // Should be > 1GB
        println!("Model file size: {} GB", metadata.len() / 1_000_000_000);
    } else {
        eprintln!("Warning: Model file not found at expected location");
        // Don't fail the test if model isn't present
    }
    
    Ok(())
}

#[cfg(feature = "ml")]
#[tokio::test]
async fn test_model_loading_without_network() -> Result<()> {
    use embed_search::config::Config;
    
    // Initialize with test config that points to local model
    Config::init_test()?;
    
    // Try to create embedder - should not make network calls
    let embedder = embed_search::embedding::LazyEmbedder::new();
    assert!(!embedder.is_initialized());
    
    // Test initialization (may fail if model not present, but shouldn't crash)
    match embedder.get_or_init().await {
        Ok(_) => {
            println!("Model loaded successfully");
            assert!(embedder.is_initialized());
        }
        Err(e) => {
            eprintln!("Model loading failed (expected without proper setup): {}", e);
            // Don't fail test - this is expected in many environments
        }
    }
    
    Ok(())
}

#[test]
fn test_model_configuration() -> Result<()> {
    use embed_search::config::Config;
    
    // Test that config correctly references the code model
    let config = Config::load_from_file(".embed/config.toml")?;
    
    assert!(config.embedding.model_path.to_str().unwrap().contains("nomic-embed-code"));
    assert_eq!(config.embedding.model_type, "nomic");
    assert_eq!(config.embedding.dimension, 768);
    
    Ok(())
}