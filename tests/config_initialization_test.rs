use embed_search::config::Config;
use anyhow::Result;

#[test]
fn test_config_from_file() -> Result<()> {
    // Test loading from actual config file
    let config = Config::load_from_file(".embed/config.toml")?;
    
    // Verify config loaded correctly
    assert_eq!(config.embedding.dimension, 768);
    assert_eq!(config.embedding.model_type, "nomic");
    assert!(config.embedding.model_path.to_str().unwrap().contains("nomic-embed-code"));
    
    config.validate()?;
    Ok(())
}

#[test]
fn test_config_validation() -> Result<()> {
    let mut config = Config::load_from_file(".embed/config.toml")?;
    
    // Test invalid dimension
    config.embedding.dimension = 0;
    assert!(config.validate().is_err());
    
    // Test invalid batch size  
    config.embedding.dimension = 768;
    config.embedding.batch_size = 0;
    assert!(config.validate().is_err());
    
    // Test valid config
    config.embedding.batch_size = 32;
    assert!(config.validate().is_ok());
    
    Ok(())
}

#[test]
fn test_config_thread_safety() -> Result<()> {
    use std::sync::Arc;
    use std::thread;
    
    let config = Arc::new(Config::load_from_file(".embed/config.toml")?);
    
    let mut handles = vec![];
    
    // Test concurrent access to config
    for _ in 0..10 {
        let config_clone = config.clone();
        let handle = thread::spawn(move || {
            // Just access the config fields
            let _dim = config_clone.embedding.dimension;
            let _batch = config_clone.embedding.batch_size;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    Ok(())
}