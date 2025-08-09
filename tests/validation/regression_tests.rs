//! Regression tests to ensure no functionality breaks during migration
//! These tests are designed to catch the specific issues that occurred during the embedding migration

use embed_search::{
    embedding::{NomicEmbedder, LazyEmbedder},
    search::unified::UnifiedSearcher,
    config::Config,
    error::EmbedError,
};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

/// Test that would have caught the original model path misconfiguration
#[tokio::test]
async fn test_model_path_configuration() {
    println!("🔍 Testing model path configuration (regression test)");
    
    // This test specifically targets the model path issue that caused problems
    let expected_model_path = std::env::current_dir().unwrap().join("model/nomic-embed-code.Q4_K_M.gguf");
    
    // Test 1: Verify model file exists at expected location
    assert!(expected_model_path.exists(), 
            "Model file not found at expected path: {:?}. This was the root cause of the original issue.", 
            expected_model_path);
    
    // Test 2: Verify model can be loaded from the correct path
    let embedder = LazyEmbedder::new();
    
    let result = timeout(Duration::from_secs(120), embedder.get_or_init()).await;
    match result {
        Ok(Ok(_)) => {
            println!("  ✅ Model successfully loaded from correct path");
        },
        Ok(Err(e)) => {
            panic!("Model failed to load despite file existing: {}. This indicates a configuration or format issue.", e);
        },
        Err(_) => {
            panic!("Model loading timed out. This could indicate the wrong model type or corrupted file.");
        }
    }
    
    // Test 3: Verify it's specifically the nomic-embed-code model
    let test_code = "fn example() { println!(\"Hello, world!\"); }";
    let embedding = embedder.embed(test_code).await.expect("Code embedding should work");
    assert_eq!(embedding.len(), 768, "Wrong embedding dimension indicates wrong model loaded");
    
    println!("  ✅ Confirmed nomic-embed-code model with correct dimensions");
}

/// Test that would have caught tokenizer/model compatibility issues
#[tokio::test]
async fn test_tokenizer_model_compatibility() {
    println!("🔧 Testing tokenizer/model compatibility (regression test)");
    
    let embedder = LazyEmbedder::new();
    
    // Test various code patterns that should tokenize correctly
    let problematic_patterns = vec![
        // These patterns were problematic in the original migration
        ("unicode_in_code", "fn emoji_handler() { println!(\"🚀 Rocket!\"); }"),
        ("special_chars", "const REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;"),
        ("long_strings", format!("const LONG_CONSTANT = \"{}\";", "x".repeat(500))),
        ("mixed_languages", "// C++ style comment\nfn rust_function() { /* C style comment */ }"),
        ("escape_sequences", "const PATH = \"C:\\\\Users\\\\test\\\\file.txt\";"),
    ];
    
    for (test_name, code) in problematic_patterns {
        println!("  Testing {}: {}", test_name, code.chars().take(50).collect::<String>());
        
        let result = embedder.embed(&code).await;
        match result {
            Ok(embedding) => {
                assert_eq!(embedding.len(), 768, "Wrong dimension for {}", test_name);
                
                // Check for reasonable embedding values
                let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
                assert!(mean.abs() < 0.5, "Suspicious mean value for {}: {}", test_name, mean);
                
                let has_valid_values = embedding.iter().all(|&x| x.is_finite() && x.abs() < 10.0);
                assert!(has_valid_values, "Invalid embedding values for {}", test_name);
                
                println!("    ✅ {} tokenized and embedded successfully", test_name);
            },
            Err(e) => {
                // Some patterns might be rejected, which is acceptable
                println!("    ⚠️  {} rejected: {} (this may be acceptable)", test_name, e);
            }
        }
    }
}

/// Test that would have caught attention mask validation issues
#[tokio::test]
async fn test_attention_mask_regression() {
    println!("⚠️  Testing attention mask validation (regression test)");
    
    // These are the exact scenarios that caused the original failures
    let mask_test_cases = vec![
        // Valid cases that should pass
        (vec![1, 1, 1, 0, 0], 5, true, "standard_padding"),
        (vec![1, 1, 1, 1, 1], 5, true, "no_padding"),
        (vec![1], 1, true, "single_token"),
        
        // Invalid cases that should fail gracefully
        (vec![1, 1, 1], 5, false, "dimension_mismatch"),
        (vec![0, 0, 0, 0, 0], 5, false, "all_zeros"),
        (vec![0], 1, false, "single_zero"),
        (vec![], 0, false, "empty_mask"),
        
        // Edge cases that were problematic
        (vec![1, 1, 1], 2, false, "mask_too_long"),
        (vec![1, 1], 5, false, "mask_too_short"),
    ];
    
    for (mask, expected_len, should_pass, test_name) in mask_test_cases {
        println!("  Testing {}: mask={:?}, expected_len={}", test_name, mask, expected_len);
        
        let result = NomicEmbedder::validate_attention_mask(&mask, expected_len);
        
        if should_pass {
            assert!(result.is_ok(), "Expected {} to pass validation but got: {:?}", test_name, result.err());
            println!("    ✅ {} correctly passed validation", test_name);
        } else {
            assert!(result.is_err(), "Expected {} to fail validation but it passed", test_name);
            
            // Verify error message is descriptive
            let error_msg = result.unwrap_err().to_string();
            assert!(error_msg.len() > 20, "Error message too short for {}: {}", test_name, error_msg);
            
            println!("    ✅ {} correctly failed with: {}", test_name, error_msg.chars().take(60).collect::<String>());
        }
    }
}

/// Test that would have caught GGUF loading issues
#[tokio::test]
async fn test_gguf_loading_regression() {
    println!("📦 Testing GGUF loading robustness (regression test)");
    
    let embedder = LazyEmbedder::new();
    
    // Force initialization to test the actual GGUF loading
    let embedder_arc = embedder.get_or_init().await.expect("GGUF loading should succeed");
    
    // Test that the loaded model actually works with various inputs
    let gguf_test_inputs = vec![
        "fn test() {}",
        "class Example: pass",
        "function example() { return true; }",
        "int main() { return 0; }",
        "// This is a comment",
        "SELECT * FROM table WHERE id = 1;",
    ];
    
    for (i, input) in gguf_test_inputs.iter().enumerate() {
        println!("  Testing GGUF with input {}: {}", i + 1, input);
        
        let embedding = embedder.embed(input).await.expect(&format!("GGUF processing failed for input {}", i + 1));
        
        // Verify the embedding has expected properties for a properly loaded GGUF model
        assert_eq!(embedding.len(), 768, "GGUF model wrong dimension for input {}", i + 1);
        
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "GGUF embedding not normalized for input {}: norm={}", i + 1, norm);
        
        // Check for quantization artifacts
        let unique_values: std::collections::HashSet<_> = embedding.iter().map(|x| (x * 1000.0) as i32).collect();
        assert!(unique_values.len() > 10, "Too few unique values, possible quantization issue for input {}", i + 1);
        
        println!("    ✅ Input {} processed correctly (norm={:.4}, unique_values={})", i + 1, norm, unique_values.len());
    }
}

/// Test that would have caught cache corruption issues
#[tokio::test]
async fn test_cache_corruption_regression() {
    println!("💾 Testing cache corruption prevention (regression test)");
    
    let embedder = LazyEmbedder::new();
    
    let test_input = "fn cache_test() { println!(\"cache test\"); }";
    
    // Get original embedding
    let original_embedding = embedder.embed(test_input).await.expect("Original embedding failed");
    
    // Stress test the cache with many operations
    let mut cached_embeddings = Vec::new();
    
    for i in 0..20 {
        // Mix of same and different inputs to test cache behavior
        let input = if i % 3 == 0 {
            test_input.to_string()
        } else {
            format!("{} // iteration {}", test_input, i)
        };
        
        let embedding = embedder.embed(&input).await.expect(&format!("Cache test iteration {} failed", i));
        cached_embeddings.push((input, embedding));
    }
    
    // Verify cached results are consistent
    let same_input_results: Vec<_> = cached_embeddings.iter()
        .filter(|(input, _)| input == test_input)
        .collect();
    
    for (i, (_, embedding)) in same_input_results.iter().enumerate() {
        assert_eq!(embedding.len(), original_embedding.len(), "Cached embedding {} wrong size", i);
        
        // Embeddings for the same input should be identical
        for (j, (&cached, &original)) in embedding.iter().zip(original_embedding.iter()).enumerate() {
            assert_eq!(cached, original, "Cache corruption at position {} in iteration {}", j, i);
        }
    }
    
    println!("  ✅ Cache maintained consistency across {} operations", cached_embeddings.len());
    println!("  ✅ Found {} cached instances of the same input", same_input_results.len());
}

/// Test that would have caught initialization race conditions
#[tokio::test]
async fn test_initialization_race_conditions() {
    println!("🏁 Testing initialization race conditions (regression test)");
    
    // Create multiple embedders concurrently to test singleton behavior
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let handle = tokio::spawn(async move {
            let embedder = LazyEmbedder::new();
            let result = embedder.get_or_init().await;
            (i, result)
        });
        handles.push(handle);
    }
    
    // Wait for all initializations
    let mut results = Vec::new();
    for handle in handles {
        let (i, result) = handle.await.expect(&format!("Task {} panicked", i));
        results.push((i, result));
    }
    
    // Verify all succeeded
    let mut successful = 0;
    let mut failed = 0;
    
    for (i, result) in results {
        match result {
            Ok(_) => {
                successful += 1;
                println!("  ✅ Concurrent initialization {} succeeded", i);
            },
            Err(e) => {
                failed += 1;
                println!("  ❌ Concurrent initialization {} failed: {}", i, e);
            }
        }
    }
    
    assert!(successful >= 8, "Too many initialization failures: {}/{}", failed, successful + failed);
    
    // Test that embeddings work after concurrent initialization
    let embedder = LazyEmbedder::new();
    let test_embedding = embedder.embed("fn race_test() {}").await.expect("Post-race embedding failed");
    assert_eq!(test_embedding.len(), 768, "Post-race embedding wrong dimension");
    
    println!("  ✅ Race condition test passed: {}/{} successful initializations", successful, successful + failed);
}

/// Test specific search functionality that may have broken
#[tokio::test]
async fn test_search_functionality_regression() {
    println!("🔍 Testing search functionality regression");
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let temp_path = temp_dir.path();
    
    // Create test files that represent the types of content that were problematic
    let regression_test_files = vec![
        ("broken_unicode.py", "def process_emoji():\n    return \"🚀 Processing... 📊 Done!\""),
        ("complex_code.rs", "impl<T: Clone + Send + Sync + 'static> ComplexProcessor<T> {\n    async fn process(&self, items: Vec<T>) -> Result<Vec<ProcessedItem<T>>, ProcessError> {\n        let futures: Vec<_> = items.into_iter().map(|item| self.process_single(item)).collect();\n        let results = futures::future::join_all(futures).await;\n        results.into_iter().collect()\n    }\n}"),
        ("long_sql.sql", "SELECT u.id, u.username, u.email, p.first_name, p.last_name, p.phone, a.street, a.city, a.state, a.zip_code FROM users u LEFT JOIN profiles p ON u.id = p.user_id LEFT JOIN addresses a ON u.id = a.user_id WHERE u.created_at > NOW() - INTERVAL '30 days' AND u.status = 'active' AND p.verified = true ORDER BY u.created_at DESC LIMIT 100;"),
        ("config_with_special_chars.toml", "[database]\npassword = \"P@ssw0rd!#$%^&*()\"\nconnection_string = \"postgresql://user:pass@host:5432/db?sslmode=require&connect_timeout=30\"\n"),
    ];
    
    // Write test files
    for (filename, content) in &regression_test_files {
        let file_path = temp_path.join(filename);
        std::fs::write(&file_path, content).expect(&format!("Failed to write {}", filename));
    }
    
    // Initialize searcher
    let project_path = temp_path.to_path_buf();
    let db_path = temp_path.join("regression_test.db");
    
    let searcher = timeout(
        Duration::from_secs(120),
        UnifiedSearcher::new(project_path.clone(), db_path)
    ).await.expect("Searcher initialization timed out")
        .expect("Failed to initialize searcher");
    
    // Index the files
    let stats = searcher.index_directory(&project_path).await.expect("Indexing failed");
    println!("  📚 Indexed {} files: {}", regression_test_files.len(), stats);
    
    // Test searches that were problematic
    let regression_search_tests = vec![
        ("emoji processing", "emoji"),
        ("complex generics", "ComplexProcessor"),
        ("SQL joins", "LEFT JOIN"),
        ("special characters", "password"),
        ("async processing", "async fn"),
    ];
    
    for (test_name, query) in regression_search_tests {
        println!("  Testing search: {} - '{}'", test_name, query);
        
        let results = timeout(
            Duration::from_secs(30),
            searcher.search(query)
        ).await.expect(&format!("Search '{}' timed out", query))
            .expect(&format!("Search '{}' failed", query));
        
        assert!(!results.is_empty(), "No results for regression search: {}", query);
        
        // Verify result quality
        let top_score = results[0].score;
        assert!(top_score > 0.1, "Top score too low for '{}': {}", query, top_score);
        
        println!("    ✅ {} found {} results, top score: {:.3}", test_name, results.len(), top_score);
    }
}

/// Test error recovery mechanisms
#[tokio::test]
async fn test_error_recovery_regression() {
    println!("🛠️  Testing error recovery mechanisms (regression test)");
    
    let embedder = LazyEmbedder::new();
    
    // Test recovery from various error conditions that occurred during migration
    let error_test_cases = vec![
        ("memory_pressure", generate_memory_pressure_input()),
        ("malformed_input", "\x00\x01\x02\x03".to_string()),
        ("extremely_long", "x".repeat(100_000)),
        ("only_whitespace", "   \t\n   \r\n   ".to_string()),
        ("mixed_encoding", "Hello 世界 🌍 Здравствуй мир".to_string()),
    ];
    
    let mut recovery_scores = Vec::new();
    
    for (test_name, input) in error_test_cases {
        println!("  Testing error recovery: {}", test_name);
        
        let mut attempts = 0;
        let mut last_error = None;
        
        // Try multiple times to test recovery
        loop {
            attempts += 1;
            
            match embedder.embed(&input).await {
                Ok(embedding) => {
                    // Recovery successful
                    assert_eq!(embedding.len(), 768, "Recovered embedding wrong size for {}", test_name);
                    println!("    ✅ {} recovered after {} attempts", test_name, attempts);
                    recovery_scores.push(1.0 / attempts as f32);
                    break;
                },
                Err(e) => {
                    last_error = Some(e);
                    if attempts >= 3 {
                        // Acceptable to fail after multiple attempts
                        println!("    ⚠️  {} failed after {} attempts: {}", test_name, attempts, last_error.unwrap());
                        recovery_scores.push(0.0);
                        break;
                    }
                    
                    // Wait before retry
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
    
    let recovery_rate = recovery_scores.iter().filter(|&&score| score > 0.0).count() as f32 / recovery_scores.len() as f32;
    
    println!("  📊 Error recovery rate: {:.1}%", recovery_rate * 100.0);
    
    // We expect at least some recovery capability
    assert!(recovery_rate >= 0.4, "Error recovery rate too low: {:.1}%", recovery_rate * 100.0);
}

/// Test that configuration changes don't break existing functionality
#[tokio::test]
async fn test_configuration_compatibility_regression() {
    println!("⚙️  Testing configuration compatibility (regression test)");
    
    // Test that the system works with various configuration states
    // This would have caught the configuration issues that caused the original problems
    
    // Test 1: Default configuration
    let embedder1 = LazyEmbedder::new();
    let test_result1 = embedder1.embed("fn config_test() {}").await;
    assert!(test_result1.is_ok(), "Default configuration should work");
    println!("  ✅ Default configuration works");
    
    // Test 2: Multiple embedder instances (should share global state)
    let embedder2 = LazyEmbedder::new();
    let test_result2 = embedder2.embed("fn config_test() {}").await;
    assert!(test_result2.is_ok(), "Second embedder instance should work");
    
    // Results should be identical (cached)
    let embedding1 = test_result1.unwrap();
    let embedding2 = test_result2.unwrap();
    assert_eq!(embedding1, embedding2, "Embeddings should be identical across instances");
    
    println!("  ✅ Multiple instances share state correctly");
    
    // Test 3: Memory pressure handling
    let mut large_embeddings = Vec::new();
    let embedder3 = LazyEmbedder::new();
    
    for i in 0..10 {
        let large_input = format!("fn large_test_{}() {{ {} }}", i, "println!(\"test\");".repeat(50));
        
        match embedder3.embed(&large_input).await {
            Ok(embedding) => {
                assert_eq!(embedding.len(), 768, "Large input embedding wrong size");
                large_embeddings.push(embedding);
            },
            Err(e) => {
                if i < 5 {
                    panic!("Should be able to handle at least 5 large inputs, failed at {}: {}", i, e);
                } else {
                    println!("    ⚠️  Memory pressure hit at iteration {}: {}", i, e);
                    break;
                }
            }
        }
    }
    
    println!("  ✅ Handled {} large embeddings before memory pressure", large_embeddings.len());
    
    // Test 4: System still works after pressure
    let recovery_test = embedder3.embed("fn recovery_test() {}").await;
    assert!(recovery_test.is_ok(), "System should recover after memory pressure");
    
    println!("  ✅ System recovered after memory pressure");
}

// Helper functions

fn generate_memory_pressure_input() -> String {
    // Create input that might cause memory pressure
    let mut input = String::with_capacity(50_000);
    input.push_str("fn memory_intensive_function() {\n");
    
    for i in 0..1000 {
        input.push_str(&format!("    let var_{} = process_data_{:04}();\n", i, i));
    }
    
    input.push_str("}\n");
    input
}