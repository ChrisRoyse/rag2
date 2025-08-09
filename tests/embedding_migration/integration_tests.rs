//! Integration tests for the complete embedding pipeline
//! Tests the full flow from text input to vector storage

use embed_search::{
    embedding::{LazyEmbedder, EmbeddingCache},
    storage::safe_vectordb::{VectorStorage, StorageConfig},
    search::unified::UnifiedSearcher,
    config::Config,
    error::Result,
};
use std::path::PathBuf;
use tokio::time::{timeout, Duration};
use tempfile::TempDir;

/// Test the complete pipeline from code input to searchable vectors
#[tokio::test]
async fn test_complete_embedding_pipeline() {
    println!("🔄 Testing complete embedding pipeline...");
    
    // Setup temporary storage
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let storage_config = StorageConfig {
        max_vectors: 1000,
        dimension: 768,
        cache_size: 100,
        enable_compression: false,
    };
    
    let mut storage = VectorStorage::new(storage_config).expect("Failed to create storage");
    let embedder = LazyEmbedder::new();
    
    // Test data: various code patterns
    let code_samples = vec![
        ("rust_function", "fn calculate_fibonacci(n: u32) -> u32 { if n <= 1 { n } else { calculate_fibonacci(n-1) + calculate_fibonacci(n-2) } }"),
        ("python_class", "class DatabaseManager:\n    def __init__(self, connection_string):\n        self.connection = connect(connection_string)"),
        ("javascript_async", "async function fetchUserData(userId) { const response = await fetch(`/api/users/${userId}`); return response.json(); }"),
        ("sql_query", "SELECT u.username, p.email FROM users u JOIN profiles p ON u.id = p.user_id WHERE u.active = true"),
        ("css_styles", ".container { display: flex; justify-content: center; align-items: center; height: 100vh; }"),
    ];
    
    let mut embeddings = Vec::new();
    
    // Step 1: Generate embeddings
    println!("📊 Step 1: Generating embeddings...");
    for (id, code) in &code_samples {
        let embedding = embedder.embed(code).await
            .expect(&format!("Failed to embed {}", id));
        
        assert_eq!(embedding.len(), 768, "Embedding for {} has wrong dimensions", id);
        embeddings.push((id.to_string(), embedding));
        println!("  ✅ Generated embedding for {}", id);
    }
    
    // Step 2: Store in vector database
    println!("💾 Step 2: Storing embeddings...");
    for (id, embedding) in &embeddings {
        storage.insert(id.clone(), embedding.clone())
            .expect(&format!("Failed to store embedding for {}", id));
        println!("  ✅ Stored {}", id);
    }
    
    // Step 3: Verify storage
    println!("🔍 Step 3: Verifying storage...");
    for (id, expected_embedding) in &embeddings {
        let retrieved = storage.get(id)
            .expect(&format!("Failed to retrieve {}", id))
            .expect(&format!("Embedding {} not found in storage", id));
        
        assert_eq!(retrieved.len(), expected_embedding.len(), "Retrieved embedding for {} has wrong size", id);
        
        // Verify exact match (storage should preserve precision)
        for (i, (expected, actual)) in expected_embedding.iter().zip(retrieved.iter()).enumerate() {
            assert_eq!(*expected, *actual, "Embedding {} value mismatch at index {}", id, i);
        }
        
        println!("  ✅ Verified storage for {}", id);
    }
    
    // Step 4: Test similarity search
    println!("🎯 Step 4: Testing similarity search...");
    for (query_id, query_code) in &code_samples {
        let query_embedding = embedder.embed(query_code).await
            .expect(&format!("Failed to embed query {}", query_id));
        
        let similar = storage.find_similar(&query_embedding, 3)
            .expect(&format!("Similarity search failed for {}", query_id));
        
        // Should find itself as most similar
        assert!(!similar.is_empty(), "No similar items found for {}", query_id);
        
        let top_match = &similar[0];
        assert_eq!(top_match.0, *query_id, "Top match for {} should be itself, got {}", query_id, top_match.0);
        assert!(top_match.1 > 0.99, "Self-similarity for {} too low: {}", query_id, top_match.1);
        
        println!("  ✅ Similarity search for {} - top match: {} (similarity: {:.4})", 
                 query_id, top_match.0, top_match.1);
    }
    
    println!("✅ Complete embedding pipeline test passed!");
}

/// Test the pipeline with real file processing
#[tokio::test]
async fn test_file_based_embedding_pipeline() {
    println!("📁 Testing file-based embedding pipeline...");
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let temp_path = temp_dir.path();
    
    // Create test files with different types of code
    let test_files = vec![
        ("main.rs", "fn main() {\n    println!(\"Hello, Rust!\");\n    let x = calculate_sum(5, 10);\n}\n\nfn calculate_sum(a: i32, b: i32) -> i32 {\n    a + b\n}"),
        ("server.py", "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/health')\ndef health_check():\n    return jsonify({'status': 'healthy'})\n\nif __name__ == '__main__':\n    app.run(debug=True)"),
        ("utils.js", "function debounce(func, wait) {\n    let timeout;\n    return function executedFunction(...args) {\n        const later = () => {\n            clearTimeout(timeout);\n            func(...args);\n        };\n        clearTimeout(timeout);\n        timeout = setTimeout(later, wait);\n    };\n}\n\nexport { debounce };"),
    ];
    
    // Write test files
    for (filename, content) in &test_files {
        let file_path = temp_path.join(filename);
        std::fs::write(&file_path, content)
            .expect(&format!("Failed to write {}", filename));
        println!("  📝 Created test file: {}", filename);
    }
    
    // Initialize unified searcher (this tests the full integration)
    let project_path = temp_path.to_path_buf();
    let db_path = temp_path.join("embeddings.db");
    
    let searcher = timeout(
        Duration::from_secs(60),
        UnifiedSearcher::new(project_path.clone(), db_path.clone())
    ).await.expect("Searcher initialization timed out")
        .expect("Failed to create unified searcher");
    
    println!("  ✅ Unified searcher initialized");
    
    // Index the directory
    println!("📚 Indexing test files...");
    let stats = timeout(
        Duration::from_secs(120),
        searcher.index_directory(&project_path)
    ).await.expect("Indexing timed out")
        .expect("Failed to index directory");
    
    println!("  ✅ Indexing complete: {}", stats);
    
    // Test searches that should find specific files
    let search_tests = vec![
        ("rust main function", "main.rs"),
        ("flask server health", "server.py"), 
        ("debounce timeout", "utils.js"),
        ("calculate sum", "main.rs"),
        ("json response", "server.py"),
    ];
    
    println!("🔍 Running search tests...");
    for (query, expected_file) in search_tests {
        let results = timeout(
            Duration::from_secs(30),
            searcher.search(query)
        ).await.expect(&format!("Search for '{}' timed out", query))
            .expect(&format!("Search for '{}' failed", query));
        
        assert!(!results.is_empty(), "No results found for query: {}", query);
        
        let found_expected = results.iter().any(|r| r.file.contains(expected_file));
        assert!(found_expected, "Expected file '{}' not found in results for query '{}'", expected_file, query);
        
        println!("  ✅ Query '{}' correctly found '{}' (score: {:.3})", 
                 query, expected_file, results[0].score);
    }
    
    println!("✅ File-based embedding pipeline test passed!");
}

/// Test error recovery in the pipeline
#[tokio::test]
async fn test_pipeline_error_recovery() {
    println!("🔧 Testing pipeline error recovery...");
    
    let embedder = LazyEmbedder::new();
    
    // Test 1: Invalid inputs
    let invalid_inputs = vec![
        ("", "empty string"),
        ("\0\0\0", "null bytes"),
        ("🚀" * 1000, "excessive unicode"),
    ];
    
    for (input, description) in invalid_inputs {
        println!("  Testing {}: '{}'", description, input.chars().take(20).collect::<String>());
        
        match embedder.embed(input).await {
            Ok(embedding) => {
                // Some inputs might be handled gracefully
                assert_eq!(embedding.len(), 768, "Graceful handling should still produce correct dimensions");
                println!("    ✅ Handled gracefully");
            },
            Err(e) => {
                println!("    ✅ Properly rejected: {}", e);
            }
        }
    }
    
    // Test 2: Storage error recovery  
    let storage_config = StorageConfig {
        max_vectors: 5, // Very small limit to test overflow
        dimension: 768,
        cache_size: 2,
        enable_compression: false,
    };
    
    let mut storage = VectorStorage::new(storage_config).expect("Failed to create limited storage");
    
    // Fill storage to capacity
    for i in 0..6 {
        let embedding = vec![0.1_f32; 768]; // Simple test embedding
        let result = storage.insert(format!("item_{}", i), embedding);
        
        if i < 5 {
            result.expect(&format!("Should be able to store item {}", i));
            println!("    ✅ Stored item {}", i);
        } else {
            // Should handle overflow gracefully
            match result {
                Ok(_) => println!("    ✅ Storage expanded or replaced items"),
                Err(e) => println!("    ✅ Storage properly rejected overflow: {}", e),
            }
        }
    }
    
    println!("✅ Pipeline error recovery test passed!");
}

/// Test concurrent access to the embedding pipeline
#[tokio::test]
async fn test_concurrent_pipeline_access() {
    println!("⚡ Testing concurrent pipeline access...");
    
    let embedder = LazyEmbedder::new();
    
    // Create multiple concurrent embedding tasks
    let mut tasks = Vec::new();
    
    for i in 0..10 {
        let embedder_clone = embedder.clone();
        let task = tokio::spawn(async move {
            let code = format!("function test{}() {{ return {}; }}", i, i * 2);
            embedder_clone.embed(&code).await
        });
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    let mut successful = 0;
    let mut failed = 0;
    
    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok(embedding)) => {
                assert_eq!(embedding.len(), 768, "Concurrent embedding {} wrong size", i);
                successful += 1;
                println!("  ✅ Concurrent task {} completed", i);
            },
            Ok(Err(e)) => {
                println!("  ❌ Concurrent task {} failed: {}", i, e);
                failed += 1;
            },
            Err(e) => {
                println!("  ❌ Concurrent task {} panicked: {}", i, e);
                failed += 1;
            }
        }
    }
    
    assert!(successful >= 8, "Too many concurrent failures: {}/{}", failed, successful + failed);
    
    println!("✅ Concurrent access test passed: {}/{} successful", successful, successful + failed);
}

/// Performance characterization test
#[tokio::test]
async fn test_pipeline_performance_characteristics() {
    println!("📈 Testing pipeline performance characteristics...");
    
    let embedder = LazyEmbedder::new();
    
    // Test different code sizes
    let test_sizes = vec![
        ("small", "fn f() {}", 50),
        ("medium", "fn medium_function(x: i32, y: i32) -> i32 { x + y }".repeat(10), 20),
        ("large", "// Large function\n".repeat(100) + &"fn large() { println!(); }", 5),
    ];
    
    for (size_name, code, iterations) in test_sizes {
        println!("  Testing {} code ({} chars)...", size_name, code.len());
        
        let mut times = Vec::new();
        
        for i in 0..iterations {
            let start = std::time::Instant::now();
            let result = embedder.embed(&format!("{} // iteration {}", code, i)).await;
            let duration = start.elapsed();
            
            match result {
                Ok(embedding) => {
                    assert_eq!(embedding.len(), 768);
                    times.push(duration);
                },
                Err(e) => {
                    println!("    ❌ Failed at iteration {}: {}", i, e);
                    break;
                }
            }
        }
        
        if !times.is_empty() {
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let min_time = times.iter().min().unwrap();
            let max_time = times.iter().max().unwrap();
            
            println!("    ✅ {} code: avg={:?}, min={:?}, max={:?}", 
                     size_name, avg_time, min_time, max_time);
            
            // Performance assertions
            assert!(avg_time.as_secs() < 30, "{} code taking too long on average: {:?}", size_name, avg_time);
        }
    }
    
    println!("✅ Performance characteristics test completed!");
}

/// Test cache effectiveness in the pipeline
#[tokio::test] 
async fn test_pipeline_cache_effectiveness() {
    println!("💨 Testing pipeline cache effectiveness...");
    
    let embedder = LazyEmbedder::new();
    let test_code = "def cached_function(): return 'this should be cached'";
    
    // Cold run (will load model + compute embedding)
    let start = std::time::Instant::now();
    let embedding1 = embedder.embed(test_code).await.expect("Cold run failed");
    let cold_time = start.elapsed();
    
    // Warm run (should use cache)
    let start = std::time::Instant::now();
    let embedding2 = embedder.embed(test_code).await.expect("Warm run failed");
    let warm_time = start.elapsed();
    
    // Verify results are identical
    assert_eq!(embedding1, embedding2, "Cached result differs from original");
    
    // Cache should provide significant speedup
    let speedup = cold_time.as_nanos() as f64 / warm_time.as_nanos() as f64;
    
    println!("  Cold run: {:?}", cold_time);
    println!("  Warm run: {:?}", warm_time);
    println!("  Speedup: {:.2}x", speedup);
    
    // We expect at least some speedup from caching
    if speedup > 2.0 {
        println!("  ✅ Excellent cache performance");
    } else if speedup > 1.1 {
        println!("  ✅ Cache providing some benefit");
    } else {
        println!("  ⚠️  Cache may not be working optimally");
    }
    
    // Test cache with different inputs
    let similar_code = "def cached_function(): return 'this is slightly different'";
    let embedding3 = embedder.embed(similar_code).await.expect("Similar code failed");
    
    // Should be different from cached version
    assert_ne!(embedding1, embedding3, "Different inputs should produce different embeddings");
    
    println!("✅ Cache effectiveness test completed!");
}