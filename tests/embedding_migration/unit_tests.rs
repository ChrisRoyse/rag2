//! Unit tests for nomic-embed-code integration
//! These tests will brutally expose any misconfiguration in the new embedding system

use embed_search::{
    embedding::{NomicEmbedder, LazyEmbedder, EmbeddingCache},
    error::{EmbedError, Result},
    config::Config,
};
use std::path::PathBuf;
use tokio::runtime::Runtime;
use anyhow::anyhow;

/// Test the specific nomic-embed-code model loading
#[tokio::test]
async fn test_nomic_code_model_loading() {
    // Test that the code-specific model loads correctly
    let model_path = PathBuf::from("./model/nomic-embed-code.Q4_K_M.gguf");
    
    if !model_path.exists() {
        panic!("Model file not found: {:?}. This test requires the nomic-embed-code model.", model_path);
    }
    
    // Test lazy embedder initialization
    let embedder = LazyEmbedder::new();
    assert!(!embedder.is_initialized(), "Embedder should not be initialized yet");
    
    // Force initialization and check for code-specific behavior
    let result = embedder.get_or_init().await;
    match result {
        Ok(embedder_arc) => {
            // Verify dimensions are correct for code model
            let dims = embedder_arc.dimensions();
            assert_eq!(dims, 768, "nomic-embed-code should have 768 dimensions, got {}", dims);
            
            println!("✅ Model loaded successfully with {} dimensions", dims);
        },
        Err(e) => panic!("Failed to initialize embedder: {}", e),
    }
}

/// Test embedding generation for code-specific content
#[tokio::test]
async fn test_code_embedding_generation() {
    let embedder = LazyEmbedder::new();
    
    // Test various code patterns that should work well with nomic-embed-code
    let code_samples = vec![
        "def calculate_hash(data: str) -> str:",
        "function createWebSocketConnection(url) {",
        "public class UserService extends BaseService {",
        "impl Iterator for CustomIterator {",
        "SELECT users.id, profiles.name FROM users JOIN profiles",
        "const API_ENDPOINT = 'https://api.example.com'",
        "@RestController\n@RequestMapping('/api/v1')",
    ];
    
    for (i, code) in code_samples.iter().enumerate() {
        println!("Testing code sample {}: {}", i + 1, code);
        
        let result = embedder.embed(code).await;
        match result {
            Ok(embedding) => {
                // Verify embedding properties
                assert_eq!(embedding.len(), 768, "Embedding dimension mismatch for code sample {}", i + 1);
                
                // Check normalization (should be close to 1.0 for L2 normalized vectors)
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 0.01, "Embedding {} not properly normalized: norm = {}", i + 1, norm);
                
                // Ensure no NaN or infinite values
                assert!(embedding.iter().all(|&x| x.is_finite()), "Embedding {} contains invalid values", i + 1);
                
                // Check that we're getting meaningful embeddings (not all zeros)
                let sum_abs: f32 = embedding.iter().map(|x| x.abs()).sum();
                assert!(sum_abs > 0.1, "Embedding {} appears to be zero or near-zero", i + 1);
                
                println!("  ✅ Sample {}: {} dims, norm={:.4}, sum_abs={:.4}", i + 1, embedding.len(), norm, sum_abs);
            },
            Err(e) => panic!("Failed to generate embedding for code sample {}: {}", i + 1, e),
        }
    }
}

/// Test that different code samples produce different embeddings
#[tokio::test]
async fn test_embedding_uniqueness() {
    let embedder = LazyEmbedder::new();
    
    let code1 = "def authenticate_user(username, password):";
    let code2 = "function calculateTotal(items) {";
    
    let embedding1 = embedder.embed(code1).await.expect("Failed to embed code1");
    let embedding2 = embedder.embed(code2).await.expect("Failed to embed code2");
    
    // Calculate cosine similarity
    let dot_product: f32 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
    
    // Since both are L2 normalized, dot product is cosine similarity
    assert!(dot_product < 0.95, "Embeddings too similar (cosine similarity: {:.4}). Different code should produce different embeddings.", dot_product);
    
    println!("✅ Embeddings are appropriately different (cosine similarity: {:.4})", dot_product);
}

/// Test caching behavior with code embeddings
#[tokio::test]
async fn test_embedding_cache_behavior() {
    let embedder = LazyEmbedder::new();
    
    let code = "class UserRepository extends BaseRepository {";
    
    // First embedding (should be computed)
    let start = std::time::Instant::now();
    let embedding1 = embedder.embed(code).await.expect("First embedding failed");
    let first_duration = start.elapsed();
    
    // Second embedding (should be cached)
    let start = std::time::Instant::now();
    let embedding2 = embedder.embed(code).await.expect("Second embedding failed");
    let second_duration = start.elapsed();
    
    // Verify embeddings are identical
    assert_eq!(embedding1.len(), embedding2.len(), "Cached embedding dimension mismatch");
    
    for (i, (a, b)) in embedding1.iter().zip(embedding2.iter()).enumerate() {
        assert_eq!(*a, *b, "Cached embedding mismatch at index {}", i);
    }
    
    // Cache should be significantly faster (unless model loading dominates)
    println!("✅ First: {:?}, Second: {:?}, Speedup: {:.2}x", first_duration, second_duration, 
             first_duration.as_nanos() as f64 / second_duration.as_nanos() as f64);
}

/// Test error handling for invalid inputs
#[tokio::test]
async fn test_error_handling() {
    let embedder = LazyEmbedder::new();
    
    // Test empty string
    let result = embedder.embed("").await;
    match result {
        Ok(embedding) => {
            // Empty string might produce a valid embedding (depends on tokenizer)
            assert_eq!(embedding.len(), 768, "Empty string embedding should still have correct dimensions");
            println!("✅ Empty string handled gracefully");
        },
        Err(e) => {
            println!("✅ Empty string rejected with error: {}", e);
        }
    }
    
    // Test extremely long string
    let very_long_code = "a".repeat(10000);
    let result = embedder.embed(&very_long_code).await;
    match result {
        Ok(embedding) => {
            assert_eq!(embedding.len(), 768, "Long string embedding should be truncated but valid");
            println!("✅ Long string handled gracefully (likely truncated)");
        },
        Err(e) => {
            println!("✅ Long string rejected with error: {}", e);
        }
    }
}

/// Test batch embedding functionality
#[tokio::test]
async fn test_batch_embedding() {
    let embedder = LazyEmbedder::new();
    
    let code_batch = vec![
        "import torch".to_string(),
        "from transformers import AutoModel".to_string(), 
        "def forward(self, x):".to_string(),
        "return self.linear(x)".to_string(),
    ];
    
    let result = embedder.embed_batch(&code_batch).await;
    match result {
        Ok(embeddings) => {
            assert_eq!(embeddings.len(), 4, "Batch should produce 4 embeddings");
            
            for (i, embedding) in embeddings.iter().enumerate() {
                assert_eq!(embedding.len(), 768, "Batch embedding {} has wrong dimensions", i);
                
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 0.01, "Batch embedding {} not normalized: {}", i, norm);
            }
            
            println!("✅ Batch embedding successful: {} embeddings generated", embeddings.len());
        },
        Err(e) => panic!("Batch embedding failed: {}", e),
    }
}

/// Test memory pressure handling
#[tokio::test]
async fn test_memory_pressure_handling() {
    // This test checks if the system handles memory pressure gracefully
    let embedder = LazyEmbedder::new();
    
    // Generate many embeddings in sequence to test memory management
    let mut embeddings = Vec::new();
    
    for i in 0..50 {
        let code = format!("function test{}() {{ return {}; }}", i, i);
        match embedder.embed(&code).await {
            Ok(embedding) => {
                assert_eq!(embedding.len(), 768, "Embedding {} has wrong dimensions", i);
                embeddings.push(embedding);
                
                if i % 10 == 0 {
                    println!("  Generated {} embeddings...", i + 1);
                }
            },
            Err(e) => {
                println!("⚠️  Memory pressure at embedding {}: {}", i, e);
                break;
            }
        }
    }
    
    assert!(embeddings.len() >= 20, "Should be able to generate at least 20 embeddings");
    println!("✅ Generated {} embeddings under memory pressure", embeddings.len());
}

/// Test attention mask validation (this was a source of the original issues)
#[tokio::test]
async fn test_attention_mask_validation() {
    // Test the specific attention mask validation that was causing issues
    let test_cases = vec![
        (vec![1, 1, 1, 0, 0], 5, true),  // Valid mask
        (vec![1, 1, 1], 5, false),       // Wrong length
        (vec![0, 0, 0, 0, 0], 5, false), // All zeros
        (vec![1], 1, true),              // Single valid token
        (vec![0], 1, false),             // Single invalid token
    ];
    
    for (mask, expected_len, should_pass) in test_cases {
        let result = NomicEmbedder::validate_attention_mask(&mask, expected_len);
        
        if should_pass {
            assert!(result.is_ok(), "Expected mask {:?} to pass validation", mask);
            println!("✅ Mask {:?} passed validation", mask);
        } else {
            assert!(result.is_err(), "Expected mask {:?} to fail validation", mask);
            println!("✅ Mask {:?} correctly rejected: {}", mask, result.unwrap_err());
        }
    }
}

/// Test GGUF model specific functionality
#[tokio::test]
async fn test_gguf_model_specifics() {
    let embedder = LazyEmbedder::new();
    
    // Force initialization to get access to internal structure
    let embedder_arc = embedder.get_or_init().await.expect("Failed to initialize");
    
    // Test that we can handle Q4_K_M quantization properly
    let technical_code = "void* malloc(size_t size) { return NULL; }";
    let embedding = embedder.embed(technical_code).await.expect("Technical code embedding failed");
    
    // Verify the embedding contains reasonable values for quantized model
    let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
    let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
    
    println!("GGUF embedding stats - mean: {:.6}, variance: {:.6}", mean, variance);
    
    // For a properly functioning quantized model, we expect:
    assert!(mean.abs() < 0.1, "Mean too large for normalized embedding: {}", mean);
    assert!(variance > 0.01, "Variance too small, might indicate quantization issues: {}", variance);
    assert!(variance < 1.0, "Variance too large for normalized embedding: {}", variance);
    
    println!("✅ GGUF Q4_K_M quantization working properly");
}

/// Integration test that would have caught the original misconfiguration
#[tokio::test]
async fn test_configuration_integration() {
    // This test specifically targets the types of misconfigurations that caused issues
    
    // Test 1: Model path configuration
    let model_path = std::env::current_dir().unwrap().join("model/nomic-embed-code.Q4_K_M.gguf");
    assert!(model_path.exists(), "Model not found at expected path: {:?}", model_path);
    
    // Test 2: Feature flag consistency
    #[cfg(not(feature = "ml"))] 
    {
        compile_error!("These tests require the 'ml' feature to be enabled");
    }
    
    // Test 3: Tokenizer and model compatibility
    let embedder = LazyEmbedder::new();
    let test_code = "fn main() { println!(\"Hello, world!\"); }";
    
    let embedding = embedder.embed(test_code).await.expect("Basic Rust code should embed successfully");
    assert_eq!(embedding.len(), 768, "Embedding dimension indicates model/tokenizer mismatch");
    
    println!("✅ Configuration integration test passed");
}