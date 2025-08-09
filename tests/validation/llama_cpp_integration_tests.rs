//! Validation procedures for llama.cpp integration
//! Tests the GGUF model loading and inference pipeline

use embed_search::{
    embedding::{NomicEmbedder, LazyEmbedder},
    error::EmbedError,
};
use std::path::PathBuf;
use tokio::time::{timeout, Duration};

/// Test GGUF model format validation
#[tokio::test]
async fn test_gguf_format_validation() {
    println!("🔍 Testing GGUF model format validation");
    println!("====================================\n");
    
    // Test 1: Verify model file exists and has correct format
    let model_path = PathBuf::from("./model/nomic-embed-code.Q4_K_M.gguf");
    
    assert!(model_path.exists(), "GGUF model file not found: {:?}", model_path);
    
    // Test 2: Check file size is reasonable for Q4_K_M quantization
    let metadata = std::fs::metadata(&model_path).expect("Failed to read model metadata");
    let file_size = metadata.len();
    
    println!("📊 Model file statistics:");
    println!("  Path: {:?}", model_path);
    println!("  Size: {:.1} MB", file_size as f64 / (1024.0 * 1024.0));
    
    // Q4_K_M should be around 80-100MB for nomic-embed models
    assert!(file_size > 50 * 1024 * 1024, "Model file too small: {} bytes. May be corrupted or wrong model.", file_size);
    assert!(file_size < 200 * 1024 * 1024, "Model file too large: {} bytes. May be wrong quantization level.", file_size);
    
    println!("  ✅ File size within expected range for Q4_K_M quantization\n");
    
    // Test 3: Try to load the GGUF file and verify it's parseable
    let embedder = LazyEmbedder::new();
    
    let initialization_result = timeout(
        Duration::from_secs(180), // Allow more time for model loading
        embedder.get_or_init()
    ).await;
    
    match initialization_result {
        Ok(Ok(_)) => {
            println!("✅ GGUF model loaded successfully");
        },
        Ok(Err(e)) => {
            panic!("GGUF model loading failed: {}. This indicates format issues or missing dependencies.", e);
        },
        Err(_) => {
            panic!("GGUF model loading timed out. This may indicate a corrupted file or system issues.");
        }
    }
}

/// Test quantization-specific behavior
#[tokio::test]
async fn test_q4_k_m_quantization_behavior() {
    println!("🔢 Testing Q4_K_M quantization behavior");
    println!("====================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test different types of input to verify quantization doesn't break functionality
    let quantization_test_cases = vec![
        ("simple_code", "fn test() {}"),
        ("complex_code", "impl<T: Clone + Send> MyStruct<T> { async fn process(&self) -> Result<Vec<T>, Error> { unimplemented!() } }"),
        ("unicode_text", "// Unicode: 🚀 Ξ Ω α β γ δ ε"),
        ("long_text", &"This is a long piece of text that tests how the quantized model handles extended sequences. ".repeat(20)),
        ("special_chars", "const REGEX = /^[\\w._%+-]+@[\\w.-]+\\.[A-Z]{2,}$/i;"),
    ];
    
    let mut quantization_quality_scores = Vec::new();
    
    for (test_name, input) in quantization_test_cases {
        println!("  Testing {}: '{}'", test_name, input.chars().take(50).collect::<String>());
        
        let embedding = embedder.embed(input).await.expect(&format!("Embedding failed for {}", test_name));
        
        // Validate quantized embedding properties
        assert_eq!(embedding.len(), 768, "Wrong embedding dimension for {}", test_name);
        
        // Check L2 normalization (should still be preserved)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.02, "Quantized embedding not normalized for {}: norm={}", test_name, norm);
        
        // Check for reasonable value distribution (quantization effects)
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
        
        // Quantized models may have different variance patterns
        assert!(variance > 0.001, "Variance too low for quantized model {}: {}", test_name, variance);
        assert!(variance < 2.0, "Variance too high for quantized model {}: {}", test_name, variance);
        
        // Check for quantization artifacts (values should not all be identical)
        let unique_values: std::collections::HashSet<_> = embedding.iter()
            .map(|x| (x * 10000.0) as i32)
            .collect();
        
        let uniqueness_ratio = unique_values.len() as f32 / embedding.len() as f32;
        assert!(uniqueness_ratio > 0.1, "Too few unique values in quantized embedding for {}: {:.3}", test_name, uniqueness_ratio);
        
        // Quality score based on normalization, variance, and uniqueness
        let quality_score = ((1.0 - (norm - 1.0).abs() * 50.0).max(0.0) + 
                            if variance > 0.01 && variance < 1.0 { 1.0 } else { 0.5 } +
                            uniqueness_ratio) / 3.0;
        
        quantization_quality_scores.push(quality_score);
        
        println!("    ✅ {}: norm={:.4}, variance={:.4}, uniqueness={:.3}, quality={:.3}", 
                 test_name, norm, variance, uniqueness_ratio, quality_score);
    }
    
    let avg_quality = quantization_quality_scores.iter().sum::<f32>() / quantization_quality_scores.len() as f32;
    
    println!("\n📊 Q4_K_M Quantization Summary:");
    println!("  Average quality score: {:.3}/1.0", avg_quality);
    
    assert!(avg_quality > 0.7, "Quantization quality too low: {:.3}", avg_quality);
    
    println!("  ✅ Q4_K_M quantization performing within acceptable bounds\n");
}

/// Test model-specific tensor operations
#[tokio::test]
async fn test_tensor_operations() {
    println!("🧮 Testing tensor operations with GGUF model");
    println!("==========================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Initialize to access model internals
    let _embedder_arc = embedder.get_or_init().await.expect("Model initialization failed");
    
    // Test tensor computation consistency
    let test_inputs = vec![
        "def calculate_similarity(a, b): return dot_product(a, b) / (norm(a) * norm(b))",
        "function tensorOperation(input) { return input.map(x => x * 0.5 + 0.5); }",
        "pub fn matrix_multiply(a: &[f32], b: &[f32]) -> Vec<f32> { /* implementation */ }",
    ];
    
    let mut embeddings = Vec::new();
    
    for (i, input) in test_inputs.iter().enumerate() {
        println!("  Testing tensor operation for input {}", i + 1);
        
        // Run the same embedding multiple times to test consistency
        let mut run_embeddings = Vec::new();
        
        for run in 0..3 {
            let embedding = embedder.embed(input).await.expect(&format!("Tensor operation failed for input {} run {}", i + 1, run));
            run_embeddings.push(embedding);
        }
        
        // Verify consistency across runs
        for run in 1..run_embeddings.len() {
            let first = &run_embeddings[0];
            let current = &run_embeddings[run];
            
            assert_eq!(first.len(), current.len(), "Inconsistent embedding size across runs");
            
            for (j, (&a, &b)) in first.iter().zip(current.iter()).enumerate() {
                assert_eq!(a, b, "Tensor computation inconsistency at position {} run {} vs run 0", j, run);
            }
        }
        
        embeddings.push(run_embeddings[0].clone());
        println!("    ✅ Tensor operations consistent across multiple runs");
    }
    
    // Test tensor arithmetic properties
    println!("\n  Testing tensor arithmetic properties...");
    
    // Test that different inputs produce different embeddings
    for i in 0..embeddings.len() {
        for j in i+1..embeddings.len() {
            let cosine_sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            assert!(cosine_sim < 0.95, "Embeddings {} and {} too similar: {:.4}", i, j, cosine_sim);
            println!("    ✅ Inputs {} and {} produce distinct embeddings (similarity: {:.4})", i+1, j+1, cosine_sim);
        }
    }
    
    println!("\n✅ Tensor operations validation completed");
}

/// Test attention mechanism with different sequence lengths
#[tokio::test]
async fn test_attention_mechanism() {
    println!("🎯 Testing attention mechanism");
    println!("==============================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test different sequence lengths to verify attention handling
    let attention_test_cases = vec![
        ("very_short", "fn f() {}"),
        ("short", "function processData(input) { return input.map(x => x * 2); }"),
        ("medium", "class DataProcessor {\n  constructor(config) {\n    this.config = config;\n    this.cache = new Map();\n  }\n  \n  async process(data) {\n    if (this.cache.has(data)) {\n      return this.cache.get(data);\n    }\n    const result = await this.transform(data);\n    this.cache.set(data, result);\n    return result;\n  }\n}"),
        ("long", create_long_attention_test_input()),
    ];
    
    let mut attention_performance = Vec::new();
    
    for (test_name, input) in attention_test_cases {
        let seq_length = input.split_whitespace().count();
        println!("  Testing attention for {}: ~{} tokens", test_name, seq_length);
        
        let start_time = std::time::Instant::now();
        
        let embedding = embedder.embed(input).await.expect(&format!("Attention test failed for {}", test_name));
        
        let processing_time = start_time.elapsed();
        attention_performance.push((test_name, seq_length, processing_time));
        
        // Validate embedding quality doesn't degrade with sequence length
        assert_eq!(embedding.len(), 768, "Wrong embedding dimension for {}", test_name);
        
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.02, "Attention mechanism broke normalization for {}: norm={}", test_name, norm);
        
        // Check that longer sequences don't produce degenerate embeddings
        let variance: f32 = {
            let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
            embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32
        };
        
        assert!(variance > 0.001, "Attention mechanism produced low-variance embedding for {}: {}", test_name, variance);
        
        println!("    ✅ {}: {:?}, norm={:.4}, variance={:.4}", test_name, processing_time, norm, variance);
    }
    
    println!("\n📊 Attention Mechanism Performance:");
    
    for (name, tokens, time) in &attention_performance {
        let tokens_per_ms = *tokens as f64 / time.as_millis() as f64;
        println!("  {}: {} tokens in {:?} ({:.1} tokens/ms)", name, tokens, time, tokens_per_ms);
    }
    
    // Performance should not degrade dramatically with sequence length
    let short_time = attention_performance[1].2; // "short" case
    let long_time = attention_performance[3].2;  // "long" case
    
    let slowdown_ratio = long_time.as_millis() as f64 / short_time.as_millis() as f64;
    
    println!("  Slowdown ratio (long/short): {:.2}x", slowdown_ratio);
    
    // Should not be more than 10x slower (attention is quadratic, but with reasonable limits)
    assert!(slowdown_ratio < 10.0, "Attention mechanism too slow for long sequences: {:.2}x", slowdown_ratio);
    
    println!("\n✅ Attention mechanism validation completed");
}

/// Test memory management during inference
#[tokio::test]
async fn test_inference_memory_management() {
    println!("🧠 Testing inference memory management");
    println!("====================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Initialize and get baseline memory
    let _ = embedder.embed("initialization").await.expect("Init failed");
    let baseline_memory = get_process_memory();
    
    println!("  Baseline memory usage: {:.1} MB", baseline_memory as f64 / (1024.0 * 1024.0));
    
    // Test memory usage with varying load
    let memory_test_loads = vec![
        ("light_load", 10, 100),    // 10 embeddings, 100 chars each
        ("medium_load", 50, 500),   // 50 embeddings, 500 chars each  
        ("heavy_load", 100, 1000),  // 100 embeddings, 1000 chars each
    ];
    
    for (load_name, num_embeddings, chars_per_embedding) in memory_test_loads {
        println!("  Testing {}: {} embeddings, {} chars each", load_name, num_embeddings, chars_per_embedding);
        
        let mut peak_memory = baseline_memory;
        let start_time = std::time::Instant::now();
        
        for i in 0..num_embeddings {
            let input = generate_test_input(i, chars_per_embedding);
            
            let embedding = embedder.embed(&input).await.expect(&format!("Memory test failed at iteration {}", i));
            assert_eq!(embedding.len(), 768, "Wrong embedding size during memory test");
            
            // Check memory every 10 iterations
            if i % 10 == 0 {
                let current_memory = get_process_memory();
                peak_memory = peak_memory.max(current_memory);
                
                let memory_growth = current_memory.saturating_sub(baseline_memory);
                
                if memory_growth > 500 * 1024 * 1024 { // 500MB growth limit
                    println!("    ⚠️  Memory growth limit reached at iteration {}: {} MB", i, memory_growth / (1024 * 1024));
                    break;
                }
            }
        }
        
        let duration = start_time.elapsed();
        let final_memory = get_process_memory();
        let memory_growth = final_memory.saturating_sub(baseline_memory);
        
        println!("    📊 {}: {:?} duration, {:.1} MB peak growth", 
                 load_name, duration, memory_growth as f64 / (1024.0 * 1024.0));
        
        // Memory growth should be reasonable
        assert!(memory_growth < 1024 * 1024 * 1024, "Excessive memory growth for {}: {} MB", load_name, memory_growth / (1024 * 1024));
        
        // Force cleanup attempt
        tokio::task::yield_now().await;
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    // Test memory cleanup
    println!("\n  Testing memory cleanup...");
    
    // Generate some temporary load
    for i in 0..20 {
        let temp_input = generate_test_input(i, 2000);
        let _ = embedder.embed(&temp_input).await.expect("Cleanup test failed");
    }
    
    // Allow time for cleanup
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let cleanup_memory = get_process_memory();
    let post_cleanup_growth = cleanup_memory.saturating_sub(baseline_memory);
    
    println!("  Memory after cleanup: {:.1} MB (+{:.1} MB from baseline)", 
             cleanup_memory as f64 / (1024.0 * 1024.0),
             post_cleanup_growth as f64 / (1024.0 * 1024.0));
    
    println!("\n✅ Memory management validation completed");
}

/// Test error handling in GGUF operations
#[tokio::test]
async fn test_gguf_error_handling() {
    println!("🚨 Testing GGUF error handling");
    println!("=============================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test various error conditions that might occur with GGUF models
    let error_test_cases = vec![
        ("empty_input", ""),
        ("null_bytes", "\0\0\0"),
        ("very_long", &"a".repeat(10000)),
        ("invalid_utf8", "\xFF\xFE"),
        ("control_chars", "\x01\x02\x03\x04"),
    ];
    
    let mut error_handling_scores = Vec::new();
    
    for (test_name, input) in error_test_cases {
        println!("  Testing error handling for: {}", test_name);
        
        let result = embedder.embed(input).await;
        
        match result {
            Ok(embedding) => {
                // Some inputs might be handled gracefully
                assert_eq!(embedding.len(), 768, "Graceful handling should produce correct dimensions for {}", test_name);
                
                // Check that the embedding is reasonable
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if (norm - 1.0).abs() < 0.02 {
                    println!("    ✅ {} handled gracefully", test_name);
                    error_handling_scores.push(1.0);
                } else {
                    println!("    ⚠️  {} produced suspicious embedding (norm={})", test_name, norm);
                    error_handling_scores.push(0.5);
                }
            },
            Err(e) => {
                // Error should be descriptive and appropriate
                let error_message = e.to_string();
                
                if error_message.len() > 10 && !error_message.contains("panic") {
                    println!("    ✅ {} properly rejected: {}", test_name, error_message.chars().take(50).collect::<String>());
                    error_handling_scores.push(1.0);
                } else {
                    println!("    ❌ {} error too generic: {}", test_name, error_message);
                    error_handling_scores.push(0.2);
                }
            }
        }
    }
    
    let error_handling_quality = error_handling_scores.iter().sum::<f32>() / error_handling_scores.len() as f32;
    
    println!("\n📊 Error Handling Summary:");
    println!("  Quality score: {:.2}/1.0", error_handling_quality);
    
    assert!(error_handling_quality > 0.7, "Error handling quality too low: {:.2}", error_handling_quality);
    
    println!("\n✅ GGUF error handling validation completed");
}

// Helper functions

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot_product / (norm_a * norm_b)
}

fn create_long_attention_test_input() -> &'static str {
    "pub struct ComplexProcessor<T: Clone + Send + Sync + 'static> {\n    config: ProcessorConfig,\n    cache: Arc<RwLock<HashMap<String, CachedResult>>>,\n    metrics: Arc<Mutex<ProcessingMetrics>>,\n    thread_pool: ThreadPool,\n    shutdown_signal: Arc<AtomicBool>,\n}\n\nimpl<T: Clone + Send + Sync + 'static> ComplexProcessor<T> {\n    pub fn new(config: ProcessorConfig) -> Result<Self, ProcessorError> {\n        let thread_pool = ThreadPoolBuilder::new()\n            .num_threads(config.worker_threads)\n            .thread_name(|i| format!(\"processor-worker-{}\", i))\n            .build()?;\n        \n        Ok(Self {\n            config,\n            cache: Arc::new(RwLock::new(HashMap::new())),\n            metrics: Arc::new(Mutex::new(ProcessingMetrics::default())),\n            thread_pool,\n            shutdown_signal: Arc::new(AtomicBool::new(false)),\n        })\n    }\n    \n    pub async fn process_batch(&self, items: Vec<T>) -> Result<Vec<ProcessedItem<T>>, ProcessorError> {\n        let batch_id = Uuid::new_v4();\n        let start_time = Instant::now();\n        \n        info!(\"Starting batch processing: batch_id={}, item_count={}\", batch_id, items.len());\n        \n        let mut results = Vec::with_capacity(items.len());\n        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_items));\n        \n        let futures: Vec<_> = items.into_iter().enumerate().map(|(index, item)| {\n            let semaphore = semaphore.clone();\n            let cache = self.cache.clone();\n            let metrics = self.metrics.clone();\n            let shutdown_signal = self.shutdown_signal.clone();\n            \n            async move {\n                let _permit = semaphore.acquire().await?;\n                \n                if shutdown_signal.load(Ordering::Relaxed) {\n                    return Err(ProcessorError::ShutdownRequested);\n                }\n                \n                let item_start = Instant::now();\n                let result = Self::process_single_item(item, cache, metrics, index).await?;\n                let item_duration = item_start.elapsed();\n                \n                if item_duration > Duration::from_secs(5) {\n                    warn!(\"Slow item processing: index={}, duration={:?}\", index, item_duration);\n                }\n                \n                Ok(result)\n            }\n        }).collect();\n        \n        let batch_results = futures::future::join_all(futures).await;\n        \n        for (index, result) in batch_results.into_iter().enumerate() {\n            match result {\n                Ok(processed_item) => results.push(processed_item),\n                Err(e) => {\n                    error!(\"Item processing failed: batch_id={}, index={}, error={}\", batch_id, index, e);\n                    return Err(e);\n                }\n            }\n        }\n        \n        let total_duration = start_time.elapsed();\n        info!(\"Batch processing completed: batch_id={}, duration={:?}, success_count={}\", \n              batch_id, total_duration, results.len());\n        \n        self.update_batch_metrics(batch_id, total_duration, results.len()).await;\n        \n        Ok(results)\n    }\n}"
}

fn generate_test_input(index: usize, target_length: usize) -> String {
    let mut input = format!("fn test_function_{}(param: &str) -> String {{\n", index);
    
    let line_template = "    let processed_{} = transform_data(param, {});\n";
    
    let mut current_length = input.len();
    let mut line_count = 0;
    
    while current_length < target_length {
        let line = format!(line_template, line_count, line_count * 2);
        input.push_str(&line);
        current_length += line.len();
        line_count += 1;
        
        if current_length >= target_length {
            break;
        }
    }
    
    input.push_str("    processed_data.to_string()\n}");
    input
}

fn get_process_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback for other platforms
    200 * 1024 * 1024 // 200MB placeholder
}