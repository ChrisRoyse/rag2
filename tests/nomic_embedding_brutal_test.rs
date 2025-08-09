/// BRUTAL NOMIC EMBEDDING TEST - SIMPLIFIED AND FOCUSED
/// 
/// INTJ Type-8 testing: No mercy, absolute truth required
/// This test validates ONLY the core embedding functionality

use std::time::{Duration, Instant};
use anyhow::{Result, anyhow};

// Basic test without complex dependencies
#[tokio::test]
#[cfg(feature = "ml")]
async fn brutal_embedding_core_test() -> Result<()> {
    use embed_search::embedding::nomic::NomicEmbedder;
    
    println!("🔥 BRUTAL TEST: NOMIC EMBEDDING CORE FUNCTIONALITY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Test 1: Model Loading
    println!("📦 TEST 1: Model Loading (CPU ONLY)");
    let load_start = Instant::now();
    
    let embedder = match NomicEmbedder::get_global().await {
        Ok(e) => {
            let load_time = load_start.elapsed();
            println!("✅ Model loaded in {:?}", load_time);
            if load_time > Duration::from_secs(120) {
                return Err(anyhow!("❌ BRUTAL FAILURE: Model loading too slow: {:?}", load_time));
            }
            e
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Model loading failed: {}", e));
        }
    };
    
    // Test 2: Basic Embedding Generation
    println!("🧠 TEST 2: Basic Embedding Generation");
    let test_text = "fn main() { println!(\"Hello, world!\"); }";
    let embed_start = Instant::now();
    
    let embedding = match embedder.embed(test_text) {
        Ok(emb) => {
            let embed_time = embed_start.elapsed();
            println!("✅ Embedding generated in {:?}", embed_time);
            println!("📏 Dimensions: {}", emb.len());
            
            if emb.len() != 768 {
                return Err(anyhow!("❌ BRUTAL FAILURE: Wrong dimensions: got {}, expected 768", emb.len()));
            }
            
            if embed_time > Duration::from_secs(10) {
                return Err(anyhow!("❌ BRUTAL FAILURE: Embedding too slow: {:?}", embed_time));
            }
            
            emb
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Embedding generation failed: {}", e));
        }
    };
    
    // Test 3: Embedding Quality
    println!("🔍 TEST 3: Embedding Quality");
    
    // Check for NaN/Inf
    let finite_count = embedding.iter().filter(|&&x| x.is_finite()).count();
    if finite_count != embedding.len() {
        return Err(anyhow!("❌ BRUTAL FAILURE: {} invalid values in embedding", embedding.len() - finite_count));
    }
    println!("✅ All {} values are finite", finite_count);
    
    // Check normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 0.01 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Embedding not normalized: norm = {}", norm));
    }
    println!("✅ Embedding normalized: norm = {:.6}", norm);
    
    // Test 4: Performance Measurement
    println!("⚡ TEST 4: Performance Measurement");
    
    let test_texts = [
        "short",
        "function calculate(a, b) { return a + b; }",
        "class DataProcessor { constructor() { this.data = []; } process(input) { return input.map(x => x * 2); } }",
    ];
    
    let mut total_chars = 0;
    let mut total_time = Duration::default();
    
    for (i, text) in test_texts.iter().enumerate() {
        let chars = text.len();
        total_chars += chars;
        
        let start = Instant::now();
        match embedder.embed(text) {
            Ok(emb) => {
                let duration = start.elapsed();
                total_time += duration;
                
                println!("  Text {}: {} chars, {:?} ({:.0} chars/sec)", 
                        i + 1, chars, duration, chars as f64 / duration.as_secs_f64());
                
                if emb.len() != 768 {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Wrong dimensions for text {}", i + 1));
                }
                
                if duration > Duration::from_secs(5) {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Text {} took too long: {:?}", i + 1, duration));
                }
            }
            Err(e) => {
                return Err(anyhow!("❌ BRUTAL FAILURE: Text {} failed: {}", i + 1, e));
            }
        }
    }
    
    let avg_chars_per_sec = total_chars as f64 / total_time.as_secs_f64();
    println!("📊 Average performance: {:.0} characters/second", avg_chars_per_sec);
    
    if avg_chars_per_sec < 100.0 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Performance too slow: {:.0} chars/sec", avg_chars_per_sec));
    }
    
    // Test 5: Similarity Test
    println!("📐 TEST 5: Similarity Test");
    
    let text1 = "function add(a, b) { return a + b; }";
    let text2 = "function sum(x, y) { return x + y; }";
    let text3 = "SELECT * FROM users WHERE age > 25";
    
    let emb1 = embedder.embed(text1)?;
    let emb2 = embedder.embed(text2)?;
    let emb3 = embedder.embed(text3)?;
    
    // Calculate cosine similarity
    let dot12: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let dot13: f32 = emb1.iter().zip(emb3.iter()).map(|(a, b)| a * b).sum();
    
    println!("  Similar functions similarity: {:.3}", dot12);
    println!("  Different content similarity: {:.3}", dot13);
    
    if dot12 <= dot13 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Similar functions should have higher similarity"));
    }
    
    if dot12 < 0.5 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Similar functions similarity too low: {:.3}", dot12));
    }
    
    // Test 6: Large Text Handling
    println!("📄 TEST 6: Large Text Handling");
    
    let large_text = "class ComplexDataProcessor {\n".repeat(100) + 
                    "  constructor() { this.data = new Map(); }\n" +
                    "  process(input) { return input.filter(x => x > 0).map(x => x * 2); }\n" +
                    "}";
    
    let large_start = Instant::now();
    match embedder.embed(&large_text) {
        Ok(emb) => {
            let large_time = large_start.elapsed();
            println!("✅ Large text ({} chars) processed in {:?}", large_text.len(), large_time);
            
            if emb.len() != 768 {
                return Err(anyhow!("❌ BRUTAL FAILURE: Large text wrong dimensions"));
            }
            
            if large_time > Duration::from_secs(15) {
                return Err(anyhow!("❌ BRUTAL FAILURE: Large text too slow: {:?}", large_time));
            }
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Large text processing failed: {}", e));
        }
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🏆 BRUTAL VERDICT: ALL TESTS PASSED");
    println!("   NOMIC EMBEDDING MODEL IS FUNCTIONAL ON CPU");
    println!("   ✅ Model loads successfully");
    println!("   ✅ Generates 768-dimensional embeddings");
    println!("   ✅ Produces normalized, finite values");
    println!("   ✅ Performs at acceptable speed");
    println!("   ✅ Shows semantic understanding");
    println!("   ✅ Handles large text inputs");
    
    Ok(())
}

#[tokio::test] 
#[cfg(not(feature = "ml"))]
async fn brutal_ml_feature_disabled() -> Result<()> {
    println!("⚠️  ML feature is disabled - skipping embedding tests");
    println!("   To run embedding tests: cargo test --features ml");
    Ok(())
}

// Memory measurement (cross-platform)
fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert kB to MB
                        }
                    }
                }
            }
        }
    }
    
    // Fallback estimate
    0.0
}