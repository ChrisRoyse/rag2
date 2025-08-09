//! BRUTAL TRUTH INTEGRATION TEST
//! INTJ Type-8 Analysis - Find every failure in the ACTUAL system
//! 
//! Tests only components that EXIST, not imaginary ones.

use std::time::{Duration, Instant};
use std::path::PathBuf;
use tempfile::TempDir;
use anyhow::Result;

// Import ACTUAL working components
use embed_search::search::bm25::{BM25Engine, BM25Document, Token};
use embed_search::cache::bounded_cache::BoundedCache;
use embed_search::storage::safe_vectordb::{VectorStorage, StorageConfig};
use embed_search::config::Config;

/// BRUTAL PERFORMANCE REQUIREMENTS
const BRUTAL_LIMITS: BrutalLimits = BrutalLimits {
    max_search_time_ms: 50,        // 50ms per search
    min_ops_per_second: 500,       // 500 operations/sec minimum
    max_memory_mb: 128,            // 128MB memory limit
    min_accuracy_percent: 70.0,    // 70% minimum accuracy
};

struct BrutalLimits {
    max_search_time_ms: u64,
    min_ops_per_second: u64,
    max_memory_mb: u64,
    min_accuracy_percent: f64,
}

/// BRUTAL TEST REPORT - NO LIES
#[derive(Debug)]
struct BrutalTestResult {
    test_name: String,
    passed: bool,
    error: Option<String>,
    performance_ms: u64,
    memory_mb: f64,
    operations_per_sec: f64,
    accuracy_percent: f64,
    details: Vec<String>,
}

impl BrutalTestResult {
    fn new(name: &str) -> Self {
        Self {
            test_name: name.to_string(),
            passed: false,
            error: None,
            performance_ms: 0,
            memory_mb: 0.0,
            operations_per_sec: 0.0,
            accuracy_percent: 0.0,
            details: Vec::new(),
        }
    }

    fn fail(&mut self, reason: &str) {
        self.passed = false;
        self.error = Some(reason.to_string());
    }

    fn pass(&mut self) {
        self.passed = true;
    }

    fn add_detail(&mut self, detail: &str) {
        self.details.push(detail.to_string());
    }
}

fn get_memory_usage() -> f64 {
    // Simple memory measurement
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert to MB
                        }
                    }
                }
            }
        }
    }
    
    // Fallback for other systems
    0.0
}

/// TEST 1: BM25 CORE FUNCTIONALITY - BRUTAL TRUTH
fn test_bm25_core() -> BrutalTestResult {
    let mut result = BrutalTestResult::new("BM25 Core Functionality");
    result.memory_mb = get_memory_usage();

    let start_time = Instant::now();

    // Create BM25 engine
    let mut engine = BM25Engine::new();

    // Add test documents with realistic Rust code
    let test_docs = vec![
        ("main.rs", "fn main() { println!(\"Hello, world!\"); }"),
        ("lib.rs", "pub mod config; pub use config::Config;"),
        ("config.rs", "pub struct Config { pub port: u16, pub host: String }"),
        ("server.rs", "use tokio::net::TcpListener; async fn start_server() {}"),
        ("client.rs", "use reqwest::Client; async fn make_request() -> Result<String, Error> { Ok(String::new()) }"),
        ("utils.rs", "use std::collections::HashMap; pub fn process_data(map: HashMap<String, i32>) {}"),
        ("error.rs", "use thiserror::Error; #[derive(Error, Debug)] pub enum AppError { #[error(\"IO error\")] Io }"),
        ("async.rs", "use futures::future::join_all; async fn concurrent_operations() { join_all(vec![]).await; }"),
        ("traits.rs", "pub trait Processor { fn process(&self, data: &str) -> String; } impl Processor for String {}"),
        ("tests.rs", "#[cfg(test)] mod tests { use super::*; #[test] fn test_basic() { assert_eq!(2 + 2, 4); } }"),
    ];

    // Add documents to index
    for (filename, content) in &test_docs {
        let tokens: Vec<Token> = content
            .split_whitespace()
            .enumerate()
            .map(|(pos, word)| Token {
                text: word.to_string(),
                position: pos,
                importance_weight: 1.0,
            })
            .collect();

        let doc = BM25Document {
            id: filename.to_string(),
            file_path: filename.to_string(),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: 1,
            language: Some("rust".to_string()),
        };

        if let Err(e) = engine.add_document(doc) {
            result.fail(&format!("Failed to add document {}: {}", filename, e));
            return result;
        }
    }

    // Test queries with expected results
    let test_queries = vec![
        ("fn main", vec!["main.rs"]),
        ("struct Config", vec!["config.rs"]),
        ("async fn", vec!["server.rs", "client.rs", "async.rs"]),
        ("use std", vec!["utils.rs"]),
        ("tokio", vec!["server.rs"]),
        ("Error", vec!["client.rs", "error.rs"]),
        ("test", vec!["tests.rs"]),
        ("pub", vec!["lib.rs", "config.rs", "utils.rs", "traits.rs"]),
        ("HashMap", vec!["utils.rs"]),
        ("impl", vec!["traits.rs"]),
    ];

    let mut correct_results = 0;
    let total_queries = test_queries.len();

    for (query, expected_files) in test_queries {
        match engine.search(query, 5) {
            Ok(results) => {
                let found_files: Vec<String> = results.iter()
                    .map(|r| r.doc_id.clone())
                    .collect();
                
                let hits = expected_files.iter()
                    .filter(|expected| found_files.contains(&expected.to_string()))
                    .count();

                if hits > 0 {
                    correct_results += 1;
                    result.add_detail(&format!("Query '{}' found {}/{} expected files", query, hits, expected_files.len()));
                } else {
                    result.add_detail(&format!("Query '{}' found NO expected files. Got: {:?}, Expected: {:?}", 
                        query, found_files, expected_files));
                }
            },
            Err(e) => {
                result.fail(&format!("Search failed for query '{}': {}", query, e));
                return result;
            }
        }
    }

    let elapsed = start_time.elapsed();
    result.performance_ms = elapsed.as_millis() as u64;
    result.accuracy_percent = (correct_results as f64 / total_queries as f64) * 100.0;

    // Check brutal requirements
    if result.accuracy_percent >= BRUTAL_LIMITS.min_accuracy_percent {
        result.pass();
    } else {
        result.fail(&format!("Accuracy {:.1}% < required {:.1}%", 
            result.accuracy_percent, BRUTAL_LIMITS.min_accuracy_percent));
    }

    result
}

/// TEST 2: PERFORMANCE BENCHMARK - BRUTAL TRUTH
fn test_performance_benchmark() -> BrutalTestResult {
    let mut result = BrutalTestResult::new("Performance Benchmark");
    result.memory_mb = get_memory_usage();

    let start_time = Instant::now();
    let mut engine = BM25Engine::new();

    // Add 100 documents
    for i in 0..100 {
        let content = format!("fn function_{i}() {{ let var_{i} = {i}; process_data(var_{i}); }}");
        let tokens: Vec<Token> = content
            .split_whitespace()
            .enumerate()
            .map(|(pos, word)| Token {
                text: word.to_string(),
                position: pos,
                importance_weight: 1.0,
            })
            .collect();

        let doc = BM25Document {
            id: format!("file_{}.rs", i),
            file_path: format!("file_{}.rs", i),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: 1,
            language: Some("rust".to_string()),
        };

        if let Err(e) = engine.add_document(doc) {
            result.fail(&format!("Failed to add document {}: {}", i, e));
            return result;
        }
    }

    // Perform 100 searches
    let search_start = Instant::now();
    let mut successful_searches = 0;

    for i in 0..100 {
        let query = format!("function_{}", i % 10);
        match engine.search(&query, 5) {
            Ok(_) => successful_searches += 1,
            Err(_) => {} // Count failures but don't stop
        }
    }

    let search_duration = search_start.elapsed();
    let total_duration = start_time.elapsed();

    // Calculate metrics
    result.performance_ms = search_duration.as_millis() as u64;
    result.operations_per_sec = if search_duration.as_secs_f64() > 0.0 {
        100.0 / search_duration.as_secs_f64()
    } else {
        0.0
    };
    result.accuracy_percent = (successful_searches as f64 / 100.0) * 100.0;

    result.add_detail(&format!("Total indexing + search time: {}ms", total_duration.as_millis()));
    result.add_detail(&format!("Search-only time: {}ms", search_duration.as_millis()));
    result.add_detail(&format!("Successful searches: {}/100", successful_searches));

    // Check brutal requirements
    let perf_ok = result.operations_per_sec >= BRUTAL_LIMITS.min_ops_per_second as f64;
    let search_time_ok = result.performance_ms <= BRUTAL_LIMITS.max_search_time_ms * 2; // Allow 2x for 100 searches
    let accuracy_ok = result.accuracy_percent >= 90.0; // Should be higher for simple queries

    if perf_ok && search_time_ok && accuracy_ok {
        result.pass();
    } else {
        let mut failures = Vec::new();
        if !perf_ok {
            failures.push(format!("Ops/sec {:.0} < {}", result.operations_per_sec, BRUTAL_LIMITS.min_ops_per_second));
        }
        if !search_time_ok {
            failures.push(format!("Search time {}ms > {}ms", result.performance_ms, BRUTAL_LIMITS.max_search_time_ms * 2));
        }
        if !accuracy_ok {
            failures.push(format!("Accuracy {:.1}% < 90%", result.accuracy_percent));
        }
        result.fail(&failures.join(", "));
    }

    result
}

/// TEST 3: MEMORY USAGE - BRUTAL TRUTH
fn test_memory_usage() -> BrutalTestResult {
    let mut result = BrutalTestResult::new("Memory Usage");
    
    let initial_memory = get_memory_usage();

    // Create storage with test data
    let storage = match VectorStorage::new(StorageConfig {
        max_vectors: 1000,
        dimension: 128,
        cache_size: 100,
        enable_compression: false,
    }) {
        Ok(s) => s,
        Err(e) => {
            result.fail(&format!("Failed to create storage: {}", e));
            return result;
        }
    };

    // Add test vectors
    for i in 0..500 {
        let vector = vec![0.1f32; 128];
        let metadata = format!("test_doc_{}", i);
        
        if let Err(e) = storage.store(i.to_string(), vector, Some(metadata)) {
            result.fail(&format!("Failed to store vector {}: {}", i, e));
            return result;
        }
    }

    let final_memory = get_memory_usage();
    result.memory_mb = final_memory - initial_memory;

    result.add_detail(&format!("Initial memory: {:.2} MB", initial_memory));
    result.add_detail(&format!("Final memory: {:.2} MB", final_memory));
    result.add_detail(&format!("Memory used: {:.2} MB", result.memory_mb));

    if result.memory_mb <= BRUTAL_LIMITS.max_memory_mb as f64 {
        result.pass();
    } else {
        result.fail(&format!("Memory usage {:.2} MB > {} MB limit", 
            result.memory_mb, BRUTAL_LIMITS.max_memory_mb));
    }

    result
}

/// TEST 4: CACHE FUNCTIONALITY - BRUTAL TRUTH
fn test_cache_functionality() -> BrutalTestResult {
    let mut result = BrutalTestResult::new("Cache Functionality");
    result.memory_mb = get_memory_usage();

    let start_time = Instant::now();

    // Test bounded cache
    let mut cache: BoundedCache<String, String> = match BoundedCache::new(100) {
        Ok(c) => c,
        Err(e) => {
            result.fail(&format!("Failed to create cache: {}", e));
            return result;
        }
    };

    // Fill cache
    for i in 0..150 {  // Exceed capacity to test eviction
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);
        cache.put(key, value);
    }

    // Test retrieval
    let mut hits = 0;
    let mut misses = 0;

    for i in 100..150 {  // Recent entries should be in cache
        let key = format!("key_{}", i);
        if cache.get(&key).is_some() {
            hits += 1;
        } else {
            misses += 1;
        }
    }

    for i in 0..50 {  // Old entries should be evicted
        let key = format!("key_{}", i);
        if cache.get(&key).is_some() {
            misses += 1;  // This is unexpected
        } else {
            hits += 1;   // This is expected (cache miss)
        }
    }

    result.performance_ms = start_time.elapsed().as_millis() as u64;
    result.accuracy_percent = (hits as f64 / (hits + misses) as f64) * 100.0;

    result.add_detail(&format!("Cache hits: {}", hits));
    result.add_detail(&format!("Cache misses: {}", misses));
    result.add_detail(&format!("Cache accuracy: {:.1}%", result.accuracy_percent));

    if result.accuracy_percent >= 70.0 {  // Cache should work reasonably well
        result.pass();
    } else {
        result.fail(&format!("Cache accuracy {:.1}% < 70%", result.accuracy_percent));
    }

    result
}

/// GENERATE BRUTAL TRUTH REPORT
fn generate_report(results: &[BrutalTestResult]) -> String {
    let mut report = String::new();
    report.push_str("🚨 BRUTAL TRUTH INTEGRATION TEST REPORT 🚨\n");
    report.push_str("==========================================\n\n");

    let passed_count = results.iter().filter(|r| r.passed).count();
    let total_count = results.len();

    if passed_count == total_count {
        report.push_str(&format!("✅ RESULT: {}/{} TESTS PASSED - SYSTEM OPERATIONAL\n\n", passed_count, total_count));
    } else {
        report.push_str(&format!("❌ RESULT: {}/{} TESTS PASSED - SYSTEM HAS FAILURES\n\n", passed_count, total_count));
    }

    for result in results {
        report.push_str(&format!("TEST: {}\n", result.test_name));
        report.push_str(&format!("STATUS: {}\n", if result.passed { "✅ PASS" } else { "❌ FAIL" }));
        
        if let Some(ref error) = result.error {
            report.push_str(&format!("ERROR: {}\n", error));
        }

        report.push_str(&format!("METRICS:\n"));
        report.push_str(&format!("  - Time: {} ms\n", result.performance_ms));
        report.push_str(&format!("  - Memory: {:.2} MB\n", result.memory_mb));
        report.push_str(&format!("  - Ops/Sec: {:.0}\n", result.operations_per_sec));
        report.push_str(&format!("  - Accuracy: {:.1}%\n", result.accuracy_percent));

        if !result.details.is_empty() {
            report.push_str("DETAILS:\n");
            for detail in &result.details {
                report.push_str(&format!("  • {}\n", detail));
            }
        }

        report.push_str("\n---\n\n");
    }

    // BRUTAL BENCHMARK SUMMARY
    report.push_str("🔥 PERFORMANCE SUMMARY 🔥\n");
    report.push_str("========================\n\n");

    if let Some(perf_test) = results.iter().find(|r| r.test_name.contains("Performance")) {
        report.push_str(&format!("Operations/Second: {:.0}\n", perf_test.operations_per_sec));
        report.push_str(&format!("Search Time: {} ms\n", perf_test.performance_ms));
        report.push_str(&format!("Accuracy: {:.1}%\n", perf_test.accuracy_percent));

        let ops_ok = perf_test.operations_per_sec >= BRUTAL_LIMITS.min_ops_per_second as f64;
        let time_ok = perf_test.performance_ms <= BRUTAL_LIMITS.max_search_time_ms * 2;
        
        report.push_str(&format!("\nREQUIREMENT CHECK:\n"));
        report.push_str(&format!("  Ops/Sec: {} (min: {})\n", 
            if ops_ok { "✅ PASS" } else { "❌ FAIL" }, BRUTAL_LIMITS.min_ops_per_second));
        report.push_str(&format!("  Time: {} (max: {} ms)\n", 
            if time_ok { "✅ PASS" } else { "❌ FAIL" }, BRUTAL_LIMITS.max_search_time_ms * 2));
    }

    report.push_str("\n🎯 BRUTAL FINAL VERDICT:\n");
    if passed_count == total_count {
        report.push_str("SYSTEM MEETS BRUTAL REQUIREMENTS ✅\n");
        report.push_str("Ready for basic production workloads.\n");
    } else {
        report.push_str("SYSTEM FAILS BRUTAL REQUIREMENTS ❌\n");
        report.push_str("NOT READY FOR PRODUCTION.\n");
        report.push_str("Fix failures before deployment.\n");
    }

    report
}

/// MAIN TEST ENTRY POINT
#[tokio::test]
async fn brutal_truth_integration_test() {
    println!("🚨 STARTING BRUTAL TRUTH INTEGRATION TESTS...\n");

    let mut results = Vec::new();

    // Run all brutal tests
    results.push(test_bm25_core());
    results.push(test_performance_benchmark());
    results.push(test_memory_usage());
    results.push(test_cache_functionality());

    // Generate and print report
    let report = generate_report(&results);
    println!("{}", report);

    // Fail the test if any individual test failed
    let failed_tests: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failed_tests.is_empty() {
        panic!("BRUTAL TRUTH: {} out of {} tests failed", failed_tests.len(), results.len());
    }
}

/// STANDALONE BENCHMARK
#[test]
fn brutal_benchmark_operations_per_second() {
    let start = Instant::now();
    let mut operations = 0u64;

    // Simple operations test
    let mut engine = BM25Engine::new();

    // Add 50 small documents
    for i in 0..50 {
        let tokens = vec![
            Token { text: format!("word_{}", i), position: 0, importance_weight: 1.0 },
            Token { text: "common".to_string(), position: 1, importance_weight: 1.0 },
        ];

        let doc = BM25Document {
            id: format!("doc_{}", i),
            file_path: format!("file_{}.rs", i),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: 2,
            language: Some("rust".to_string()),
        };

        if engine.add_document(doc).is_ok() {
            operations += 1;
        }
    }

    // Perform 50 searches
    for i in 0..50 {
        let query = format!("word_{}", i % 10);
        if engine.search(&query, 3).is_ok() {
            operations += 1;
        }
    }

    let duration = start.elapsed();
    let ops_per_second = operations as f64 / duration.as_secs_f64();

    println!("BRUTAL BENCHMARK: {:.0} operations/second", ops_per_second);
    println!("Total operations: {}", operations);
    println!("Duration: {:.3} seconds", duration.as_secs_f64());

    // Assert minimum performance
    assert!(
        ops_per_second >= BRUTAL_LIMITS.min_ops_per_second as f64,
        "BRUTAL FAILURE: {:.0} ops/sec < {} required",
        ops_per_second,
        BRUTAL_LIMITS.min_ops_per_second
    );

    println!("✅ BENCHMARK PASSED: System meets performance requirements");
}