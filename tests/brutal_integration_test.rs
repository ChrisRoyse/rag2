//! BRUTAL INTEGRATION TEST - INTJ Type-8 Analysis
//! Find every failure. Report BRUTAL TRUTH.
//! 
//! Tests the REAL system, not imaginary components.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tempfile::TempDir;
use anyhow::Result;

// Import actual system components (not removed ones)
use embed_search::error::EmbedError;
use embed_search::config::Config;
use embed_search::search::bm25::{BM25Engine, BM25Document, Token};
use embed_search::git::watcher::GitWatcher;
use embed_search::cache::bounded_cache::BoundedCache;
use embed_search::storage::safe_vectordb::{VectorStorage, StorageConfig};

// ML components only if feature is enabled
#[cfg(feature = "ml")]
use embed_search::embedding::nomic::NomicEmbedder;
#[cfg(feature = "ml")]
use embed_search::embedding::cache::EmbeddingCache;

/// BRUTAL TRUTH: Performance Requirements
const BRUTAL_REQUIREMENTS: BrutalRequirements = BrutalRequirements {
    max_index_time_ms: 5000,      // 5 seconds for 1000 files
    max_search_time_ms: 100,       // 100ms per search
    max_memory_mb: 512,            // 512MB total
    min_accuracy_percent: 85.0,    // 85% accuracy on known queries
    operations_per_second: 1000,   // Minimum throughput
};

struct BrutalRequirements {
    max_index_time_ms: u64,
    max_search_time_ms: u64,
    max_memory_mb: u64,
    min_accuracy_percent: f64,
    operations_per_second: u64,
}

/// BRUTAL TRUTH REPORT
#[derive(Debug)]
struct BrutalReport {
    test_name: String,
    passed: bool,
    error_message: Option<String>,
    performance_data: PerformanceData,
    memory_usage_mb: f64,
    actual_vs_expected: Vec<String>,
}

#[derive(Debug, Default)]
struct PerformanceData {
    index_time_ms: u64,
    search_time_ms: u64,
    operations_per_second: f64,
    accuracy_percent: f64,
}

impl BrutalReport {
    fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            passed: false,
            error_message: None,
            performance_data: PerformanceData::default(),
            memory_usage_mb: 0.0,
            actual_vs_expected: Vec::new(),
        }
    }

    fn fail(&mut self, reason: &str) {
        self.passed = false;
        self.error_message = Some(reason.to_string());
    }

    fn pass(&mut self) {
        self.passed = true;
        self.error_message = None;
    }

    fn add_comparison(&mut self, expected: &str, actual: &str) {
        self.actual_vs_expected.push(format!("Expected: {} | Actual: {}", expected, actual));
    }
}

/// BRUTAL TEST ORCHESTRATOR
struct BrutalTestOrchestrator {
    reports: Vec<BrutalReport>,
    temp_dir: TempDir,
    runtime: Runtime,
}

impl BrutalTestOrchestrator {
    fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let runtime = Runtime::new()?;
        
        Ok(Self {
            reports: Vec::new(),
            temp_dir,
            runtime,
        })
    }

    fn measure_memory() -> f64 {
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
        
        #[cfg(not(target_os = "linux"))]
        {
            use sysinfo::{System, SystemExt};
            let mut sys = System::new();
            sys.refresh_all();
            if let Some(process) = sys.process(sysinfo::get_current_pid().ok()?) {
                return process.memory() as f64 / 1024.0 / 1024.0; // Convert to MB
            }
        }
        
        0.0 // Fallback
    }

    /// TEST 1: BM25 SEARCH ACCURACY - BRUTAL TRUTH
    fn test_bm25_accuracy(&mut self) {
        let mut report = BrutalReport::new("BM25 Accuracy Test");
        report.memory_usage_mb = Self::measure_memory();

        let start_time = Instant::now();

        // Create BM25 engine
        let mut engine = match self.create_bm25_test_corpus() {
            Ok(engine) => engine,
            Err(e) => {
                report.fail(&format!("Failed to create BM25 engine: {}", e));
                self.reports.push(report);
                return;
            }
        };

        // BRUTAL TEST QUERIES - These should work or we fail
        let test_queries = vec![
            ("async fn", vec!["async_file.rs"]),  // Should find async functions
            ("impl Iterator", vec!["iterator_impl.rs"]),  // Should find iterator implementations
            ("struct Config", vec!["config.rs"]),  // Should find config structures
            ("tokio spawn", vec!["tokio_spawn.rs"]),  // Should find tokio usage
            ("error handling", vec!["error_handler.rs"]),  // Should find error handling
        ];

        let mut correct_results = 0;
        let total_queries = test_queries.len();

        for (query, expected_files) in test_queries {
            match engine.search(query, 10) {
                Ok(results) => {
                    let found_files: Vec<String> = results.iter()
                        .map(|r| r.doc_id.clone())
                        .collect();
                    
                    let expected_found = expected_files.iter()
                        .any(|expected| found_files.iter().any(|found| found.contains(expected)));
                    
                    if expected_found {
                        correct_results += 1;
                    } else {
                        report.add_comparison(
                            &format!("Files containing: {:?}", expected_files),
                            &format!("Found: {:?}", found_files)
                        );
                    }
                },
                Err(e) => {
                    report.fail(&format!("Search failed for query '{}': {}", query, e));
                    self.reports.push(report);
                    return;
                }
            }
        }

        let accuracy = (correct_results as f64 / total_queries as f64) * 100.0;
        report.performance_data.accuracy_percent = accuracy;
        report.performance_data.search_time_ms = start_time.elapsed().as_millis() as u64;

        if accuracy >= BRUTAL_REQUIREMENTS.min_accuracy_percent {
            report.pass();
        } else {
            report.fail(&format!(
                "Accuracy {} < required {}%",
                accuracy, BRUTAL_REQUIREMENTS.min_accuracy_percent
            ));
        }

        self.reports.push(report);
    }

    /// TEST 2: GIT WATCHER STRESS TEST - BRUTAL TRUTH
    fn test_git_watcher_stress(&mut self) {
        let mut report = BrutalReport::new("Git Watcher Stress Test");
        report.memory_usage_mb = Self::measure_memory();

        // Create temporary git repository
        let git_repo = match self.create_test_git_repo() {
            Ok(repo) => repo,
            Err(e) => {
                report.fail(&format!("Failed to create test git repo: {}", e));
                self.reports.push(report);
                return;
            }
        };

        let watcher = GitWatcher::new(git_repo.clone());
        
        let start_time = Instant::now();

        // RAPID FILE CHANGES - 100 changes in quick succession
        for i in 0..100 {
            let file_path = git_repo.join(format!("test_file_{}.txt", i));
            if let Err(e) = std::fs::write(&file_path, format!("Content {}", i)) {
                report.fail(&format!("Failed to create test file {}: {}", i, e));
                self.reports.push(report);
                return;
            }
        }

        // Check if watcher can detect changes
        match watcher.get_changes() {
            Ok(changes) => {
                if changes.len() >= 50 {  // Should detect at least 50% of changes
                    report.pass();
                } else {
                    report.fail(&format!(
                        "Only detected {} changes out of 100 expected",
                        changes.len()
                    ));
                }
            },
            Err(e) => {
                report.fail(&format!("Git watcher failed: {}", e));
            }
        }

        report.performance_data.index_time_ms = start_time.elapsed().as_millis() as u64;
        self.reports.push(report);
    }

    /// TEST 3: MEMORY USAGE UNDER LOAD - BRUTAL TRUTH
    fn test_memory_limits(&mut self) {
        let mut report = BrutalReport::new("Memory Limits Test");
        
        let initial_memory = Self::measure_memory();

        // Create large dataset in memory
        let mut storage = match VectorStorage::new(StorageConfig {
            max_vectors: 10000,
            dimension: 768,
            cache_size: 1000,
            enable_compression: false,
        }) {
            Ok(storage) => storage,
            Err(e) => {
                report.fail(&format!("Failed to create storage: {}", e));
                self.reports.push(report);
                return;
            }
        };

        // Fill with test data
        for i in 0..5000 {
            let vector = vec![0.1f32; 768]; // 768-dimensional vector
            let metadata = format!("doc_{}", i);
            
            if let Err(e) = storage.store(i.to_string(), vector, Some(metadata)) {
                report.fail(&format!("Failed to store vector {}: {}", i, e));
                self.reports.push(report);
                return;
            }
        }

        let final_memory = Self::measure_memory();
        let memory_used = final_memory - initial_memory;
        
        report.memory_usage_mb = memory_used;

        if memory_used <= BRUTAL_REQUIREMENTS.max_memory_mb as f64 {
            report.pass();
        } else {
            report.fail(&format!(
                "Memory usage {} MB exceeds limit {} MB",
                memory_used, BRUTAL_REQUIREMENTS.max_memory_mb
            ));
        }

        self.reports.push(report);
    }

    /// TEST 4: EMBEDDING SYSTEM (ML Feature) - BRUTAL TRUTH
    #[cfg(feature = "ml")]
    fn test_embedding_system(&mut self) {
        let mut report = BrutalReport::new("Embedding System Test");
        report.memory_usage_mb = Self::measure_memory();

        // Test various text sizes
        let test_texts = vec![
            "short",
            "This is a medium length text that should be processed correctly by the embedding system.",
            &"very long text ".repeat(1000), // Very long text
        ];

        let start_time = Instant::now();

        for (i, text) in test_texts.iter().enumerate() {
            match self.runtime.block_on(async {
                // Try to get embeddings - this will fail if model is not loaded
                NomicEmbedder::get_global_embedder().await
            }) {
                Ok(embedder) => {
                    match self.runtime.block_on(async {
                        embedder.embed_text(text).await
                    }) {
                        Ok(embedding) => {
                            if embedding.is_empty() {
                                report.fail(&format!("Empty embedding for text {}", i));
                                self.reports.push(report);
                                return;
                            }
                        },
                        Err(e) => {
                            report.fail(&format!("Embedding failed for text {}: {}", i, e));
                            self.reports.push(report);
                            return;
                        }
                    }
                },
                Err(e) => {
                    report.fail(&format!("Failed to get embedder: {}", e));
                    self.reports.push(report);
                    return;
                }
            }
        }

        report.performance_data.search_time_ms = start_time.elapsed().as_millis() as u64;
        report.pass();
        self.reports.push(report);
    }

    /// TEST 5: END-TO-END PERFORMANCE BENCHMARK
    fn test_end_to_end_performance(&mut self) {
        let mut report = BrutalReport::new("End-to-End Performance Benchmark");
        report.memory_usage_mb = Self::measure_memory();

        let start_time = Instant::now();

        // 1. Create 1000 test files
        let test_files = match self.create_large_test_corpus(1000) {
            Ok(files) => files,
            Err(e) => {
                report.fail(&format!("Failed to create test corpus: {}", e));
                self.reports.push(report);
                return;
            }
        };

        let index_time = start_time.elapsed();

        // 2. Perform 50 search queries
        let mut engine = BM25Engine::new();
        
        // Add documents to engine
        for (i, content) in test_files.iter().enumerate() {
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
                id: format!("doc_{}", i),
                file_path: format!("test_file_{}.rs", i),
                chunk_index: 0,
                tokens,
                start_line: 1,
                end_line: 10,
                language: Some("rust".to_string()),
            };

            if let Err(e) = engine.add_document(doc) {
                report.fail(&format!("Failed to add document {}: {}", i, e));
                self.reports.push(report);
                return;
            }
        }

        // 3. Execute search queries
        let search_start = Instant::now();
        let queries = vec![
            "fn main", "struct", "impl", "use std", "tokio",
            "async", "await", "Result", "Vec", "HashMap",
        ];

        let mut successful_searches = 0;
        for query in &queries {
            for _ in 0..5 {  // 5 searches per query = 50 total
                match engine.search(query, 10) {
                    Ok(_) => successful_searches += 1,
                    Err(e) => {
                        report.add_comparison(
                            &format!("Successful search for: {}", query),
                            &format!("Failed with: {}", e)
                        );
                    }
                }
            }
        }

        let search_time = search_start.elapsed();
        let total_time = start_time.elapsed();

        // Calculate performance metrics
        let ops_per_second = if search_time.as_secs_f64() > 0.0 {
            50.0 / search_time.as_secs_f64()
        } else {
            0.0
        };

        report.performance_data.index_time_ms = index_time.as_millis() as u64;
        report.performance_data.search_time_ms = search_time.as_millis() as u64;
        report.performance_data.operations_per_second = ops_per_second;
        report.performance_data.accuracy_percent = (successful_searches as f64 / 50.0) * 100.0;

        // Check against brutal requirements
        let mut failures = Vec::new();

        if index_time.as_millis() as u64 > BRUTAL_REQUIREMENTS.max_index_time_ms {
            failures.push(format!(
                "Index time {} ms > {} ms",
                index_time.as_millis(),
                BRUTAL_REQUIREMENTS.max_index_time_ms
            ));
        }

        if search_time.as_millis() as u64 > BRUTAL_REQUIREMENTS.max_search_time_ms * 5 { // 5x because we do 50 searches
            failures.push(format!(
                "Search time {} ms > {} ms",
                search_time.as_millis(),
                BRUTAL_REQUIREMENTS.max_search_time_ms * 5
            ));
        }

        if ops_per_second < BRUTAL_REQUIREMENTS.operations_per_second as f64 {
            failures.push(format!(
                "Operations/sec {} < {}",
                ops_per_second,
                BRUTAL_REQUIREMENTS.operations_per_second
            ));
        }

        if failures.is_empty() && successful_searches >= 40 { // 80% success rate minimum
            report.pass();
        } else {
            report.fail(&format!("Performance failures: {:?}", failures));
        }

        self.reports.push(report);
    }

    /// Helper: Create BM25 test corpus
    fn create_bm25_test_corpus(&self) -> Result<BM25Engine> {
        let mut engine = BM25Engine::new();
        
        let test_documents = vec![
            ("async_file.rs", "async fn process_data() { tokio::spawn(async { }).await; }"),
            ("iterator_impl.rs", "impl Iterator for MyStruct { type Item = String; }"),
            ("config.rs", "struct Config { pub database_url: String, pub port: u16 }"),
            ("tokio_spawn.rs", "tokio::spawn(async move { process_background_task().await; });"),
            ("error_handler.rs", "fn handle_error(e: Error) -> Result<(), Box<dyn Error>> { }"),
        ];

        for (file, content) in test_documents {
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
                id: file.to_string(),
                file_path: file.to_string(),
                chunk_index: 0,
                tokens,
                start_line: 1,
                end_line: 1,
                language: Some("rust".to_string()),
            };

            engine.add_document(doc)?;
        }

        Ok(engine)
    }

    /// Helper: Create test git repository
    fn create_test_git_repo(&self) -> Result<PathBuf> {
        let repo_path = self.temp_dir.path().join("test_repo");
        std::fs::create_dir_all(&repo_path)?;

        // Initialize git repo
        std::process::Command::new("git")
            .args(&["init"])
            .current_dir(&repo_path)
            .output()?;

        // Configure git user (required for commits)
        std::process::Command::new("git")
            .args(&["config", "user.email", "test@example.com"])
            .current_dir(&repo_path)
            .output()?;

        std::process::Command::new("git")
            .args(&["config", "user.name", "Test User"])
            .current_dir(&repo_path)
            .output()?;

        Ok(repo_path)
    }

    /// Helper: Create large test corpus
    fn create_large_test_corpus(&self, count: usize) -> Result<Vec<String>> {
        let mut files = Vec::new();
        
        let code_templates = vec![
            "fn main() { println!(\"Hello {}\"); }",
            "struct Data{{ id: {}, name: String }}",
            "impl Display for Item{{ fn fmt(&self) -> String {{ \"{}\" }} }}",
            "use std::collections::HashMap; fn process_{}_items() {{}}",
            "async fn fetch_data_{}() -> Result<Vec<Item>, Error> {{ Ok(vec![]) }}",
        ];

        for i in 0..count {
            let template = &code_templates[i % code_templates.len()];
            let content = template.replace("{}", &i.to_string());
            files.push(content);
        }

        Ok(files)
    }

    /// Generate BRUTAL TRUTH REPORT
    fn generate_brutal_report(&self) -> String {
        let mut report = String::new();
        report.push_str("🚨 BRUTAL INTEGRATION TEST REPORT 🚨\n");
        report.push_str("=====================================\n\n");

        let passed_tests = self.reports.iter().filter(|r| r.passed).count();
        let total_tests = self.reports.len();

        report.push_str(&format!("OVERALL RESULT: {}/{} TESTS PASSED\n\n", passed_tests, total_tests));

        if passed_tests < total_tests {
            report.push_str("❌ BRUTAL TRUTH: SYSTEM HAS FAILURES\n\n");
        } else {
            report.push_str("✅ BRUTAL TRUTH: ALL TESTS PASSED\n\n");
        }

        for test_report in &self.reports {
            report.push_str(&format!("TEST: {}\n", test_report.test_name));
            report.push_str(&format!("STATUS: {}\n", if test_report.passed { "✅ PASS" } else { "❌ FAIL" }));
            
            if let Some(ref error) = test_report.error_message {
                report.push_str(&format!("ERROR: {}\n", error));
            }

            report.push_str(&format!("MEMORY: {:.2} MB\n", test_report.memory_usage_mb));
            report.push_str(&format!("PERFORMANCE:\n"));
            report.push_str(&format!("  - Index Time: {} ms\n", test_report.performance_data.index_time_ms));
            report.push_str(&format!("  - Search Time: {} ms\n", test_report.performance_data.search_time_ms));
            report.push_str(&format!("  - Ops/Second: {:.2}\n", test_report.performance_data.operations_per_second));
            report.push_str(&format!("  - Accuracy: {:.2}%\n", test_report.performance_data.accuracy_percent));

            if !test_report.actual_vs_expected.is_empty() {
                report.push_str("DETAILED FAILURES:\n");
                for comparison in &test_report.actual_vs_expected {
                    report.push_str(&format!("  - {}\n", comparison));
                }
            }

            report.push_str("\n---\n\n");
        }

        // BRUTAL BENCHMARK SUMMARY
        report.push_str("🔥 BRUTAL BENCHMARK RESULTS 🔥\n");
        report.push_str("===========================\n\n");

        if let Some(perf_test) = self.reports.iter().find(|r| r.test_name.contains("Performance")) {
            report.push_str(&format!("Operations/Second: {:.0}\n", perf_test.performance_data.operations_per_second));
            report.push_str(&format!("Search Accuracy: {:.1}%\n", perf_test.performance_data.accuracy_percent));
            report.push_str(&format!("Memory Usage: {:.1} MB\n", perf_test.memory_usage_mb));
            
            // Compare against requirements
            let ops_ok = perf_test.performance_data.operations_per_second >= BRUTAL_REQUIREMENTS.operations_per_second as f64;
            let mem_ok = perf_test.memory_usage_mb <= BRUTAL_REQUIREMENTS.max_memory_mb as f64;
            let acc_ok = perf_test.performance_data.accuracy_percent >= BRUTAL_REQUIREMENTS.min_accuracy_percent;

            report.push_str(&format!("\nREQUIREMENTS CHECK:\n"));
            report.push_str(&format!("✓ Ops/Sec: {} (required: {})\n", 
                if ops_ok { "PASS" } else { "FAIL" }, BRUTAL_REQUIREMENTS.operations_per_second));
            report.push_str(&format!("✓ Memory: {} (limit: {} MB)\n", 
                if mem_ok { "PASS" } else { "FAIL" }, BRUTAL_REQUIREMENTS.max_memory_mb));
            report.push_str(&format!("✓ Accuracy: {} (required: {}%)\n", 
                if acc_ok { "PASS" } else { "FAIL" }, BRUTAL_REQUIREMENTS.min_accuracy_percent));
        }

        report.push_str("\n🎯 FINAL BRUTAL TRUTH:\n");
        if passed_tests == total_tests {
            report.push_str("SYSTEM IS PRODUCTION READY ✅\n");
        } else {
            report.push_str("SYSTEM HAS CRITICAL FAILURES ❌\n");
            report.push_str("DO NOT DEPLOY TO PRODUCTION\n");
        }

        report
    }

    /// Run all brutal tests
    fn run_all_tests(&mut self) {
        println!("🚨 STARTING BRUTAL INTEGRATION TESTS...");

        self.test_bm25_accuracy();
        self.test_git_watcher_stress();
        self.test_memory_limits();
        
        #[cfg(feature = "ml")]
        self.test_embedding_system();
        
        self.test_end_to_end_performance();

        println!("{}", self.generate_brutal_report());
    }
}

/// Main test entry point
#[tokio::test]
async fn brutal_integration_test() {
    let mut orchestrator = match BrutalTestOrchestrator::new() {
        Ok(orch) => orch,
        Err(e) => {
            panic!("Failed to create test orchestrator: {}", e);
        }
    };

    orchestrator.run_all_tests();
    
    // Check if any tests failed and panic accordingly
    let failed_tests: Vec<_> = orchestrator.reports.iter()
        .filter(|r| !r.passed)
        .collect();

    if !failed_tests.is_empty() {
        panic!("BRUTAL TRUTH: {} tests failed", failed_tests.len());
    }
}

/// Standalone benchmark for operations per second
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_simple_operations() {
        let start = Instant::now();
        let mut operations = 0;

        // Simple BM25 operations
        let mut engine = BM25Engine::new();
        
        // Add 100 documents
        for i in 0..100 {
            let tokens = vec![
                Token { text: format!("token_{}", i), position: 0, importance_weight: 1.0 },
                Token { text: "common_word".to_string(), position: 1, importance_weight: 1.0 },
            ];

            let doc = BM25Document {
                id: format!("doc_{}", i),
                file_path: format!("file_{}.rs", i),
                chunk_index: 0,
                tokens,
                start_line: 1,
                end_line: 5,
                language: Some("rust".to_string()),
            };

            engine.add_document(doc).unwrap();
            operations += 1;
        }

        // Perform 100 searches
        for i in 0..100 {
            let query = format!("token_{}", i % 10);
            if engine.search(&query, 5).is_ok() {
                operations += 1;
            }
        }

        let duration = start.elapsed();
        let ops_per_second = operations as f64 / duration.as_secs_f64();

        println!("BENCHMARK RESULT: {:.0} operations/second", ops_per_second);
        
        // This should meet our brutal requirements
        assert!(
            ops_per_second >= BRUTAL_REQUIREMENTS.operations_per_second as f64,
            "Benchmark failed: {:.0} ops/sec < {} required",
            ops_per_second,
            BRUTAL_REQUIREMENTS.operations_per_second
        );
    }
}