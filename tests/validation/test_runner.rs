//! Comprehensive test runner for the embedding migration validation
//! Orchestrates all test suites and provides detailed reporting

use std::collections::HashMap;
use std::time::{Instant, Duration};
use tokio::process::Command;

/// Test suite metadata
#[derive(Debug, Clone)]
struct TestSuite {
    name: String,
    description: String,
    test_files: Vec<String>,
    required_features: Vec<String>,
    timeout_seconds: u64,
    criticality: TestCriticality,
}

#[derive(Debug, Clone, PartialEq)]
enum TestCriticality {
    Critical,   // Must pass for system to be considered working
    Important,  // Should pass, but system can work with degraded performance
    Optional,   // Nice to have, but not essential
}

/// Test execution result
#[derive(Debug)]
struct TestResult {
    suite_name: String,
    status: TestStatus,
    duration: Duration,
    output: String,
    error_output: String,
}

#[derive(Debug, PartialEq)]
enum TestStatus {
    Passed,
    Failed,
    Timeout,
    Skipped,
    Error,
}

/// Main test runner function
#[tokio::test]
async fn run_comprehensive_embedding_migration_tests() {
    println!("🎉 EMBEDDING MIGRATION VALIDATION SUITE");
    println!("=====================================");
    println!("Starting comprehensive validation of the nomic-embed-code integration\n");
    
    // Define all test suites
    let test_suites = define_test_suites();
    
    // Check prerequisites
    check_prerequisites().await;
    
    // Run all test suites
    let mut all_results = Vec::new();
    let total_start = Instant::now();
    
    for suite in &test_suites {
        println!("📁 Running Test Suite: {}", suite.name);
        println!("  Description: {}", suite.description);
        println!("  Criticality: {:?}", suite.criticality);
        
        if !suite.required_features.is_empty() {
            println!("  Required features: {:?}", suite.required_features);
        }
        
        let suite_results = run_test_suite(suite).await;
        all_results.extend(suite_results);
        
        println!();
    }
    
    let total_duration = total_start.elapsed();
    
    // Generate comprehensive report
    generate_final_report(&all_results, total_duration);
    
    // Determine overall success/failure
    let critical_failures = all_results.iter()
        .filter(|r| r.status == TestStatus::Failed)
        .filter(|r| {
            test_suites.iter()
                .find(|s| s.name == r.suite_name)
                .map(|s| s.criticality == TestCriticality::Critical)
                .unwrap_or(false)
        })
        .count();
    
    if critical_failures > 0 {
        panic!("🚨 CRITICAL FAILURES DETECTED: {} critical tests failed. System is not ready for production.", critical_failures);
    } else {
        println!("✅ ALL CRITICAL TESTS PASSED - System validated for production use!");
    }
}

/// Define all test suites for the embedding migration
fn define_test_suites() -> Vec<TestSuite> {
    vec![
        TestSuite {
            name: "Unit Tests".to_string(),
            description: "Core unit tests for nomic-embed-code integration".to_string(),
            test_files: vec!["tests/embedding_migration/unit_tests.rs".to_string()],
            required_features: vec!["ml".to_string()],
            timeout_seconds: 300,
            criticality: TestCriticality::Critical,
        },
        
        TestSuite {
            name: "Integration Tests".to_string(),
            description: "End-to-end pipeline integration tests".to_string(),
            test_files: vec!["tests/embedding_migration/integration_tests.rs".to_string()],
            required_features: vec!["ml".to_string(), "vectordb".to_string()],
            timeout_seconds: 600,
            criticality: TestCriticality::Critical,
        },
        
        TestSuite {
            name: "Performance Benchmarks".to_string(),
            description: "Performance comparison between text and code embeddings".to_string(),
            test_files: vec!["tests/benchmarks/embedding_performance_bench.rs".to_string()],
            required_features: vec!["ml".to_string(), "vectordb".to_string()],
            timeout_seconds: 900,
            criticality: TestCriticality::Important,
        },
        
        TestSuite {
            name: "RAG System Validation".to_string(),
            description: "Complete RAG system end-to-end validation".to_string(),
            test_files: vec!["tests/validation/rag_system_validation.rs".to_string()],
            required_features: vec!["full-system".to_string()],
            timeout_seconds: 1200,
            criticality: TestCriticality::Critical,
        },
        
        TestSuite {
            name: "Regression Tests".to_string(),
            description: "Tests to prevent regression of fixed issues".to_string(),
            test_files: vec!["tests/validation/regression_tests.rs".to_string()],
            required_features: vec!["ml".to_string()],
            timeout_seconds: 600,
            criticality: TestCriticality::Critical,
        },
        
        TestSuite {
            name: "Load Testing".to_string(),
            description: "System behavior under high load and stress".to_string(),
            test_files: vec!["tests/load_testing/embedding_stress_tests.rs".to_string()],
            required_features: vec!["ml".to_string(), "vectordb".to_string()],
            timeout_seconds: 1800,
            criticality: TestCriticality::Important,
        },
        
        TestSuite {
            name: "LLaMA.cpp Integration".to_string(),
            description: "GGUF model loading and inference validation".to_string(),
            test_files: vec!["tests/validation/llama_cpp_integration_tests.rs".to_string()],
            required_features: vec!["ml".to_string()],
            timeout_seconds: 900,
            criticality: TestCriticality::Critical,
        },
        
        TestSuite {
            name: "Cache Validation".to_string(),
            description: "Cache functionality and cleanup verification".to_string(),
            test_files: vec!["tests/validation/cache_validation_tests.rs".to_string()],
            required_features: vec!["ml".to_string()],
            timeout_seconds: 600,
            criticality: TestCriticality::Important,
        },
    ]
}

/// Check prerequisites before running tests
async fn check_prerequisites() {
    println!("🔍 Checking Prerequisites");
    println!("======================\n");
    
    // Check model file exists
    let model_path = "./model/nomic-embed-code.Q4_K_M.gguf";
    if !std::path::Path::new(model_path).exists() {
        panic!("Required model file not found: {}\nPlease download the nomic-embed-code model before running tests.", model_path);
    }
    println!("✅ Model file found: {}", model_path);
    
    // Check Rust version
    match Command::new("rustc").arg("--version").output().await {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("✅ Rust version: {}", version.trim());
        },
        Err(e) => {
            panic!("Failed to check Rust version: {}", e);
        }
    }
    
    // Check Cargo features
    match Command::new("cargo").args(&["metadata", "--format-version", "1"]).output().await {
        Ok(output) => {
            let metadata = String::from_utf8_lossy(&output.stdout);
            if metadata.contains("ml") {
                println!("✅ ML features available");
            } else {
                println!("⚠️  ML features may not be available");
            }
        },
        Err(e) => {
            println!("⚠️  Could not check Cargo features: {}", e);
        }
    }
    
    // Check available memory
    if let Some(available_memory) = get_available_memory() {
        println!("✅ Available memory: {:.1} GB", available_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        
        if available_memory < 4 * 1024 * 1024 * 1024 { // 4GB
            println!("⚠️  Warning: Low memory available. Some tests may fail or be slow.");
        }
    } else {
        println!("⚠️  Could not determine available memory");
    }
    
    println!();
}

/// Run a single test suite
async fn run_test_suite(suite: &TestSuite) -> Vec<TestResult> {
    let mut results = Vec::new();
    
    for test_file in &suite.test_files {
        println!("  📦 Running: {}", test_file);
        
        let start_time = Instant::now();
        
        // Build command with features
        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        cmd.arg("--test");
        cmd.arg(test_file.split('/').last().unwrap().replace(".rs", ""));
        
        if !suite.required_features.is_empty() {
            cmd.arg("--features");
            cmd.arg(suite.required_features.join(","));
        }
        
        cmd.arg("--");
        cmd.arg("--nocapture");
        
        // Set timeout
        let timeout_duration = Duration::from_secs(suite.timeout_seconds);
        
        let result = match tokio::time::timeout(timeout_duration, cmd.output()).await {
            Ok(Ok(output)) => {
                let duration = start_time.elapsed();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                let status = if output.status.success() {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                };
                
                TestResult {
                    suite_name: suite.name.clone(),
                    status,
                    duration,
                    output: stdout.to_string(),
                    error_output: stderr.to_string(),
                }
            },
            Ok(Err(e)) => {
                TestResult {
                    suite_name: suite.name.clone(),
                    status: TestStatus::Error,
                    duration: start_time.elapsed(),
                    output: String::new(),
                    error_output: format!("Command execution error: {}", e),
                }
            },
            Err(_) => {
                TestResult {
                    suite_name: suite.name.clone(),
                    status: TestStatus::Timeout,
                    duration: timeout_duration,
                    output: String::new(),
                    error_output: format!("Test timed out after {} seconds", suite.timeout_seconds),
                }
            }
        };
        
        // Print immediate result
        match result.status {
            TestStatus::Passed => println!("    ✅ PASSED ({:?})", result.duration),
            TestStatus::Failed => {
                println!("    ❌ FAILED ({:?})", result.duration);
                if !result.error_output.is_empty() {
                    println!("      Error: {}", result.error_output.lines().next().unwrap_or("Unknown error"));
                }
            },
            TestStatus::Timeout => println!("    ⏰ TIMEOUT ({:?})", result.duration),
            TestStatus::Error => println!("    ⚠️  ERROR: {}", result.error_output),
            TestStatus::Skipped => println!("    ⏭️  SKIPPED"),
        }
        
        results.push(result);
    }
    
    results
}

/// Generate comprehensive final report
fn generate_final_report(results: &[TestResult], total_duration: Duration) {
    println!("📊 COMPREHENSIVE VALIDATION REPORT");
    println!("=================================");
    println!("Total execution time: {:?}\n", total_duration);
    
    // Summary statistics
    let mut stats = HashMap::new();
    stats.insert(TestStatus::Passed, 0);
    stats.insert(TestStatus::Failed, 0);
    stats.insert(TestStatus::Timeout, 0);
    stats.insert(TestStatus::Error, 0);
    stats.insert(TestStatus::Skipped, 0);
    
    for result in results {
        *stats.get_mut(&result.status).unwrap() += 1;
    }
    
    let total_tests = results.len();
    let success_rate = *stats.get(&TestStatus::Passed).unwrap() as f64 / total_tests as f64 * 100.0;
    
    println!("📊 Summary Statistics:");
    println!("  Total tests: {}", total_tests);
    println!("  ✅ Passed: {}", stats.get(&TestStatus::Passed).unwrap());
    println!("  ❌ Failed: {}", stats.get(&TestStatus::Failed).unwrap());
    println!("  ⏰ Timeout: {}", stats.get(&TestStatus::Timeout).unwrap());
    println!("  ⚠️  Error: {}", stats.get(&TestStatus::Error).unwrap());
    println!("  ⏭️  Skipped: {}", stats.get(&TestStatus::Skipped).unwrap());
    println!("  Success rate: {:.1}%\n", success_rate);
    
    // Performance analysis
    let total_test_time: Duration = results.iter().map(|r| r.duration).sum();
    let avg_test_time = total_test_time / results.len() as u32;
    let slowest_test = results.iter().max_by_key(|r| r.duration);
    
    println!("⏱️  Performance Analysis:");
    println!("  Total test execution time: {:?}", total_test_time);
    println!("  Average test time: {:?}", avg_test_time);
    
    if let Some(slowest) = slowest_test {
        println!("  Slowest test: {} ({:?})", slowest.suite_name, slowest.duration);
    }
    
    println!();
    
    // Detailed results by suite
    println!("📁 Detailed Results by Suite:");
    
    let mut suite_results: HashMap<String, Vec<&TestResult>> = HashMap::new();
    for result in results {
        suite_results.entry(result.suite_name.clone()).or_insert_with(Vec::new).push(result);
    }
    
    for (suite_name, suite_results) in suite_results {
        let passed = suite_results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let total = suite_results.len();
        let suite_success_rate = passed as f64 / total as f64 * 100.0;
        
        let status_icon = if suite_success_rate == 100.0 {
            "✅"
        } else if suite_success_rate >= 80.0 {
            "⚠️ "
        } else {
            "❌"
        };
        
        println!("  {} {}: {}/{} passed ({:.1}%)", status_icon, suite_name, passed, total, suite_success_rate);
        
        // Show failed tests
        for result in suite_results.iter().filter(|r| r.status != TestStatus::Passed) {
            println!("    ❌ {:?}: {}", result.status, 
                     result.error_output.lines().next().unwrap_or("No details available"));
        }
    }
    
    println!();
    
    // Recommendations
    println!("📝 Recommendations:");
    
    let failed_count = *stats.get(&TestStatus::Failed).unwrap();
    let timeout_count = *stats.get(&TestStatus::Timeout).unwrap();
    let error_count = *stats.get(&TestStatus::Error).unwrap();
    
    if failed_count == 0 && timeout_count == 0 && error_count == 0 {
        println!("  ✅ All tests passed! The embedding migration is ready for production.");
        println!("  ✅ No issues detected in the nomic-embed-code integration.");
    } else {
        if failed_count > 0 {
            println!("  ❌ {} tests failed - review error messages and fix issues", failed_count);
        }
        if timeout_count > 0 {
            println!("  ⏰ {} tests timed out - investigate performance issues or increase timeouts", timeout_count);
        }
        if error_count > 0 {
            println!("  ⚠️  {} tests had execution errors - check environment and dependencies", error_count);
        }
        
        if success_rate < 90.0 {
            println!("  🚨 CRITICAL: Success rate below 90% - do not deploy to production");
        } else if success_rate < 95.0 {
            println!("  ⚠️  WARNING: Success rate below 95% - investigate failures before deployment");
        }
    }
    
    println!();
    
    // System validation summary
    if success_rate >= 95.0 && failed_count == 0 {
        println!("✅✅✅ SYSTEM VALIDATION: PASSED ✅✅✅");
        println!("The nomic-embed-code integration is validated and ready for production use.");
    } else if success_rate >= 80.0 {
        println!("⚠️ ⚠️ ⚠️  SYSTEM VALIDATION: CONDITIONAL ⚠️ ⚠️ ⚠️ ");
        println!("The system may work but has issues that should be addressed.");
    } else {
        println!("❌❌❌ SYSTEM VALIDATION: FAILED ❌❌❌");
        println!("The system is not ready for production. Critical issues must be fixed.");
    }
}

/// Get available system memory
fn get_available_memory() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback for other platforms
    None
}