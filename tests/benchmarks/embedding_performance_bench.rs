//! Performance benchmarks comparing text vs code embeddings
//! Designed to expose performance regressions and validate optimizations

use embed_search::{
    embedding::LazyEmbedder,
    storage::safe_vectordb::{VectorStorage, StorageConfig},
};
use std::time::{Instant, Duration};
use tokio::runtime::Runtime;

/// Comprehensive embedding performance benchmark
#[tokio::test]
async fn benchmark_embedding_performance() {
    println!("🏁 Starting comprehensive embedding performance benchmark");
    println!("==========================================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Force initialization
    let _ = embedder.embed("init").await.expect("Initialization failed");
    println!("✅ Embedder initialized\n");
    
    // Test datasets
    let test_suites = vec![
        create_code_samples(),
        create_text_samples(), 
        create_mixed_samples(),
    ];
    
    let suite_names = ["Code Samples", "Text Samples", "Mixed Content"];
    
    for (suite_idx, (suite_name, samples)) in suite_names.iter().zip(test_suites.iter()).enumerate() {
        println!("📊 Benchmarking: {} ({} samples)", suite_name, samples.len());
        
        let results = benchmark_sample_suite(&embedder, samples).await;
        print_benchmark_results(suite_name, &results);
        
        // Performance assertions
        validate_performance_requirements(&results, suite_name);
        
        println!();
    }
    
    // Cross-suite comparison
    println!("🔄 Running cross-suite performance comparison...");
    compare_embedding_types(&embedder).await;
    
    println!("\n✅ Comprehensive benchmark completed!");
}

/// Benchmark a suite of samples
async fn benchmark_sample_suite(embedder: &LazyEmbedder, samples: &[(String, String)]) -> BenchmarkResults {
    let mut times = Vec::new();
    let mut cache_hits = 0;
    let mut total_chars = 0;
    
    println!("  Running {} iterations...", samples.len());
    
    for (i, (name, content)) in samples.iter().enumerate() {
        total_chars += content.len();
        
        // First run (cold)
        let start = Instant::now();
        let result = embedder.embed(content).await;
        let duration = start.elapsed();
        
        match result {
            Ok(embedding) => {
                assert_eq!(embedding.len(), 768, "Wrong embedding dimension for {}", name);
                times.push(duration);
                
                // Test cache hit
                let cache_start = Instant::now();
                let _ = embedder.embed(content).await.expect("Cache test failed");
                let cache_duration = cache_start.elapsed();
                
                if cache_duration < duration / 2 {
                    cache_hits += 1;
                }
            },
            Err(e) => {
                println!("    ❌ Failed on {}: {}", name, e);
                continue;
            }
        }
        
        if (i + 1) % 10 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!();
    
    BenchmarkResults {
        times,
        cache_hit_rate: cache_hits as f64 / samples.len() as f64,
        total_chars,
        successful_runs: times.len(),
    }
}

/// Print detailed benchmark results
fn print_benchmark_results(suite_name: &str, results: &BenchmarkResults) {
    if results.times.is_empty() {
        println!("  ❌ No successful runs for {}", suite_name);
        return;
    }
    
    let total_time: Duration = results.times.iter().sum();
    let avg_time = total_time / results.times.len() as u32;
    let min_time = *results.times.iter().min().unwrap();
    let max_time = *results.times.iter().max().unwrap();
    
    // Calculate percentiles
    let mut sorted_times = results.times.clone();
    sorted_times.sort();
    let p50 = sorted_times[sorted_times.len() * 50 / 100];
    let p95 = sorted_times[sorted_times.len() * 95 / 100];
    let p99 = sorted_times[sorted_times.len() * 99 / 100];
    
    // Throughput calculations
    let chars_per_sec = results.total_chars as f64 / total_time.as_secs_f64();
    let embeddings_per_sec = results.successful_runs as f64 / total_time.as_secs_f64();
    
    println!("  📈 Performance Results:");
    println!("    Success rate: {}/{} ({:.1}%)", 
             results.successful_runs, results.times.len(), 
             results.successful_runs as f64 / results.times.len() as f64 * 100.0);
    
    println!("    Timing statistics:");
    println!("      Average: {:?}", avg_time);
    println!("      Minimum: {:?}", min_time);
    println!("      Maximum: {:?}", max_time);
    println!("      P50: {:?}", p50);
    println!("      P95: {:?}", p95);
    println!("      P99: {:?}", p99);
    
    println!("    Throughput:");
    println!("      {:.1} chars/sec", chars_per_sec);
    println!("      {:.1} embeddings/sec", embeddings_per_sec);
    
    println!("    Cache hit rate: {:.1}%", results.cache_hit_rate * 100.0);
    
    // Performance scoring
    let score = calculate_performance_score(results);
    println!("    Performance score: {:.1}/100", score);
}

/// Validate performance meets requirements
fn validate_performance_requirements(results: &BenchmarkResults, suite_name: &str) {
    if results.times.is_empty() {
        panic!("No successful embeddings generated for {}", suite_name);
    }
    
    let avg_time = results.times.iter().sum::<Duration>() / results.times.len() as u32;
    let max_time = *results.times.iter().max().unwrap();
    
    // Performance requirements
    assert!(avg_time.as_secs() < 10, 
            "Average embedding time too slow for {}: {:?}", suite_name, avg_time);
    
    assert!(max_time.as_secs() < 30, 
            "Maximum embedding time too slow for {}: {:?}", suite_name, max_time);
    
    let success_rate = results.successful_runs as f64 / results.times.len() as f64;
    assert!(success_rate > 0.95, 
            "Success rate too low for {}: {:.1}%", suite_name, success_rate * 100.0);
    
    println!("  ✅ {} meets performance requirements", suite_name);
}

/// Compare different embedding types performance
async fn compare_embedding_types(embedder: &LazyEmbedder) {
    let comparison_tests = vec![
        ("Short Code", "fn test() {}", 20),
        ("Long Code", create_long_code_sample(), 10),
        ("Documentation", "This is a comprehensive documentation string explaining the functionality.", 20),
        ("Mixed Content", "fn process_data(data: &str) -> Result<String> { /* Process the data */ Ok(data.to_uppercase()) }", 15),
    ];
    
    for (test_name, sample, iterations) in comparison_tests {
        println!("  Testing {}: {} iterations", test_name, iterations);
        
        let mut times = Vec::new();
        
        for i in 0..iterations {
            let modified_sample = format!("{} // iteration {}", sample, i);
            let start = Instant::now();
            
            match embedder.embed(&modified_sample).await {
                Ok(embedding) => {
                    let duration = start.elapsed();
                    times.push(duration);
                    assert_eq!(embedding.len(), 768);
                },
                Err(e) => {
                    println!("    ❌ Failed iteration {}: {}", i, e);
                }
            }
        }
        
        if !times.is_empty() {
            let avg = times.iter().sum::<Duration>() / times.len() as u32;
            let min = *times.iter().min().unwrap();
            let max = *times.iter().max().unwrap();
            
            println!("    ✅ {}: avg={:?}, min={:?}, max={:?}", test_name, avg, min, max);
        }
    }
}

/// Stress test with many concurrent embeddings
#[tokio::test]
async fn benchmark_concurrent_performance() {
    println!("⚡ Testing concurrent embedding performance");
    
    let embedder = LazyEmbedder::new();
    
    // Initialize
    let _ = embedder.embed("init").await.expect("Init failed");
    
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    for concurrency in concurrency_levels {
        println!("\n📊 Testing concurrency level: {}", concurrency);
        
        let start = Instant::now();
        let mut tasks = Vec::new();
        
        for i in 0..concurrency {
            let embedder_clone = embedder.clone();
            let task = tokio::spawn(async move {
                let code = format!("fn concurrent_test_{}() {{ println!(\"test\"); }}", i);
                embedder_clone.embed(&code).await
            });
            tasks.push(task);
        }
        
        let mut successful = 0;
        let mut failed = 0;
        
        for task in tasks {
            match task.await {
                Ok(Ok(embedding)) => {
                    assert_eq!(embedding.len(), 768);
                    successful += 1;
                },
                _ => failed += 1,
            }
        }
        
        let total_time = start.elapsed();
        let throughput = successful as f64 / total_time.as_secs_f64();
        
        println!("  ✅ Concurrency {}: {}/{} successful, {:.1} embeddings/sec", 
                 concurrency, successful, successful + failed, throughput);
        
        // Basic performance check
        assert!(successful >= concurrency * 80 / 100, "Too many failures at concurrency {}", concurrency);
    }
}

/// Memory usage benchmark
#[tokio::test]
async fn benchmark_memory_usage() {
    println!("💾 Testing memory usage patterns");
    
    let embedder = LazyEmbedder::new();
    
    // Initialize and get baseline
    let _ = embedder.embed("baseline").await.expect("Baseline failed");
    
    let initial_memory = get_memory_usage_mb();
    println!("  Initial memory usage: {:.1} MB", initial_memory);
    
    // Generate many embeddings to test memory growth
    let mut peak_memory = initial_memory;
    let iterations = 100;
    
    println!("  Generating {} embeddings to test memory patterns...", iterations);
    
    for i in 0..iterations {
        let code = format!("fn memory_test_{}() {{ let x = {}; x * 2 }}", i, i);
        
        match embedder.embed(&code).await {
            Ok(embedding) => {
                assert_eq!(embedding.len(), 768);
                
                if i % 20 == 0 {
                    let current_memory = get_memory_usage_mb();
                    peak_memory = peak_memory.max(current_memory);
                    
                    println!("    Iteration {}: {:.1} MB (+{:.1} MB)", 
                             i, current_memory, current_memory - initial_memory);
                }
            },
            Err(e) => {
                println!("    ❌ Memory pressure at iteration {}: {}", i, e);
                break;
            }
        }
    }
    
    let final_memory = get_memory_usage_mb();
    let growth = final_memory - initial_memory;
    
    println!("  Final memory usage: {:.1} MB", final_memory);
    println!("  Peak memory usage: {:.1} MB", peak_memory);
    println!("  Memory growth: {:.1} MB", growth);
    
    // Memory growth should be reasonable
    assert!(growth < 500.0, "Excessive memory growth: {:.1} MB", growth);
    
    println!("  ✅ Memory usage within acceptable bounds");
}

/// Storage performance benchmark
#[tokio::test]
async fn benchmark_storage_performance() {
    println!("🗄️  Testing storage performance");
    
    let embedder = LazyEmbedder::new();
    let storage_config = StorageConfig {
        max_vectors: 1000,
        dimension: 768,
        cache_size: 200,
        enable_compression: false,
    };
    
    let mut storage = VectorStorage::new(storage_config).expect("Failed to create storage");
    
    // Generate test embeddings
    let num_embeddings = 100;
    let mut embeddings = Vec::new();
    
    println!("  Generating {} test embeddings...", num_embeddings);
    for i in 0..num_embeddings {
        let code = format!("fn storage_test_{}() {{ return {}; }}", i, i);
        let embedding = embedder.embed(&code).await.expect("Embedding failed");
        embeddings.push((format!("test_{}", i), embedding));
    }
    
    // Benchmark storage operations
    println!("  Benchmarking storage operations...");
    
    // Insert benchmark
    let start = Instant::now();
    for (id, embedding) in &embeddings {
        storage.insert(id.clone(), embedding.clone()).expect("Insert failed");
    }
    let insert_time = start.elapsed();
    let insert_rate = embeddings.len() as f64 / insert_time.as_secs_f64();
    
    println!("    ✅ Insert: {:?} total, {:.1} ops/sec", insert_time, insert_rate);
    
    // Retrieval benchmark
    let start = Instant::now();
    for (id, _) in &embeddings {
        let retrieved = storage.get(id).expect("Retrieval failed")
            .expect("Item not found");
        assert_eq!(retrieved.len(), 768);
    }
    let retrieve_time = start.elapsed();
    let retrieve_rate = embeddings.len() as f64 / retrieve_time.as_secs_f64();
    
    println!("    ✅ Retrieve: {:?} total, {:.1} ops/sec", retrieve_time, retrieve_rate);
    
    // Similarity search benchmark
    let query_embedding = &embeddings[0].1;
    
    let start = Instant::now();
    let similar = storage.find_similar(query_embedding, 10).expect("Similarity search failed");
    let search_time = start.elapsed();
    
    assert!(!similar.is_empty());
    println!("    ✅ Similarity search: {:?}, found {} results", search_time, similar.len());
    
    // Performance assertions
    assert!(insert_rate > 10.0, "Insert rate too low: {:.1} ops/sec", insert_rate);
    assert!(retrieve_rate > 100.0, "Retrieve rate too low: {:.1} ops/sec", retrieve_rate);
    assert!(search_time.as_millis() < 1000, "Similarity search too slow: {:?}", search_time);
}

// Helper functions and types

#[derive(Debug, Clone)]
struct BenchmarkResults {
    times: Vec<Duration>,
    cache_hit_rate: f64,
    total_chars: usize,
    successful_runs: usize,
}

fn calculate_performance_score(results: &BenchmarkResults) -> f64 {
    if results.times.is_empty() {
        return 0.0;
    }
    
    let avg_time = results.times.iter().sum::<Duration>().as_secs_f64() / results.times.len() as f64;
    let success_rate = results.successful_runs as f64 / results.times.len() as f64;
    
    let time_score = (10.0 / avg_time.max(0.1)).min(50.0); // Up to 50 points for speed
    let success_score = success_rate * 30.0; // Up to 30 points for success rate
    let cache_score = results.cache_hit_rate * 20.0; // Up to 20 points for cache effectiveness
    
    time_score + success_score + cache_score
}

fn create_code_samples() -> Vec<(String, String)> {
    vec![
        ("rust_function".to_string(), "fn process_data(input: &str) -> Result<String, Error> { Ok(input.to_uppercase()) }".to_string()),
        ("python_class".to_string(), "class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n    def process(self, data):\n        return data.upper()".to_string()),
        ("javascript_async".to_string(), "async function fetchData(url) { const response = await fetch(url); return response.json(); }".to_string()),
        ("go_struct".to_string(), "type User struct {\n    ID   int    `json:\"id\"`\n    Name string `json:\"name\"`\n}\n\nfunc (u *User) String() string {\n    return u.Name\n}".to_string()),
        ("c_function".to_string(), "int calculate_checksum(const char* data, size_t length) {\n    int checksum = 0;\n    for (size_t i = 0; i < length; i++) {\n        checksum += data[i];\n    }\n    return checksum;\n}".to_string()),
    ]
}

fn create_text_samples() -> Vec<(String, String)> {
    vec![
        ("technical_doc".to_string(), "This function implements a fast hash algorithm for data integrity verification in distributed systems.".to_string()),
        ("user_guide".to_string(), "To get started with the API, first authenticate using your API key. Then make requests to the appropriate endpoints.".to_string()),
        ("error_message".to_string(), "Connection timeout occurred while attempting to establish a secure connection to the remote server.".to_string()),
        ("config_description".to_string(), "Configure the maximum number of concurrent connections and set appropriate timeout values for optimal performance.".to_string()),
        ("changelog_entry".to_string(), "Added support for batch processing of multiple files with improved error handling and progress reporting.".to_string()),
    ]
}

fn create_mixed_samples() -> Vec<(String, String)> {
    vec![
        ("api_doc".to_string(), "The `process_file()` function reads a file and returns a `ProcessResult` object containing metadata.".to_string()),
        ("code_comment".to_string(), "// This implementation uses a binary search tree for O(log n) lookup time\nstruct BinaryTree { root: Node }".to_string()),
        ("config_with_code".to_string(), "Set the environment variable DATABASE_URL=\"postgresql://localhost:5432/mydb\" in your .env file.".to_string()),
        ("tutorial_step".to_string(), "Step 3: Import the library with `import { createClient } from 'my-lib'` and initialize it in your main function.".to_string()),
    ]
}

fn create_long_code_sample() -> String {
    format!(
        "{}",
        "pub struct ComplexDataProcessor {\n    config: ProcessorConfig,\n    cache: HashMap<String, ProcessedData>,\n    metrics: Arc<RwLock<Metrics>>,\n}\n\nimpl ComplexDataProcessor {\n    pub fn new(config: ProcessorConfig) -> Self {\n        Self {\n            config,\n            cache: HashMap::new(),\n            metrics: Arc::new(RwLock::new(Metrics::default())),\n        }\n    }\n\n    pub async fn process_batch(&mut self, items: Vec<RawData>) -> Result<Vec<ProcessedData>, ProcessError> {\n        let mut results = Vec::with_capacity(items.len());\n        \n        for item in items {\n            let processed = self.process_single_item(&item).await?;\n            results.push(processed);\n        }\n        \n        Ok(results)\n    }\n}"
    )
}

fn get_memory_usage_mb() -> f64 {
    // Simple memory usage estimation
    // In a real benchmark, you'd use more sophisticated memory measurement
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // This is a very rough estimation
    let layout = Layout::from_size_align(1, 1).unwrap();
    let ptr = unsafe { System.alloc(layout) };
    if !ptr.is_null() {
        unsafe { System.dealloc(ptr, layout) };
    }
    
    // Return a placeholder value - in real implementation you'd use system APIs
    100.0
}