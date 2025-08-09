//! Load tests for the new embedding system under stress
//! Designed to expose system limits and ensure stability under high load

use embed_search::{
    embedding::LazyEmbedder,
    storage::safe_vectordb::{VectorStorage, StorageConfig},
    search::unified::UnifiedSearcher,
};
use std::sync::Arc;
use std::time::{Instant, Duration};
use tokio::sync::Semaphore;
use tempfile::TempDir;

/// High-load concurrent embedding test
#[tokio::test]
async fn test_concurrent_embedding_load() {
    println!("⚡ Starting concurrent embedding load test");
    println!("==========================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Initialize the embedder first
    let _ = embedder.embed("initialization test").await.expect("Init failed");
    println!("✅ Embedder initialized\n");
    
    // Test different concurrency levels
    let concurrency_levels = vec![5, 10, 20, 50, 100];
    
    for concurrency in concurrency_levels {
        println!("🔄 Testing concurrency level: {}", concurrency);
        
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let embedder = Arc::new(embedder.clone());
        
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Spawn concurrent tasks
        for i in 0..concurrency * 2 { // 2x concurrency for more stress
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let embedder_clone = embedder.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = permit; // Keep permit alive
                
                let code = generate_stress_test_code(i);
                let start = Instant::now();
                
                let result = embedder_clone.embed(&code).await;
                let duration = start.elapsed();
                
                (i, result, duration)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut successful = 0;
        let mut failed = 0;
        let mut total_time = Duration::from_millis(0);
        let mut max_time = Duration::from_millis(0);
        
        for handle in handles {
            match handle.await {
                Ok((i, result, duration)) => {
                    match result {
                        Ok(embedding) => {
                            assert_eq!(embedding.len(), 768, "Wrong embedding size for task {}", i);
                            successful += 1;
                            total_time += duration;
                            max_time = max_time.max(duration);
                        },
                        Err(e) => {
                            println!("    ❌ Task {} failed: {}", i, e);
                            failed += 1;
                        }
                    }
                },
                Err(e) => {
                    println!("    💥 Task panicked: {}", e);
                    failed += 1;
                }
            }
        }
        
        let total_duration = start_time.elapsed();
        let avg_time = if successful > 0 {
            total_time / successful as u32
        } else {
            Duration::from_millis(0)
        };
        let throughput = successful as f64 / total_duration.as_secs_f64();
        
        println!("  📊 Results for concurrency {}:", concurrency);
        println!("    ✅ Successful: {}", successful);
        println!("    ❌ Failed: {}", failed);
        println!("    🏁 Total time: {:?}", total_duration);
        println!("    ⏱️  Average task time: {:?}", avg_time);
        println!("    🐌 Maximum task time: {:?}", max_time);
        println!("    🚀 Throughput: {:.1} embeddings/sec", throughput);
        
        // Performance assertions
        let success_rate = successful as f64 / (successful + failed) as f64;
        assert!(success_rate >= 0.80, "Success rate too low for concurrency {}: {:.1}%", concurrency, success_rate * 100.0);
        
        // Throughput should not completely collapse under load
        if concurrency <= 20 {
            assert!(throughput > 1.0, "Throughput too low for concurrency {}: {:.1} ops/sec", concurrency, throughput);
        }
        
        println!("  ✅ Concurrency {} passed requirements\n");
    }
    
    println!("✅ Concurrent embedding load test completed!");
}

/// Memory pressure test with large batches
#[tokio::test]
async fn test_memory_pressure_load() {
    println!("💾 Starting memory pressure load test");
    println!("==================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test different batch sizes to find memory limits
    let batch_sizes = vec![10, 25, 50, 100, 200];
    
    for batch_size in batch_sizes {
        println!("🔄 Testing batch size: {}", batch_size);
        
        let start_memory = get_memory_usage();
        let start_time = Instant::now();
        
        let mut batch_embeddings = Vec::new();
        let mut failed_at = None;
        
        for i in 0..batch_size {
            let code = generate_large_code_sample(i, 1000); // 1KB per sample
            
            match embedder.embed(&code).await {
                Ok(embedding) => {
                    assert_eq!(embedding.len(), 768, "Wrong embedding size in batch {} item {}", batch_size, i);
                    batch_embeddings.push(embedding);
                    
                    if i % 10 == 0 {
                        let current_memory = get_memory_usage();
                        let memory_growth = current_memory.saturating_sub(start_memory);
                        
                        if memory_growth > 500 * 1024 * 1024 { // 500MB growth limit
                            println!("    ⚠️  Memory growth limit reached at item {}: {} MB", i, memory_growth / (1024 * 1024));
                            failed_at = Some(i);
                            break;
                        }
                    }
                },
                Err(e) => {
                    println!("    ❌ Memory pressure failure at item {}: {}", i, e);
                    failed_at = Some(i);
                    break;
                }
            }
        }
        
        let duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        let memory_growth = end_memory.saturating_sub(start_memory);
        
        let successful_items = failed_at.unwrap_or(batch_size);
        
        println!("  📊 Results for batch size {}:", batch_size);
        println!("    ✅ Processed items: {}/{}", successful_items, batch_size);
        println!("    ⏱️  Total time: {:?}", duration);
        println!("    💾 Memory growth: {} MB", memory_growth / (1024 * 1024));
        println!("    🧮 Avg time per item: {:?}", duration / successful_items as u32);
        
        // Memory growth should be reasonable
        assert!(memory_growth < 1024 * 1024 * 1024, "Excessive memory growth: {} MB", memory_growth / (1024 * 1024));
        
        // Should be able to process at least some items
        assert!(successful_items >= std::cmp::min(10, batch_size), "Failed too early: processed {}/{}", successful_items, batch_size);
        
        println!("  ✅ Batch size {} within memory limits\n");
        
        // Force cleanup
        drop(batch_embeddings);
        tokio::task::yield_now().await;
    }
    
    println!("✅ Memory pressure load test completed!");
}

/// Storage system stress test
#[tokio::test]
async fn test_storage_stress_load() {
    println!("🗄️  Starting storage stress load test");
    println!("==================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test different storage configurations
    let storage_configs = vec![
        ("small", 1000, 50),
        ("medium", 5000, 100),
        ("large", 10000, 200),
    ];
    
    for (config_name, max_vectors, cache_size) in storage_configs {
        println!("🔄 Testing storage config: {} (max_vectors={}, cache_size={})", 
                 config_name, max_vectors, cache_size);
        
        let storage_config = StorageConfig {
            max_vectors,
            dimension: 768,
            cache_size,
            enable_compression: false,
        };
        
        let mut storage = VectorStorage::new(storage_config)
            .expect(&format!("Failed to create {} storage", config_name));
        
        let start_time = Instant::now();
        let test_items = std::cmp::min(max_vectors / 2, 1000); // Don't overwhelm
        
        // Generate and store embeddings
        let mut store_times = Vec::new();
        let mut retrieve_times = Vec::new();
        let mut stored_ids = Vec::new();
        
        println!("  📥 Storing {} embeddings...", test_items);
        
        for i in 0..test_items {
            let code = generate_stress_test_code(i);
            let embedding = embedder.embed(&code).await
                .expect(&format!("Embedding generation failed at item {}", i));
            
            let id = format!("stress_test_{}_{}", config_name, i);
            
            let store_start = Instant::now();
            match storage.insert(id.clone(), embedding) {
                Ok(_) => {
                    let store_time = store_start.elapsed();
                    store_times.push(store_time);
                    stored_ids.push(id);
                },
                Err(e) => {
                    println!("    ❌ Storage failed at item {}: {}", i, e);
                    break;
                }
            }
            
            if i % 100 == 0 && i > 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }
        
        println!();
        println!("  📤 Testing retrieval performance...");
        
        // Test retrieval performance
        let retrieval_sample_size = std::cmp::min(stored_ids.len(), 200);
        for i in 0..retrieval_sample_size {
            let id = &stored_ids[i * stored_ids.len() / retrieval_sample_size];
            
            let retrieve_start = Instant::now();
            match storage.get(id) {
                Ok(Some(embedding)) => {
                    let retrieve_time = retrieve_start.elapsed();
                    retrieve_times.push(retrieve_time);
                    assert_eq!(embedding.len(), 768, "Retrieved embedding wrong size");
                },
                Ok(None) => {
                    panic!("Stored item {} not found during retrieval", id);
                },
                Err(e) => {
                    println!("    ❌ Retrieval failed for {}: {}", id, e);
                    break;
                }
            }
        }
        
        // Test similarity search under load
        println!("  🔍 Testing similarity search performance...");
        
        let query_embedding = embedder.embed(&generate_stress_test_code(9999)).await
            .expect("Query embedding failed");
        
        let search_start = Instant::now();
        let similar_results = storage.find_similar(&query_embedding, 10);
        let search_time = search_start.elapsed();
        
        match similar_results {
            Ok(results) => {
                assert!(!results.is_empty(), "Similarity search returned no results");
                println!("    ✅ Similarity search found {} results in {:?}", results.len(), search_time);
            },
            Err(e) => {
                println!("    ❌ Similarity search failed: {}", e);
            }
        }
        
        let total_time = start_time.elapsed();
        
        // Calculate performance metrics
        let avg_store_time = if !store_times.is_empty() {
            store_times.iter().sum::<Duration>() / store_times.len() as u32
        } else {
            Duration::from_millis(0)
        };
        
        let avg_retrieve_time = if !retrieve_times.is_empty() {
            retrieve_times.iter().sum::<Duration>() / retrieve_times.len() as u32
        } else {
            Duration::from_millis(0)
        };
        
        let store_throughput = stored_ids.len() as f64 / total_time.as_secs_f64();
        
        println!("  📊 Storage performance for {}:", config_name);
        println!("    📥 Stored items: {}", stored_ids.len());
        println!("    📤 Retrieved items: {}", retrieve_times.len());
        println!("    ⏱️  Average store time: {:?}", avg_store_time);
        println!("    ⏱️  Average retrieve time: {:?}", avg_retrieve_time);
        println!("    🔍 Similarity search time: {:?}", search_time);
        println!("    🚀 Store throughput: {:.1} items/sec", store_throughput);
        
        // Performance assertions
        assert!(stored_ids.len() >= test_items * 90 / 100, "Too many storage failures for {}", config_name);
        assert!(avg_store_time.as_millis() < 1000, "Store time too slow for {}: {:?}", config_name, avg_store_time);
        assert!(avg_retrieve_time.as_millis() < 100, "Retrieve time too slow for {}: {:?}", config_name, avg_retrieve_time);
        assert!(search_time.as_secs() < 10, "Similarity search too slow for {}: {:?}", config_name, search_time);
        
        println!("  ✅ Storage config {} passed performance requirements\n");
    }
    
    println!("✅ Storage stress load test completed!");
}

/// End-to-end system stress test
#[tokio::test]
async fn test_end_to_end_system_stress() {
    println!("🌐 Starting end-to-end system stress test");
    println!("======================================\n");
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let temp_path = temp_dir.path();
    
    // Create a large test codebase
    println!("📁 Creating large test codebase...");
    let num_files = 50;
    
    for i in 0..num_files {
        let filename = format!("stress_test_file_{:03}.py", i);
        let content = generate_large_file_content(i, 2000); // 2KB per file
        let file_path = temp_path.join(filename);
        std::fs::write(file_path, content).expect(&format!("Failed to write file {}", i));
        
        if i % 10 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!();
    println!("✅ Created {} test files\n", num_files);
    
    // Initialize the system
    let project_path = temp_path.to_path_buf();
    let db_path = temp_path.join("stress_test.db");
    
    println!("🚀 Initializing unified searcher...");
    let searcher = UnifiedSearcher::new(project_path.clone(), db_path.clone()).await
        .expect("Failed to initialize searcher");
    
    // Index everything
    println!("📚 Indexing entire codebase...");
    let index_start = Instant::now();
    let stats = searcher.index_directory(&project_path).await
        .expect("Indexing failed");
    let index_time = index_start.elapsed();
    
    println!("✅ Indexing completed: {}", stats);
    println!("⏱️  Indexing time: {:?}\n", index_time);
    
    // Stress test searches
    println!("🔍 Running stress search tests...");
    
    let stress_queries = vec![
        "function definition",
        "class implementation", 
        "error handling",
        "data processing",
        "configuration setup",
        "async operation",
        "database connection",
        "user authentication",
        "logging functionality",
        "test validation",
    ];
    
    let concurrent_searches = 20;
    let mut search_handles = Vec::new();
    
    for i in 0..concurrent_searches {
        let query = stress_queries[i % stress_queries.len()].to_string();
        let searcher_clone = Arc::new(searcher.clone());
        
        let handle = tokio::spawn(async move {
            let mut search_results = Vec::new();
            let mut search_times = Vec::new();
            
            // Each task performs multiple searches
            for j in 0..5 {
                let modified_query = format!("{} test_{}", query, j);
                
                let search_start = Instant::now();
                match searcher_clone.search(&modified_query).await {
                    Ok(results) => {
                        let search_time = search_start.elapsed();
                        search_times.push(search_time);
                        search_results.push(results.len());
                    },
                    Err(e) => {
                        println!("    ❌ Search failed for '{}': {}", modified_query, e);
                        return (i, Err(e));
                    }
                }
            }
            
            (i, Ok((search_results, search_times)))
        });
        
        search_handles.push(handle);
    }
    
    // Collect concurrent search results
    let mut successful_searches = 0;
    let mut failed_searches = 0;
    let mut all_search_times = Vec::new();
    let mut total_results = 0;
    
    for handle in search_handles {
        match handle.await {
            Ok((task_id, result)) => {
                match result {
                    Ok((results, times)) => {
                        successful_searches += results.len();
                        total_results += results.iter().sum::<usize>();
                        all_search_times.extend(times);
                        println!("    ✅ Task {} completed {} searches", task_id, results.len());
                    },
                    Err(e) => {
                        failed_searches += 1;
                        println!("    ❌ Task {} failed: {}", task_id, e);
                    }
                }
            },
            Err(e) => {
                failed_searches += 1;
                println!("    💥 Task panicked: {}", e);
            }
        }
    }
    
    // Calculate performance metrics
    let avg_search_time = if !all_search_times.is_empty() {
        all_search_times.iter().sum::<Duration>() / all_search_times.len() as u32
    } else {
        Duration::from_millis(0)
    };
    
    let max_search_time = all_search_times.iter().max().copied().unwrap_or(Duration::from_millis(0));
    let search_throughput = successful_searches as f64 / index_time.as_secs_f64();
    
    println!("\n📊 End-to-end stress test results:");
    println!("  📁 Files indexed: {}", num_files);
    println!("  ⏱️  Indexing time: {:?}", index_time);
    println!("  ✅ Successful searches: {}", successful_searches);
    println!("  ❌ Failed searches: {}", failed_searches);
    println!("  📊 Total results found: {}", total_results);
    println!("  ⏱️  Average search time: {:?}", avg_search_time);
    println!("  🐌 Maximum search time: {:?}", max_search_time);
    println!("  🚀 Search throughput: {:.1} searches/sec", search_throughput);
    
    // System stress assertions
    let search_success_rate = successful_searches as f64 / (successful_searches + failed_searches) as f64;
    assert!(search_success_rate >= 0.90, "Search success rate too low: {:.1}%", search_success_rate * 100.0);
    assert!(index_time.as_secs() < 300, "Indexing took too long: {:?}", index_time);
    assert!(avg_search_time.as_secs() < 5, "Average search too slow: {:?}", avg_search_time);
    assert!(max_search_time.as_secs() < 15, "Maximum search too slow: {:?}", max_search_time);
    
    println!("\n✅ End-to-end system stress test passed!");
}

// Helper functions for generating test data

fn generate_stress_test_code(index: usize) -> String {
    let patterns = vec![
        "fn process_data_{}(input: &str) -> Result<String, Error> {{ Ok(input.to_uppercase()) }}",
        "class DataProcessor{}:\n    def __init__(self):\n        self.data = []\n    def process(self, item):\n        return item * 2",
        "async function handleRequest{}(req, res) {{ const result = await processData(req.body); res.json(result); }}",
        "public class Service{} {{ private final Repository repo; public Service{}(Repository r) {{ this.repo = r; }} }}",
        "struct Config{} {{ host: String, port: u16, timeout: Duration }}",
    ];
    
    let pattern = &patterns[index % patterns.len()];
    format!(pattern, index)
}

fn generate_large_code_sample(index: usize, target_size: usize) -> String {
    let mut code = format!("// Large code sample {}\n", index);
    code.push_str(&format!("class LargeClass{} {{\n", index));
    
    let method_template = "    def method_{}(self, param_{}):\n        # Processing data for {}\n        result = self.transform_{}(param_{})\n        return self.validate_result(result)\n\n";
    
    let mut current_size = code.len();
    let mut method_count = 0;
    
    while current_size < target_size {
        let method_code = format!(method_template, method_count, method_count, index, method_count, method_count);
        code.push_str(&method_code);
        current_size += method_code.len();
        method_count += 1;
    }
    
    code.push_str("}\n");
    code
}

fn generate_large_file_content(file_index: usize, target_size: usize) -> String {
    let mut content = format!("#!/usr/bin/env python3\n# Stress test file {}\n\nimport os\nimport sys\nimport json\nfrom typing import List, Dict, Optional\n\n", file_index);
    
    let class_template = "class TestClass{}:\n    \"\"\"Test class for stress testing the embedding system.\"\"\"\n    \n    def __init__(self, config: Dict):\n        self.config = config\n        self.data = []\n        self.processed = False\n    \n    def process_data(self, input_data: List) -> Dict:\n        \"\"\"Process input data and return results.\"\"\"\n        results = {{\n            'processed': len(input_data),\n            'timestamp': time.now(),\n            'file_index': {},\n            'class_index': {}\n        }}\n        return results\n    \n    def validate_results(self, results: Dict) -> bool:\n        \"\"\"Validate processing results.\"\"\"\n        return 'processed' in results and results['processed'] > 0\n\n";
    
    let mut current_size = content.len();
    let mut class_count = 0;
    
    while current_size < target_size {
        let class_code = format!(class_template, class_count, file_index, class_count);
        content.push_str(&class_code);
        current_size += class_code.len();
        class_count += 1;
        
        if current_size > target_size {
            break;
        }
    }
    
    content.push_str("\nif __name__ == '__main__':\n    print(f'Stress test file {} loaded')\n");
    content
}

fn get_memory_usage() -> usize {
    // Simple memory usage estimation
    // In production, you'd use more sophisticated memory measurement
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
    
    // Fallback: rough estimation
    1024 * 1024 * 100 // 100MB placeholder
}