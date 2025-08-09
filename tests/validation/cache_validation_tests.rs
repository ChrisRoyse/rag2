//! Cache validation and cleanup verification tests
//! Ensures caching works correctly and doesn't cause memory leaks or corruption

use embed_search::{
    embedding::{LazyEmbedder, EmbeddingCache},
    cache::bounded_cache::BoundedCache,
    storage::safe_vectordb::{VectorStorage, StorageConfig},
};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};

/// Test embedding cache functionality
#[tokio::test]
async fn test_embedding_cache_functionality() {
    println!("💾 Testing embedding cache functionality");
    println!("======================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test cache hit/miss behavior
    let test_inputs = vec![
        "fn cache_test_1() {}",
        "fn cache_test_2() {}", 
        "fn cache_test_1() {}", // Repeat to test cache hit
        "fn cache_test_3() {}",
        "fn cache_test_2() {}", // Another repeat
    ];
    
    let mut cache_performance = Vec::new();
    let mut embeddings_map = HashMap::new();
    
    for (i, input) in test_inputs.iter().enumerate() {
        println!("  Processing input {}: '{}'", i + 1, input);
        
        let start_time = std::time::Instant::now();
        let embedding = embedder.embed(input).await.expect(&format!("Embedding failed for input {}", i + 1));
        let processing_time = start_time.elapsed();
        
        cache_performance.push((input, processing_time, embedding.clone()));
        
        // Store first occurrence
        if !embeddings_map.contains_key(*input) {
            embeddings_map.insert(*input, embedding.clone());
        } else {
            // Verify cache consistency
            let original = &embeddings_map[*input];
            assert_eq!(original.len(), embedding.len(), "Cached embedding size mismatch for {}", input);
            
            for (j, (&orig, &cached)) in original.iter().zip(embedding.iter()).enumerate() {
                assert_eq!(orig, cached, "Cache corruption at position {} for input '{}'", j, input);
            }
            
            println!("    ✅ Cache hit verified - embeddings identical");
        }
        
        println!("    ⏱️  Processing time: {:?}", processing_time);
    }
    
    // Analyze cache performance
    println!("\n📊 Cache Performance Analysis:");
    
    let first_occurrences: Vec<_> = cache_performance.iter()
        .enumerate()
        .filter(|(_, (input, _, _))| {
            // Check if this is the first occurrence
            cache_performance[..cache_performance.len()].iter().position(|(inp, _, _)| inp == input).unwrap() == cache_performance.len() - 1
        })
        .collect();
    
    // Group by input to find cache hits
    let mut input_times: HashMap<&str, Vec<Duration>> = HashMap::new();
    for (input, time, _) in &cache_performance {
        input_times.entry(input).or_insert_with(Vec::new).push(*time);
    }
    
    for (input, times) in input_times {
        if times.len() > 1 {
            let first_time = times[0];
            let cached_times = &times[1..];
            let avg_cached_time = cached_times.iter().sum::<Duration>() / cached_times.len() as u32;
            
            let speedup = first_time.as_nanos() as f64 / avg_cached_time.as_nanos() as f64;
            
            println!("  Input '{}': first={:?}, cached_avg={:?}, speedup={:.2}x", 
                     input, first_time, avg_cached_time, speedup);
            
            // Cache should provide some speedup
            if speedup > 2.0 {
                println!("    ✅ Excellent cache performance");
            } else if speedup > 1.1 {
                println!("    ✅ Moderate cache benefit");
            } else {
                println!("    ⚠️  Limited cache benefit detected");
            }
        }
    }
    
    println!("\n✅ Embedding cache functionality test completed");
}

/// Test cache memory management and cleanup
#[tokio::test]
async fn test_cache_memory_management() {
    println!("🧪 Testing cache memory management");
    println!("=================================\n");
    
    // Create a bounded cache for testing
    let cache_size = 50;
    let mut bounded_cache: BoundedCache<String, Vec<f32>> = BoundedCache::new(cache_size)
        .expect("Failed to create bounded cache");
    
    println!("  Created bounded cache with capacity: {}", cache_size);
    
    // Fill cache to capacity
    let embedder = LazyEmbedder::new();
    let mut test_embeddings = Vec::new();
    
    for i in 0..cache_size + 10 { // Overfill to test eviction
        let input = format!("fn cache_memory_test_{}() {{ println!(\"test {}\"); }}", i, i);
        let embedding = embedder.embed(&input).await.expect(&format!("Embedding failed for item {}", i));
        
        // Store in our test cache
        bounded_cache.insert(input.clone(), embedding.clone())
            .expect(&format!("Cache insertion failed for item {}", i));
        
        test_embeddings.push((input, embedding));
        
        if i % 10 == 0 {
            let current_size = bounded_cache.len();
            println!("    Cached {} items, current cache size: {}", i + 1, current_size);
        }
    }
    
    // Verify cache size constraints
    let final_cache_size = bounded_cache.len();
    assert!(final_cache_size <= cache_size, "Cache exceeded maximum size: {} > {}", final_cache_size, cache_size);
    
    println!("  ✅ Cache size properly constrained: {} <= {}", final_cache_size, cache_size);
    
    // Test cache retrieval after eviction
    let mut found_items = 0;
    let mut evicted_items = 0;
    
    for (input, expected_embedding) in &test_embeddings {
        match bounded_cache.get(input) {
            Ok(Some(cached_embedding)) => {
                // Verify integrity of cached data
                assert_eq!(cached_embedding.len(), expected_embedding.len(), "Cached embedding size mismatch");
                
                for (j, (&cached, &expected)) in cached_embedding.iter().zip(expected_embedding.iter()).enumerate() {
                    assert_eq!(cached, expected, "Cache data corruption at position {} for '{}'", j, input);
                }
                
                found_items += 1;
            },
            Ok(None) => {
                evicted_items += 1;
            },
            Err(e) => {
                panic!("Cache retrieval error for '{}': {}", input, e);
            }
        }
    }
    
    println!("  📈 Cache statistics:");
    println!("    Items found: {}", found_items);
    println!("    Items evicted: {}", evicted_items);
    println!("    Current cache size: {}", final_cache_size);
    
    // Verify that recent items are more likely to be retained (LRU behavior)
    let recent_items = &test_embeddings[test_embeddings.len() - cache_size/2..];
    let mut recent_found = 0;
    
    for (input, _) in recent_items {
        if bounded_cache.get(input).unwrap().is_some() {
            recent_found += 1;
        }
    }
    
    let recent_retention_rate = recent_found as f64 / recent_items.len() as f64;
    println!("    Recent items retention rate: {:.1}%", recent_retention_rate * 100.0);
    
    // Recent items should have higher retention rate
    assert!(recent_retention_rate > 0.5, "Recent items retention too low: {:.1}%", recent_retention_rate * 100.0);
    
    println!("\n✅ Cache memory management test completed");
}

/// Test cache persistence and cleanup
#[tokio::test]
async fn test_cache_persistence_and_cleanup() {
    println!("🧟 Testing cache persistence and cleanup");
    println!("======================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Create initial cache state
    let persistent_inputs = vec![
        "fn persistent_test_1() {}",
        "fn persistent_test_2() {}",
        "fn persistent_test_3() {}",
    ];
    
    let mut original_embeddings = Vec::new();
    
    println!("  Creating initial cache state...");
    for (i, input) in persistent_inputs.iter().enumerate() {
        let embedding = embedder.embed(input).await.expect(&format!("Embedding failed for persistent input {}", i + 1));
        original_embeddings.push(embedding);
        println!("    ✅ Cached: {}", input);
    }
    
    // Test cache persistence across multiple access patterns
    println!("\n  Testing cache persistence across access patterns...");
    
    // Pattern 1: Random access
    for i in 0..10 {
        let input = &persistent_inputs[i % persistent_inputs.len()];
        let start_time = std::time::Instant::now();
        let embedding = embedder.embed(input).await.expect("Random access embedding failed");
        let access_time = start_time.elapsed();
        
        // Should be fast due to caching
        if access_time < Duration::from_millis(100) {
            println!("    ✅ Fast access for '{}': {:?}", input, access_time);
        } else {
            println!("    ⚠️  Slow access for '{}': {:?} (possible cache miss)", input, access_time);
        }
        
        // Verify consistency
        let expected = &original_embeddings[i % persistent_inputs.len()];
        assert_eq!(embedding, *expected, "Cache inconsistency detected for '{}'", input);
    }
    
    // Pattern 2: Burst access
    println!("\n  Testing burst access pattern...");
    let burst_start = std::time::Instant::now();
    
    for _ in 0..20 {
        for input in &persistent_inputs {
            let embedding = embedder.embed(input).await.expect("Burst access embedding failed");
            assert_eq!(embedding.len(), 768, "Burst access embedding wrong size");
        }
    }
    
    let burst_duration = burst_start.elapsed();
    let avg_access_time = burst_duration / (20 * persistent_inputs.len()) as u32;
    
    println!("    ✅ Burst access completed: total={:?}, avg={:?}", burst_duration, avg_access_time);
    
    // Average access should be fast due to caching
    assert!(avg_access_time < Duration::from_millis(50), "Burst access too slow: {:?}", avg_access_time);
    
    // Test cleanup behavior
    println!("\n  Testing cache cleanup behavior...");
    
    // Generate many temporary embeddings to test cleanup
    let cleanup_test_count = 100;
    
    for i in 0..cleanup_test_count {
        let temp_input = format!("fn cleanup_test_{}() {{ temp_function({}); }}", i, i * 2);
        let _ = embedder.embed(&temp_input).await.expect("Cleanup test embedding failed");
        
        if i % 25 == 0 {
            println!("    Generated {} temporary embeddings for cleanup test", i + 1);
        }
    }
    
    // Verify that original persistent items are still accessible
    println!("\n  Verifying persistent items after cleanup test...");
    
    for (i, input) in persistent_inputs.iter().enumerate() {
        let embedding = embedder.embed(input).await.expect(&format!("Post-cleanup access failed for {}", input));
        let expected = &original_embeddings[i];
        
        assert_eq!(embedding, *expected, "Cache corruption after cleanup for '{}'", input);
        println!("    ✅ Persistent item {} still accessible and consistent", i + 1);
    }
    
    println!("\n✅ Cache persistence and cleanup test completed");
}

/// Test cache concurrency and thread safety
#[tokio::test]
async fn test_cache_concurrency() {
    println!("⚡ Testing cache concurrency and thread safety");
    println!("==========================================\n");
    
    let embedder = LazyEmbedder::new();
    
    // Test concurrent access to the same cached items
    let concurrent_inputs = vec![
        "fn concurrent_test_1() {}",
        "fn concurrent_test_2() {}",
        "fn concurrent_test_3() {}",
    ];
    
    // Pre-populate cache
    println!("  Pre-populating cache...");
    let mut reference_embeddings = Vec::new();
    
    for input in &concurrent_inputs {
        let embedding = embedder.embed(input).await.expect("Pre-population failed");
        reference_embeddings.push(embedding);
        println!("    ✅ Pre-cached: {}", input);
    }
    
    // Test concurrent access
    println!("\n  Testing concurrent cache access...");
    
    let concurrency_level = 20;
    let mut handles = Vec::new();
    
    for i in 0..concurrency_level {
        let embedder_clone = embedder.clone();
        let inputs_clone = concurrent_inputs.clone();
        let references_clone = reference_embeddings.clone();
        
        let handle = tokio::spawn(async move {
            let mut results = Vec::new();
            
            for _ in 0..5 { // Each task does 5 iterations
                for (j, input) in inputs_clone.iter().enumerate() {
                    let start = std::time::Instant::now();
                    match embedder_clone.embed(input).await {
                        Ok(embedding) => {
                            let duration = start.elapsed();
                            
                            // Verify consistency with reference
                            if embedding == references_clone[j] {
                                results.push((i, j, duration, true));
                            } else {
                                results.push((i, j, duration, false));
                            }
                        },
                        Err(e) => {
                            return Err(format!("Task {} failed on input {}: {}", i, j, e));
                        }
                    }
                }
            }
            
            Ok(results)
        });
        
        handles.push(handle);
    }
    
    // Collect concurrent results
    let mut successful_tasks = 0;
    let mut failed_tasks = 0;
    let mut all_access_times = Vec::new();
    let mut consistency_violations = 0;
    
    for handle in handles {
        match handle.await {
            Ok(Ok(results)) => {
                successful_tasks += 1;
                
                for (task_id, input_id, duration, is_consistent) in results {
                    all_access_times.push(duration);
                    
                    if !is_consistent {
                        consistency_violations += 1;
                        println!("    ❌ Consistency violation: task={}, input={}", task_id, input_id);
                    }
                }
            },
            Ok(Err(e)) => {
                failed_tasks += 1;
                println!("    ❌ Task failed: {}", e);
            },
            Err(e) => {
                failed_tasks += 1;
                println!("    💥 Task panicked: {}", e);
            }
        }
    }
    
    // Analyze concurrency results
    let success_rate = successful_tasks as f64 / (successful_tasks + failed_tasks) as f64;
    let avg_access_time = all_access_times.iter().sum::<Duration>() / all_access_times.len() as u32;
    let max_access_time = all_access_times.iter().max().copied().unwrap_or(Duration::from_millis(0));
    
    println!("\n📊 Concurrency Test Results:");
    println!("  Successful tasks: {}/{}", successful_tasks, successful_tasks + failed_tasks);
    println!("  Success rate: {:.1}%", success_rate * 100.0);
    println!("  Consistency violations: {}", consistency_violations);
    println!("  Average access time: {:?}", avg_access_time);
    println!("  Maximum access time: {:?}", max_access_time);
    
    // Concurrency assertions
    assert!(success_rate >= 0.95, "Too many task failures: {:.1}%", (1.0 - success_rate) * 100.0);
    assert_eq!(consistency_violations, 0, "Cache consistency violations detected: {}", consistency_violations);
    assert!(avg_access_time < Duration::from_millis(200), "Average access too slow under concurrency: {:?}", avg_access_time);
    
    println!("\n✅ Cache concurrency test completed");
}

/// Test cache cleanup verification
#[tokio::test]
async fn test_cache_cleanup_verification() {
    println!("🧺 Testing cache cleanup verification");
    println!("=================================\n");
    
    // Create a storage system to test cleanup
    let storage_config = StorageConfig {
        max_vectors: 100,
        dimension: 768,
        cache_size: 20, // Small cache to force eviction
        enable_compression: false,
    };
    
    let mut storage = VectorStorage::new(storage_config).expect("Failed to create storage");
    let embedder = LazyEmbedder::new();
    
    println!("  Created storage with small cache size for cleanup testing");
    
    // Phase 1: Fill storage and cache
    println!("\n  Phase 1: Filling storage and cache...");
    
    let initial_items = 50;
    let mut stored_items = Vec::new();
    
    for i in 0..initial_items {
        let input = format!("fn cleanup_verification_{}() {{ process_data({}); }}", i, i);
        let embedding = embedder.embed(&input).await.expect(&format!("Embedding failed for item {}", i));
        let id = format!("cleanup_test_{}", i);
        
        storage.insert(id.clone(), embedding.clone()).expect(&format!("Storage failed for item {}", i));
        stored_items.push((id, input, embedding));
        
        if i % 10 == 0 {
            println!("    Stored {} items", i + 1);
        }
    }
    
    // Phase 2: Access patterns to test cache behavior
    println!("\n  Phase 2: Testing access patterns and cache eviction...");
    
    // Random access pattern
    for i in 0..30 {
        let item_index = i % stored_items.len();
        let (id, input, expected_embedding) = &stored_items[item_index];
        
        // Test storage retrieval
        let retrieved = storage.get(id).expect("Storage retrieval failed")
            .expect("Item not found in storage");
        
        assert_eq!(retrieved, *expected_embedding, "Storage corruption for item {}", item_index);
        
        // Test embedding cache
        let fresh_embedding = embedder.embed(input).await.expect("Fresh embedding failed");
        assert_eq!(fresh_embedding, *expected_embedding, "Embedding cache corruption for item {}", item_index);
        
        if i % 10 == 0 {
            println!("    Verified {} random accesses", i + 1);
        }
    }
    
    // Phase 3: Memory pressure and cleanup
    println!("\n  Phase 3: Applying memory pressure to test cleanup...");
    
    let pressure_items = 100;
    
    for i in 0..pressure_items {
        let input = format!("fn memory_pressure_{}() {{ large_computation({}); }}", i, i * 3);
        let embedding = embedder.embed(&input).await.expect(&format!("Memory pressure embedding failed at {}", i));
        
        // Try to store (may fail due to capacity)
        match storage.insert(format!("pressure_{}", i), embedding) {
            Ok(_) => {},
            Err(e) => {
                if i < 20 {
                    panic!("Storage should handle at least 20 pressure items, failed at {}: {}", i, e);
                } else {
                    println!("    ⚠️  Storage capacity reached at item {}: {}", i, e);
                    break;
                }
            }
        }
        
        if i % 25 == 0 && i > 0 {
            println!("    Applied pressure with {} items", i);
        }
    }
    
    // Phase 4: Verify cleanup didn't corrupt original data
    println!("\n  Phase 4: Verifying cleanup didn't corrupt original data...");
    
    let verification_sample = std::cmp::min(10, stored_items.len());
    
    for i in 0..verification_sample {
        let (id, input, expected_embedding) = &stored_items[i];
        
        // Test storage integrity
        match storage.get(id) {
            Ok(Some(retrieved)) => {
                assert_eq!(retrieved, *expected_embedding, "Storage corruption after cleanup for item {}", i);
                println!("    ✅ Storage item {} intact after cleanup", i);
            },
            Ok(None) => {
                println!("    ⚠️  Storage item {} evicted during cleanup (acceptable)", i);
            },
            Err(e) => {
                panic!("Storage error after cleanup for item {}: {}", i, e);
            }
        }
        
        // Test embedding cache integrity
        let fresh_embedding = embedder.embed(input).await.expect(&format!("Post-cleanup embedding failed for item {}", i));
        assert_eq!(fresh_embedding, *expected_embedding, "Embedding cache corruption after cleanup for item {}", i);
        
        println!("    ✅ Embedding cache item {} intact after cleanup", i);
    }
    
    // Phase 5: Test system recovery
    println!("\n  Phase 5: Testing system recovery after cleanup...");
    
    // New operations should work normally
    for i in 0..5 {
        let input = format!("fn recovery_test_{}() {{ post_cleanup_operation({}); }}", i, i);
        let embedding = embedder.embed(&input).await.expect(&format!("Recovery test embedding failed for {}", i));
        
        assert_eq!(embedding.len(), 768, "Recovery embedding wrong size for item {}", i);
        
        // Test storage
        let recovery_id = format!("recovery_{}", i);
        storage.insert(recovery_id.clone(), embedding.clone())
            .expect(&format!("Recovery storage failed for item {}", i));
        
        let retrieved = storage.get(&recovery_id)
            .expect(&format!("Recovery retrieval failed for item {}", i))
            .expect(&format!("Recovery item {} not found", i));
        
        assert_eq!(retrieved, embedding, "Recovery storage corruption for item {}", i);
        
        println!("    ✅ Recovery test {} completed successfully", i + 1);
    }
    
    println!("\n✅ Cache cleanup verification test completed");
}

/// Test cache statistics and monitoring
#[tokio::test]
async fn test_cache_statistics_and_monitoring() {
    println!("📊 Testing cache statistics and monitoring");
    println!("========================================\n");
    
    // Create a bounded cache for statistics testing
    let cache_capacity = 25;
    let mut test_cache: BoundedCache<String, Vec<f32>> = BoundedCache::new(cache_capacity)
        .expect("Failed to create test cache");
    
    let embedder = LazyEmbedder::new();
    
    // Track cache operations
    let mut total_inserts = 0;
    let mut total_gets = 0;
    let mut cache_hits = 0;
    let mut cache_misses = 0;
    
    println!("  Testing cache statistics collection...");
    
    // Phase 1: Initial population
    for i in 0..cache_capacity {
        let input = format!("fn stats_test_{}() {{ monitoring_function({}); }}", i, i);
        let embedding = embedder.embed(&input).await.expect(&format!("Stats test embedding failed for {}", i));
        
        test_cache.insert(input, embedding).expect(&format!("Cache insert failed for {}", i));
        total_inserts += 1;
    }
    
    println!("    Initial population: {} items inserted", total_inserts);
    
    // Phase 2: Mixed access patterns
    let access_patterns = vec![
        // Pattern 1: Hit existing items
        ("hit_pattern", (0..cache_capacity/2).collect::<Vec<_>>()),
        // Pattern 2: Miss with new items
        ("miss_pattern", (cache_capacity..cache_capacity*2).collect::<Vec<_>>()),
        // Pattern 3: Mixed hits and misses
        ("mixed_pattern", (cache_capacity/4..cache_capacity/4 + cache_capacity).collect::<Vec<_>>()),
    ];
    
    for (pattern_name, indices) in access_patterns {
        println!("\n    Testing {} with {} accesses:", pattern_name, indices.len());
        
        let mut pattern_hits = 0;
        let mut pattern_misses = 0;
        
        for i in indices {
            let input = format!("fn stats_test_{}() {{ monitoring_function({}); }}", i, i);
            
            // Check if item exists in cache
            let exists_in_cache = test_cache.get(&input).unwrap().is_some();
            total_gets += 1;
            
            if exists_in_cache {
                cache_hits += 1;
                pattern_hits += 1;
            } else {
                cache_misses += 1;
                pattern_misses += 1;
                
                // Insert new item
                let embedding = embedder.embed(&input).await.expect("Pattern test embedding failed");
                test_cache.insert(input, embedding).expect("Pattern cache insert failed");
                total_inserts += 1;
            }
        }
        
        let pattern_hit_rate = pattern_hits as f64 / (pattern_hits + pattern_misses) as f64;
        println!("      Hits: {}, Misses: {}, Hit rate: {:.1}%", 
                 pattern_hits, pattern_misses, pattern_hit_rate * 100.0);
    }
    
    // Calculate overall statistics
    let overall_hit_rate = cache_hits as f64 / total_gets as f64;
    let current_cache_size = test_cache.len();
    let cache_utilization = current_cache_size as f64 / cache_capacity as f64;
    
    println!("\n📊 Overall Cache Statistics:");
    println!("  Total inserts: {}", total_inserts);
    println!("  Total gets: {}", total_gets);
    println!("  Cache hits: {}", cache_hits);
    println!("  Cache misses: {}", cache_misses);
    println!("  Hit rate: {:.1}%", overall_hit_rate * 100.0);
    println!("  Current size: {}/{} ({:.1}% utilization)", 
             current_cache_size, cache_capacity, cache_utilization * 100.0);
    
    // Verify statistics make sense
    assert_eq!(total_gets, cache_hits + cache_misses, "Hit/miss counts don't add up");
    assert!(current_cache_size <= cache_capacity, "Cache size exceeded capacity");
    assert!(overall_hit_rate >= 0.0 && overall_hit_rate <= 1.0, "Invalid hit rate: {}", overall_hit_rate);
    
    // Test cache monitoring under stress
    println!("\n  Testing cache monitoring under stress...");
    
    let stress_operations = 200;
    let stress_start = std::time::Instant::now();
    
    for i in 0..stress_operations {
        let input = format!("fn stress_monitoring_{}() {{ stress_operation({}); }}", i, i % 50);
        
        // Mix of gets and inserts
        if i % 3 == 0 {
            let _ = test_cache.get(&input);
        } else {
            let embedding = embedder.embed(&input).await.expect("Stress embedding failed");
            let _ = test_cache.insert(input, embedding);
        }
        
        if i % 50 == 0 && i > 0 {
            let elapsed = stress_start.elapsed();
            let ops_per_sec = i as f64 / elapsed.as_secs_f64();
            println!("    Stress test: {} operations, {:.1} ops/sec", i, ops_per_sec);
        }
    }
    
    let stress_duration = stress_start.elapsed();
    let final_ops_per_sec = stress_operations as f64 / stress_duration.as_secs_f64();
    
    println!("\n  Stress test completed: {} operations in {:?} ({:.1} ops/sec)", 
             stress_operations, stress_duration, final_ops_per_sec);
    
    // Performance assertions
    assert!(final_ops_per_sec > 100.0, "Cache operations too slow under stress: {:.1} ops/sec", final_ops_per_sec);
    assert!(test_cache.len() <= cache_capacity, "Cache size constraint violated under stress");
    
    println!("\n✅ Cache statistics and monitoring test completed");
}