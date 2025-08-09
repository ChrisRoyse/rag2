//! MINIMAL WORKING TEST - Test only what actually compiles
//! 
//! This test validates the few components that are actually functional.

use embed_search::search::bm25::{BM25Engine, BM25Document, Token};
use embed_search::cache::bounded_cache::BoundedCache;

#[test]
fn test_bm25_basic_functionality() {
    println!("🔍 Testing BM25 Basic Functionality...");
    
    let mut engine = BM25Engine::new();
    
    // Add a simple document
    let tokens = vec![
        Token {
            text: "hello".to_string(),
            position: 0,
            importance_weight: 1.0,
        },
        Token {
            text: "world".to_string(),
            position: 1,
            importance_weight: 1.0,
        },
    ];
    
    let doc = BM25Document {
        id: "test_doc".to_string(),
        file_path: "test.txt".to_string(),
        chunk_index: 0,
        tokens,
        start_line: 1,
        end_line: 1,
        language: Some("text".to_string()),
    };
    
    // Test document addition
    assert!(engine.add_document(doc).is_ok(), "Failed to add document to BM25 engine");
    
    // Test search
    match engine.search("hello", 5) {
        Ok(results) => {
            assert!(!results.is_empty(), "BM25 search returned no results for 'hello'");
            assert_eq!(results[0].doc_id, "test_doc", "BM25 search returned wrong document");
            println!("✅ BM25 search works - found {} results", results.len());
        }
        Err(e) => {
            panic!("BM25 search failed: {}", e);
        }
    }
    
    println!("✅ BM25 basic functionality test passed");
}

#[test]
fn test_bounded_cache_functionality() {
    println!("🔍 Testing BoundedCache Functionality...");
    
    let mut cache: BoundedCache<String, i32> = BoundedCache::new(3)
        .expect("Failed to create bounded cache");
    
    // Test cache insertion
    cache.put("key1".to_string(), 100);
    cache.put("key2".to_string(), 200);
    cache.put("key3".to_string(), 300);
    
    // Test cache retrieval - fix API usage
    let key1_str = "key1".to_string();
    let key2_str = "key2".to_string();
    let key3_str = "key3".to_string();
    
    assert_eq!(cache.get(&key1_str), Some(100), "Cache failed to retrieve key1");
    assert_eq!(cache.get(&key2_str), Some(200), "Cache failed to retrieve key2");
    assert_eq!(cache.get(&key3_str), Some(300), "Cache failed to retrieve key3");
    
    // Test cache eviction (add 4th item to 3-capacity cache)
    cache.put("key4".to_string(), 400);
    
    // One of the original keys should be evicted
    let key4_str = "key4".to_string();
    let key1_present = cache.get(&key1_str).is_some();
    let key2_present = cache.get(&key2_str).is_some();
    let key3_present = cache.get(&key3_str).is_some();
    let key4_present = cache.get(&key4_str).is_some();
    
    assert!(key4_present, "Newly inserted key4 should be present");
    
    // Should have exactly 3 items
    let present_count = [key1_present, key2_present, key3_present].iter()
        .filter(|&&x| x).count();
    assert_eq!(present_count, 2, "Cache should have exactly 2 old items + 1 new = 3 total");
    
    println!("✅ BoundedCache functionality test passed");
}

#[test]
fn test_performance_benchmark_minimal() {
    use std::time::Instant;
    
    println!("🔍 Testing Minimal Performance Benchmark...");
    
    let start = Instant::now();
    let mut engine = BM25Engine::new();
    let mut operations = 0;
    
    // Add 10 small documents
    for i in 0..10 {
        let tokens = vec![
            Token {
                text: format!("word{}", i),
                position: 0,
                importance_weight: 1.0,
            },
            Token {
                text: "common".to_string(),
                position: 1,
                importance_weight: 1.0,
            },
        ];
        
        let doc = BM25Document {
            id: format!("doc{}", i),
            file_path: format!("file{}.txt", i),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: 1,
            language: Some("text".to_string()),
        };
        
        if engine.add_document(doc).is_ok() {
            operations += 1;
        }
    }
    
    // Perform 10 searches
    for i in 0..10 {
        let query = format!("word{}", i % 5);
        if engine.search(&query, 3).is_ok() {
            operations += 1;
        }
    }
    
    let duration = start.elapsed();
    let ops_per_second = operations as f64 / duration.as_secs_f64();
    
    println!("📊 Performance Results:");
    println!("  Operations: {}", operations);
    println!("  Duration: {:.3}s", duration.as_secs_f64());
    println!("  Ops/Second: {:.0}", ops_per_second);
    
    // Very basic performance assertion
    assert!(operations >= 15, "Should have completed at least 15 operations");
    assert!(ops_per_second >= 100.0, "Should achieve at least 100 operations per second");
    
    println!("✅ Minimal performance benchmark passed");
}

#[test] 
fn test_system_compilation_works() {
    println!("🔍 Testing that system actually compiles...");
    
    // If we get here, compilation worked
    println!("✅ System compilation test passed - code actually builds!");
}

/// Integration test runner
#[test]
fn run_all_working_tests() {
    println!("\n🚨 RUNNING ALL WORKING TESTS 🚨");
    println!("================================");
    
    test_system_compilation_works();
    test_bm25_basic_functionality();
    test_bounded_cache_functionality();
    test_performance_benchmark_minimal();
    
    println!("\n🎯 MINIMAL TEST SUITE RESULTS:");
    println!("✅ System compiles");
    println!("✅ BM25 engine works");
    println!("✅ BoundedCache works");
    println!("✅ Basic performance acceptable");
    
    println!("\n💡 WHAT THIS PROVES:");
    println!("- Core search functionality exists");
    println!("- Basic caching works");
    println!("- Performance is measurable");
    println!("- System has some working components");
    
    println!("\n⚠️ WHAT THIS DOESN'T PROVE:");
    println!("- Full-text search quality");
    println!("- ML embeddings functionality"); 
    println!("- Real-world performance");
    println!("- Production readiness");
    
    println!("\n🚨 CONCLUSION: MINIMAL FUNCTIONALITY EXISTS");
    println!("But system needs significant work for production use.");
}