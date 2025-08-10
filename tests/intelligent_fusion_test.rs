use std::sync::Arc;
use tokio::sync::RwLock;
use embed_search::search::{BM25Searcher, BM25Document, BM25Token};
use embed_search::search::search_adapter::{UnifiedSearchAdapter, UnifiedMatch};

/// Test that the intelligent_fusion method compiles and can be called
#[tokio::test]
async fn test_intelligent_fusion_compilation() {
    // Create a simple BM25 searcher
    let mut bm25_searcher = BM25Searcher::new();
    
    // Add a test document
    let test_doc = BM25Document {
        id: "test_doc".to_string(),
        file_path: "test.rs".to_string(),
        chunk_index: 0,
        tokens: vec![
            BM25Token { term: "hello".to_string(), position: 0 },
            BM25Token { term: "world".to_string(), position: 1 },
        ],
        start_line: 1,
        end_line: 5,
        language: Some("rust".to_string()),
    };
    
    bm25_searcher.add_document(test_doc).unwrap();
    
    // Create the unified search adapter
    let adapter = UnifiedSearchAdapter::new(Arc::new(RwLock::new(bm25_searcher)));
    
    // Test the intelligent_fusion method
    let results = adapter.intelligent_fusion(
        "hello world",  // query
        10,             // max_results
        60.0,          // k parameter
        0.7            // alpha parameter (0.7 = 70% BM25, 30% semantic)
    ).await;
    
    // Verify it returns a result (even if empty due to no semantic results)
    assert!(results.is_ok());
    let fusion_results = results.unwrap();
    
    // Should have at least some results from BM25
    assert!(!fusion_results.is_empty());
    
    // Verify the result has the expected structure
    let first_result = &fusion_results[0];
    assert_eq!(first_result.doc_id, "test.rs");
    assert!(first_result.score > 0.0);
    assert!(first_result.match_type == "Statistical" || first_result.match_type == "Hybrid");
}

/// Test score normalization function behavior
#[tokio::test]
async fn test_score_normalization_behavior() {
    let bm25_searcher = BM25Searcher::new();
    let adapter = UnifiedSearchAdapter::new(Arc::new(RwLock::new(bm25_searcher)));
    
    // Test with empty results
    let empty_results = adapter.intelligent_fusion("nonexistent", 10, 60.0, 0.5).await.unwrap();
    assert!(empty_results.is_empty());
}

/// Test different alpha values for fusion weighting
#[tokio::test]
async fn test_fusion_weighting() {
    let mut bm25_searcher = BM25Searcher::new();
    
    let test_doc = BM25Document {
        id: "test_doc_2".to_string(),
        file_path: "test2.rs".to_string(),
        chunk_index: 0,
        tokens: vec![
            BM25Token { term: "function".to_string(), position: 0 },
            BM25Token { term: "test".to_string(), position: 1 },
        ],
        start_line: 1,
        end_line: 10,
        language: Some("rust".to_string()),
    };
    
    bm25_searcher.add_document(test_doc).unwrap();
    let adapter = UnifiedSearchAdapter::new(Arc::new(RwLock::new(bm25_searcher)));
    
    // Test with pure BM25 weighting (alpha = 1.0)
    let bm25_only = adapter.intelligent_fusion("function test", 10, 60.0, 1.0).await.unwrap();
    assert!(!bm25_only.is_empty());
    
    // Test with balanced weighting (alpha = 0.5)
    let balanced = adapter.intelligent_fusion("function test", 10, 60.0, 0.5).await.unwrap();
    assert!(!balanced.is_empty());
    
    // Both should return results, scores may differ
    assert_eq!(bm25_only[0].doc_id, balanced[0].doc_id);
}