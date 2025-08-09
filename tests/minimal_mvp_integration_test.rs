use embed_search::minimal_mvp::MinimalRAG;
use std::time::Duration;
use anyhow::Result;

#[test]
fn test_minimal_mvp_integration() -> Result<()> {
    println!("🚀 Running Minimal MVP Integration Test");
    
    let mut rag = MinimalRAG::new();
    
    // Test the complete MVP workflow
    rag.test_all_components()?;
    
    // Verify stats
    let stats = rag.get_stats();
    assert!(stats.total_documents > 0, "Should have indexed documents");
    assert!(stats.fuzzy_ready, "Fuzzy search should be ready");
    assert!(stats.bm25_ready, "BM25 search should be ready");
    
    println!("✅ Minimal MVP Integration Test PASSED");
    Ok(())
}

#[test] 
fn test_search_accuracy() -> Result<()> {
    println!("🔍 Testing Search Accuracy");
    
    let mut rag = MinimalRAG::new();
    
    // Add test documents with known content
    rag.add_document("/src/database.rs", "Database Module", 
        "pub struct DatabaseConnection { conn: Connection } impl DatabaseConnection { pub fn new() -> Self { ... } }")?;
        
    rag.add_document("/src/user.rs", "User Management", 
        "pub struct User { id: u32, name: String } impl User { pub fn create(name: &str) -> Self { ... } }")?;
        
    rag.add_document("/src/payment.rs", "Payment Processing", 
        "pub struct PaymentProcessor { api_key: String } impl PaymentProcessor { pub fn process_payment() { ... } }")?;
    
    // Test exact matches
    let db_results = rag.combined_search("database")?;
    assert!(!db_results.is_empty(), "Should find database-related documents");
    
    let user_results = rag.combined_search("user")?;
    assert!(!user_results.is_empty(), "Should find user-related documents");
    
    // Test fuzzy matches (typos)
    let typo_results = rag.fuzzy_search("databse")?; // Missing 'a'
    assert!(!typo_results.is_empty(), "Should handle typos in fuzzy search");
    
    // Test BM25 scoring
    let bm25_results = rag.bm25_search("struct")?;
    assert!(!bm25_results.is_empty(), "BM25 should find struct keyword");
    
    println!("✅ Search Accuracy Test PASSED - All search methods working");
    Ok(())
}

#[test]
fn test_performance_requirements() -> Result<()> {
    println!("⚡ Testing Performance Requirements");
    
    let mut rag = MinimalRAG::new();
    
    // Add 100 documents to test performance
    for i in 0..100 {
        rag.add_document(
            &format!("/docs/doc_{}.rs", i),
            &format!("Document {}", i),
            &format!("This is document {} with content about various topics like database, user management, configuration, and payment processing", i)
        )?;
    }
    
    let stats = rag.get_stats();
    assert_eq!(stats.total_documents, 100, "Should have indexed 100 documents");
    
    // Test search performance
    let start = std::time::Instant::now();
    let results = rag.combined_search("database")?;
    let search_duration = start.elapsed();
    
    assert!(!results.is_empty(), "Should find results in large dataset");
    assert!(search_duration.as_millis() < 500, "Search should complete in under 500ms with 100 docs");
    
    println!("✅ Performance Test PASSED - Search completed in {}ms", search_duration.as_millis());
    Ok(())
}

#[test] 
fn test_error_handling() -> Result<()> {
    println!("🛡️ Testing Error Handling");
    
    let mut rag = MinimalRAG::new();
    
    // Test empty query
    let empty_results = rag.combined_search("")?;
    assert!(empty_results.is_empty(), "Empty query should return empty results");
    
    // File watcher was removed from MVP to keep it minimal
    // Testing error resilience instead
    let empty_results = rag.combined_search("")?;
    assert!(empty_results.is_empty(), "Empty query should return empty results");
    
    // Test search with no documents
    let no_doc_rag = MinimalRAG::new();
    let no_results = no_doc_rag.combined_search("anything")?;
    assert!(no_results.is_empty(), "Search with no documents should return empty results");
    
    println!("✅ Error Handling Test PASSED");
    Ok(())
}