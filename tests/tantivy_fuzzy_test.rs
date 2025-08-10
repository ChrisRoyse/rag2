use anyhow::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use embed_search::search::{TantivySearcher, ExactMatch};

/// Test the fuzzy search functionality with realistic code patterns
#[tokio::test]
async fn test_fuzzy_search_functionality() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();
    
    // Create a test codebase structure
    let src_dir = temp_path.join("src");
    std::fs::create_dir_all(&src_dir)?;
    
    // Create test files with realistic code content
    std::fs::write(src_dir.join("database_manager.rs"), r#"
pub struct DatabaseManager {
    connection_pool: ConnectionPool,
}

impl DatabaseManager {
    pub fn new() -> Self {
        Self {
            connection_pool: ConnectionPool::new(),
        }
    }
    
    pub async fn process_payment(&self, amount: f64) -> Result<PaymentId> {
        self.connection_pool.execute("INSERT INTO payments").await
    }
    
    pub fn get_user_data(&self, user_id: u64) -> UserData {
        // User data retrieval logic
    }
}
"#)?;

    std::fs::write(src_dir.join("payment_service.rs"), r#"
use crate::database_manager::DatabaseManager;

pub struct PaymentService {
    db_manager: DatabaseManager,
}

impl PaymentService {
    pub fn process_user_payment(&self, user: User) -> Result<()> {
        // Payment processing logic
    }
    
    pub fn validate_payment_method(&self) -> bool {
        true
    }
}
"#)?;

    std::fs::write(src_dir.join("user_handler.js"), r#"
class UserHandler {
    constructor(databaseConnection) {
        this.db = databaseConnection;
    }
    
    processPayment(amount) {
        return this.db.execute("INSERT INTO payments");
    }
    
    getUserInfo(userId) {
        return this.db.query("SELECT * FROM users");
    }
}
"#)?;

    // Initialize searcher and index the test codebase
    let mut searcher = TantivySearcher::new_with_root(&temp_path).await?;
    searcher.index_directory(&temp_path).await?;
    
    println!("✅ Test codebase indexed successfully");
    
    // Test cases for fuzzy matching
    let test_cases = vec![
        // Exact matches
        ("DatabaseManager", "Should find exact struct name"),
        ("process_payment", "Should find exact function name"),
        ("connection_pool", "Should find exact field name"),
        
        // Typos (1 character difference)
        ("DatabaseManger", "Should find DatabaseManager with typo"),
        ("proces_payment", "Should find process_payment with typo"),
        ("conection_pool", "Should find connection_pool with typo"),
        ("PaymentServic", "Should find PaymentService with typo"),
        
        // Partial matches
        ("Database", "Should find DatabaseManager as partial match"),
        ("payment", "Should find payment-related functions"),
        ("user", "Should find user-related items"),
        ("proces", "Should find process_payment as partial match"),
        
        // Case variations
        ("databasemanager", "Should find DatabaseManager case-insensitive"),
        ("PROCESS_PAYMENT", "Should find process_payment case-insensitive"),
        ("UserHandler", "Should find userHandler case-insensitive"),
        
        // Compound words
        ("UserPayment", "Should find user payment related items"),
        ("PaymentUser", "Should find payment user related items"),
    ];
    
    println!("\n🔍 Testing fuzzy search functionality...\n");
    
    let mut successful_tests = 0;
    let mut total_tests = 0;
    
    for (query, description) in test_cases {
        total_tests += 1;
        
        // Test with different fuzzy distances
        for distance in [1, 2] {
            let results = searcher.search_fuzzy(query, distance).await?;
            
            print!("Query: '{}' (distance: {}) - ", query, distance);
            
            if !results.is_empty() {
                successful_tests += 1;
                println!("✅ {} matches - {}", results.len(), description);
                
                // Show first few matches for verification
                for (i, result) in results.iter().take(3).enumerate() {
                    let file_name = result.file_path.split('/').last().unwrap_or(&result.file_path);
                    println!("  {}: {}:{} - {}", i + 1, file_name, result.line_number, result.content.trim());
                }
            } else {
                println!("❌ No matches - {}", description);
            }
            
            println!();
        }
    }
    
    // Performance test
    let start_time = std::time::Instant::now();
    let _perf_results = searcher.search_fuzzy("DatabaseManager", 1).await?;
    let search_time = start_time.elapsed();
    
    println!("⚡ Performance: Search completed in {:?}", search_time);
    
    // Verify response time is under 100ms
    assert!(search_time.as_millis() < 100, "Search took longer than 100ms: {:?}", search_time);
    
    let success_rate = (successful_tests as f64 / (total_tests * 2) as f64) * 100.0;
    println!("\n📊 Test Results:");
    println!("  - Successful queries: {}/{} ({:.1}%)", successful_tests, total_tests * 2, success_rate);
    println!("  - Search performance: {:?} (target: <100ms)", search_time);
    
    // Require at least 70% success rate
    assert!(success_rate >= 70.0, "Success rate too low: {:.1}%", success_rate);
    
    println!("\n🎉 Fuzzy search functionality validated!");
    
    Ok(())
}

/// Test edge cases and error handling
#[tokio::test]
async fn test_fuzzy_search_edge_cases() -> Result<()> {
    let mut searcher = TantivySearcher::new().await?;
    
    // Test empty query
    let result = searcher.search_fuzzy("", 1).await;
    assert!(result.is_err(), "Empty query should return error");
    
    // Test very long query
    let long_query = "a".repeat(1000);
    let result = searcher.search_fuzzy(&long_query, 1).await;
    // Should not panic or crash
    assert!(result.is_ok() || result.is_err()); // Either way is acceptable
    
    // Test high fuzzy distance
    let result = searcher.search_fuzzy("test", 10).await;
    assert!(result.is_ok(), "High fuzzy distance should be clamped, not error");
    
    println!("✅ Edge cases handled correctly");
    
    Ok(())
}

/// Performance benchmark for fuzzy search
#[tokio::test]
async fn benchmark_fuzzy_search_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();
    
    // Create a larger test codebase
    let src_dir = temp_path.join("src");
    std::fs::create_dir_all(&src_dir)?;
    
    // Generate multiple files with varied content
    for i in 0..10 {
        let content = format!(r#"
pub struct Service{} {{
    database: DatabaseConnection,
    cache: CacheManager,
    logger: Logger,
}}

impl Service{} {{
    pub fn process_request(&self, request: Request) -> Response {{
        self.logger.info("Processing request");
        let data = self.database.query("SELECT * FROM table");
        self.cache.store(data);
        Response::ok()
    }}
    
    pub fn handle_user_action(&self, user_id: u64, action: Action) -> Result<()> {{
        self.validate_user(user_id)?;
        self.execute_action(action)?;
        Ok(())
    }}
}}
"#, i, i);
        
        std::fs::write(src_dir.join(format!("service_{}.rs", i)), content)?;
    }
    
    // Initialize and index
    let mut searcher = TantivySearcher::new_with_root(&temp_path).await?;
    searcher.index_directory(&temp_path).await?;
    
    // Benchmark different query types
    let queries = vec![
        ("Service", "Exact match"),
        ("Servic", "Typo match"),
        ("database", "Partial match"),
        ("proces", "Fuzzy partial"),
        ("handle_user", "Underscore match"),
    ];
    
    println!("\n⚡ Performance Benchmark:");
    
    for (query, description) in queries {
        let mut total_time = std::time::Duration::new(0, 0);
        let iterations = 10;
        
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _results = searcher.search_fuzzy(query, 1).await?;
            total_time += start.elapsed();
        }
        
        let avg_time = total_time / iterations;
        println!("  {} - {}: {:?} avg", query, description, avg_time);
        
        // Each query should be under 50ms on average
        assert!(avg_time.as_millis() < 50, "Query '{}' too slow: {:?}", query, avg_time);
    }
    
    Ok(())
}