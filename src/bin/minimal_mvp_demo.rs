use embed_search::minimal_mvp::MinimalRAG;
use std::io::{self, Write};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🚀 Minimal MVP RAG System Demo");
    println!("==============================");
    
    let mut rag = MinimalRAG::new();
    
    // Initialize with sample data
    println!("\n📚 Initializing with sample documents...");
    initialize_sample_data(&mut rag)?;
    
    // Show stats
    let stats = rag.get_stats();
    println!("\n📊 System Status:");
    println!("  • Documents: {}", stats.total_documents);
    println!("  • Fuzzy Search: {}", if stats.fuzzy_ready { "✅ Ready" } else { "❌ Not Ready" });
    println!("  • BM25 Search: {}", if stats.bm25_ready { "✅ Ready" } else { "❌ Not Ready" });
    
    // Interactive search loop
    println!("\n🔍 Interactive Search (type 'quit' to exit, 'test' to run automated test):");
    loop {
        print!("\nEnter search query: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let query = input.trim();
        
        if query.is_empty() {
            continue;
        }
        
        if query == "quit" {
            break;
        }
        
        if query == "test" {
            run_automated_test(&mut rag)?;
            continue;
        }
        
        // Perform search
        match rag.combined_search(query) {
            Ok(results) => {
                if results.is_empty() {
                    println!("  📭 No results found for '{}'", query);
                } else {
                    println!("  📋 Found {} results for '{}':", results.len(), query);
                    for (i, result) in results.iter().take(5).enumerate() {
                        println!("    {}. {} (score: {:.3}, type: {})", 
                            i + 1, result.title, result.score, result.search_type);
                        if result.content.len() > 80 {
                            println!("       {}", &result.content[..80].replace('\n', " "));
                        } else {
                            println!("       {}", result.content.replace('\n', " "));
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Search error: {}", e);
            }
        }
    }
    
    println!("\n👋 Goodbye!");
    Ok(())
}

fn initialize_sample_data(rag: &mut MinimalRAG) -> Result<()> {
    // Add realistic programming documents
    rag.add_document(
        "/src/database/connection.rs",
        "Database Connection",
        "use std::sync::Arc;\nuse tokio_postgres::{Client, NoTls};\n\npub struct DatabaseConnection {\n    client: Arc<Client>,\n}\n\nimpl DatabaseConnection {\n    pub async fn new(connection_string: &str) -> Result<Self, Box<dyn std::error::Error>> {\n        let (client, connection) = tokio_postgres::connect(connection_string, NoTls).await?;\n        Ok(Self { client: Arc::new(client) })\n    }\n\n    pub async fn execute_query(&self, query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {\n        // Implementation here\n        Ok(vec![])\n    }\n}"
    )?;
    
    rag.add_document(
        "/src/auth/user.rs", 
        "User Authentication",
        "use serde::{Deserialize, Serialize};\nuse uuid::Uuid;\n\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct User {\n    pub id: Uuid,\n    pub username: String,\n    pub email: String,\n    pub created_at: chrono::DateTime<chrono::Utc>,\n}\n\nimpl User {\n    pub fn new(username: String, email: String) -> Self {\n        Self {\n            id: Uuid::new_v4(),\n            username,\n            email,\n            created_at: chrono::Utc::now(),\n        }\n    }\n\n    pub fn authenticate(&self, password: &str) -> bool {\n        // Implementation here\n        true\n    }\n}"
    )?;
    
    rag.add_document(
        "/src/payment/processor.rs",
        "Payment Processing", 
        "use serde_json::Value;\nuse reqwest::Client;\n\npub struct PaymentProcessor {\n    client: Client,\n    api_key: String,\n    base_url: String,\n}\n\nimpl PaymentProcessor {\n    pub fn new(api_key: String) -> Self {\n        Self {\n            client: Client::new(),\n            api_key,\n            base_url: \"https://api.stripe.com/v1\".to_string(),\n        }\n    }\n\n    pub async fn process_payment(&self, amount: u64, currency: &str) -> Result<Value, Box<dyn std::error::Error>> {\n        let response = self.client\n            .post(&format!(\"{}/charges\", self.base_url))\n            .header(\"Authorization\", format!(\"Bearer {}\", self.api_key))\n            .json(&serde_json::json!({\n                \"amount\": amount,\n                \"currency\": currency\n            }))\n            .send()\n            .await?;\n        Ok(response.json().await?)\n    }\n}"
    )?;
    
    rag.add_document(
        "/src/config/settings.rs",
        "Configuration Management",
        "use serde::{Deserialize, Serialize};\nuse std::fs;\n\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct AppConfig {\n    pub database_url: String,\n    pub redis_url: String,\n    pub jwt_secret: String,\n    pub api_port: u16,\n}\n\nimpl AppConfig {\n    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {\n        let contents = fs::read_to_string(path)?;\n        let config: AppConfig = toml::from_str(&contents)?;\n        Ok(config)\n    }\n\n    pub fn validate(&self) -> Result<(), String> {\n        if self.database_url.is_empty() {\n            return Err(\"Database URL cannot be empty\".to_string());\n        }\n        if self.api_port == 0 {\n            return Err(\"API port must be specified\".to_string());\n        }\n        Ok(())\n    }\n}"
    )?;
    
    rag.add_document(
        "/docs/README.md",
        "Project Documentation",
        "# RAG System\n\nThis is a minimal Retrieval-Augmented Generation (RAG) system built in Rust.\n\n## Features\n\n- **Fuzzy Search**: Handle typos and approximate matches using Levenshtein distance\n- **BM25 Search**: Statistical ranking for text relevance\n- **File Watching**: Real-time monitoring of file changes\n- **In-Memory Storage**: Fast document storage without external dependencies\n\n## Components\n\n1. **WorkingFuzzySearch**: Implements Levenshtein distance for fuzzy matching\n2. **SimpleBM25**: Statistical text ranking algorithm\n3. **SimpleFileWatcher**: File system monitoring with debouncing\n4. **In-Memory Vector Storage**: Document storage without embeddings\n\n## Usage\n\n```rust\nuse embed_search::minimal_mvp::MinimalRAG;\n\nlet mut rag = MinimalRAG::new();\nrag.add_document(\"/path/file.rs\", \"Title\", \"Content...\");\nlet results = rag.combined_search(\"query\")?;\n```\n\n## Performance\n\n- Search latency: <100ms for datasets up to 10,000 documents\n- Memory usage: ~1MB per 1,000 documents\n- File watching: <200ms event processing latency\n"
    )?;
    
    println!("  ✅ Added {} sample documents", rag.get_stats().total_documents);
    Ok(())
}

fn run_automated_test(rag: &mut MinimalRAG) -> Result<()> {
    println!("\n🧪 Running Automated Test Suite...");
    
    // Test different search types
    let test_queries = vec![
        ("database", "Should find database-related content"),
        ("user", "Should find user authentication content"), 
        ("payment", "Should find payment processing content"),
        ("config", "Should find configuration content"),
        ("databse", "Should handle typo in 'database'"),  // Typo test
        ("authentica", "Should partially match 'authentication'"), // Partial test
        ("README", "Should find documentation"),
        ("struct", "Should find code structures"),
    ];
    
    for (query, description) in test_queries {
        print!("  Testing '{}': ", query);
        match rag.combined_search(query) {
            Ok(results) => {
                if results.is_empty() {
                    println!("❌ No results ({})", description);
                } else {
                    println!("✅ {} results", results.len());
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }
    
    // Performance test
    print!("  Performance test: ");
    let start = std::time::Instant::now();
    let _results = rag.combined_search("system architecture performance optimization")?;
    let duration = start.elapsed();
    
    if duration.as_millis() < 100 {
        println!("✅ {}ms (excellent)", duration.as_millis());
    } else if duration.as_millis() < 500 {
        println!("✅ {}ms (good)", duration.as_millis());
    } else {
        println!("⚠️ {}ms (slow but acceptable)", duration.as_millis());
    }
    
    println!("✅ Automated test completed");
    Ok(())
}