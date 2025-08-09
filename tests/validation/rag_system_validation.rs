//! Functional validation tests for the complete RAG system end-to-end
//! Tests the full pipeline: indexing → embedding → storage → retrieval → ranking

use embed_search::{
    search::unified::UnifiedSearcher,
    embedding::LazyEmbedder,
    config::Config,
    storage::safe_vectordb::{VectorStorage, StorageConfig},
};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

/// Comprehensive RAG system validation test
#[tokio::test]
async fn test_complete_rag_system() {
    println!("🧪 Starting comprehensive RAG system validation");
    println!("================================================\n");
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let temp_path = temp_dir.path();
    
    // Create comprehensive test codebase
    create_test_codebase(temp_path).await;
    
    // Initialize RAG system
    let project_path = temp_path.to_path_buf();
    let db_path = temp_path.join("rag_test.db");
    
    let searcher = timeout(
        Duration::from_secs(120),
        UnifiedSearcher::new(project_path.clone(), db_path.clone())
    ).await.expect("RAG initialization timed out")
        .expect("Failed to initialize RAG system");
    
    println!("✅ RAG system initialized\n");
    
    // Phase 1: Indexing validation
    validate_indexing_phase(&searcher, &project_path).await;
    
    // Phase 2: Embedding validation
    validate_embedding_phase(&searcher).await;
    
    // Phase 3: Storage validation
    validate_storage_phase(&searcher).await;
    
    // Phase 4: Retrieval validation
    validate_retrieval_phase(&searcher).await;
    
    // Phase 5: Ranking validation
    validate_ranking_phase(&searcher).await;
    
    // Phase 6: End-to-end workflow validation
    validate_end_to_end_workflows(&searcher).await;
    
    println!("\n🎉 Complete RAG system validation passed!");
}

/// Test indexing phase of RAG system
async fn validate_indexing_phase(searcher: &UnifiedSearcher, project_path: &PathBuf) {
    println!("📚 Phase 1: Indexing Validation");
    println!("===============================");
    
    // Index the entire test codebase
    let start = std::time::Instant::now();
    let stats = timeout(
        Duration::from_secs(300), // 5 minutes for comprehensive indexing
        searcher.index_directory(project_path)
    ).await.expect("Indexing timed out")
        .expect("Indexing failed");
    
    let indexing_time = start.elapsed();
    
    println!("  ✅ Indexing completed: {}", stats);
    println!("  ⏱️  Indexing time: {:?}", indexing_time);
    
    // Validate indexing results
    let system_stats = searcher.get_stats().await.expect("Failed to get stats");
    
    assert!(system_stats.total_embeddings > 0, "No embeddings created during indexing");
    assert!(system_stats.total_embeddings >= 20, "Too few embeddings created: {}", system_stats.total_embeddings);
    
    println!("  📊 Indexed embeddings: {}", system_stats.total_embeddings);
    println!("  💾 Cache utilization: {}/{}", system_stats.cache_entries, system_stats.cache_max_size);
    
    // Performance assertions
    assert!(indexing_time.as_secs() < 300, "Indexing took too long: {:?}", indexing_time);
    
    println!("  ✅ Indexing validation passed\n");
}

/// Test embedding generation phase
async fn validate_embedding_phase(searcher: &UnifiedSearcher) {
    println!("🧠 Phase 2: Embedding Validation");
    println!("===============================");
    
    let embedder = LazyEmbedder::new();
    
    // Test embedding quality for different code types
    let test_samples = vec![
        ("Rust function", "fn calculate_hash(data: &[u8]) -> u64 { data.iter().fold(0, |acc, &byte| acc.wrapping_mul(31).wrapping_add(byte as u64)) }"),
        ("Python class", "class DatabaseConnection:\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port\n    async def connect(self):\n        return await asyncpg.connect(f'postgresql://{self.host}:{self.port}')"),
        ("JavaScript module", "export class EventEmitter {\n  constructor() {\n    this.listeners = new Map();\n  }\n  on(event, callback) {\n    if (!this.listeners.has(event)) {\n      this.listeners.set(event, []);\n    }\n    this.listeners.get(event).push(callback);\n  }\n}"),
        ("SQL query", "SELECT u.id, u.username, p.email, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN profiles p ON u.id = p.user_id\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > NOW() - INTERVAL '30 days'\nGROUP BY u.id, u.username, p.email\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;"),
        ("Configuration file", "[database]\nhost = \"localhost\"\nport = 5432\nname = \"production_db\"\nmax_connections = 100\n\n[cache]\nredis_url = \"redis://localhost:6379\"\nttl = 3600\n"),
    ];
    
    let mut embedding_times = Vec::new();
    let mut embedding_quality_scores = Vec::new();
    
    for (sample_name, code) in test_samples {
        println!("  Testing embedding for: {}", sample_name);
        
        let start = std::time::Instant::now();
        let embedding = embedder.embed(code).await.expect(&format!("Failed to embed {}", sample_name));
        let embed_time = start.elapsed();
        
        embedding_times.push(embed_time);
        
        // Validate embedding properties
        assert_eq!(embedding.len(), 768, "Wrong embedding dimension for {}", sample_name);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding for {} not normalized: {}", sample_name, norm);
        
        // Check for reasonable variance (not all zeros or constant)
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
        
        assert!(variance > 0.001, "Embedding for {} has too low variance: {}", sample_name, variance);
        assert!(variance < 1.0, "Embedding for {} has too high variance: {}", sample_name, variance);
        
        let quality_score = calculate_embedding_quality_score(&embedding);
        embedding_quality_scores.push(quality_score);
        
        println!("    ✅ {}: {:?}, quality={:.2}", sample_name, embed_time, quality_score);
    }
    
    // Aggregate embedding performance metrics
    let avg_time: std::time::Duration = embedding_times.iter().sum::<std::time::Duration>() / embedding_times.len() as u32;
    let max_time = *embedding_times.iter().max().unwrap();
    let avg_quality = embedding_quality_scores.iter().sum::<f32>() / embedding_quality_scores.len() as f32;
    
    println!("\n  📊 Embedding Performance Summary:");
    println!("    Average time: {:?}", avg_time);
    println!("    Maximum time: {:?}", max_time);
    println!("    Average quality score: {:.2}", avg_quality);
    
    // Performance assertions
    assert!(avg_time.as_secs() < 10, "Average embedding time too slow: {:?}", avg_time);
    assert!(max_time.as_secs() < 30, "Maximum embedding time too slow: {:?}", max_time);
    assert!(avg_quality > 0.7, "Average embedding quality too low: {:.2}", avg_quality);
    
    println!("  ✅ Embedding validation passed\n");
}

/// Test vector storage phase
async fn validate_storage_phase(searcher: &UnifiedSearcher) {
    println!("🗄️  Phase 3: Storage Validation");
    println!("=============================");
    
    // Test direct storage operations
    let storage_config = StorageConfig {
        max_vectors: 1000,
        dimension: 768,
        cache_size: 100,
        enable_compression: false,
    };
    
    let mut storage = VectorStorage::new(storage_config).expect("Failed to create test storage");
    let embedder = LazyEmbedder::new();
    
    // Test storage with various types of embeddings
    let storage_test_cases = vec![
        ("short_code", "fn f() {}"),
        ("medium_code", "pub struct Config { pub host: String, pub port: u16 }"),
        ("long_code", create_long_test_code()),
        ("text_content", "This is a documentation string for testing storage."),
        ("mixed_content", "The function `process()` returns a `Result<T, E>` type."),
    ];
    
    println!("  Testing storage operations...");
    
    let mut stored_ids = Vec::new();
    let mut storage_times = Vec::new();
    
    // Test storage insertion
    for (id, content) in &storage_test_cases {
        let embedding = embedder.embed(content).await.expect(&format!("Failed to embed {}", id));
        
        let start = std::time::Instant::now();
        storage.insert(id.to_string(), embedding.clone()).expect(&format!("Failed to store {}", id));
        let store_time = start.elapsed();
        
        storage_times.push(store_time);
        stored_ids.push(id.to_string());
        
        println!("    ✅ Stored {}: {:?}", id, store_time);
    }
    
    // Test storage retrieval
    let mut retrieval_times = Vec::new();
    
    for id in &stored_ids {
        let start = std::time::Instant::now();
        let retrieved = storage.get(id).expect(&format!("Failed to retrieve {}", id))
            .expect(&format!("Item {} not found", id));
        let retrieve_time = start.elapsed();
        
        retrieval_times.push(retrieve_time);
        
        assert_eq!(retrieved.len(), 768, "Retrieved embedding for {} wrong size", id);
        println!("    ✅ Retrieved {}: {:?}", id, retrieve_time);
    }
    
    // Test similarity search
    let query_embedding = embedder.embed("fn search_test() {}").await.expect("Query embedding failed");
    
    let start = std::time::Instant::now();
    let similar = storage.find_similar(&query_embedding, 3).expect("Similarity search failed");
    let search_time = start.elapsed();
    
    assert!(!similar.is_empty(), "Similarity search returned no results");
    println!("    ✅ Similarity search: {:?}, found {} results", search_time, similar.len());
    
    // Performance metrics
    let avg_storage_time = storage_times.iter().sum::<std::time::Duration>() / storage_times.len() as u32;
    let avg_retrieval_time = retrieval_times.iter().sum::<std::time::Duration>() / retrieval_times.len() as u32;
    
    println!("\n  📊 Storage Performance Summary:");
    println!("    Average storage time: {:?}", avg_storage_time);
    println!("    Average retrieval time: {:?}", avg_retrieval_time);
    println!("    Similarity search time: {:?}", search_time);
    
    // Performance assertions
    assert!(avg_storage_time.as_millis() < 1000, "Storage too slow: {:?}", avg_storage_time);
    assert!(avg_retrieval_time.as_millis() < 100, "Retrieval too slow: {:?}", avg_retrieval_time);
    assert!(search_time.as_millis() < 5000, "Similarity search too slow: {:?}", search_time);
    
    println!("  ✅ Storage validation passed\n");
}

/// Test retrieval and ranking phase
async fn validate_retrieval_phase(searcher: &UnifiedSearcher) {
    println!("🔍 Phase 4: Retrieval Validation");
    println!("===============================");
    
    // Test various search scenarios
    let retrieval_test_cases = vec![
        // Exact matches
        ("UserController", vec!["user_controller.py"], "Exact class name match"),
        ("async function", vec!["async_utils.js"], "Language construct search"),
        ("database connection", vec!["db_manager.py", "config.toml"], "Concept-based search"),
        
        // Fuzzy matches
        ("authenticate user", vec!["auth_service.py"], "Natural language to code"),
        ("calculate hash", vec!["crypto_utils.rs"], "Functionality description"),
        
        // Multi-modal matches
        ("JWT token validation", vec!["auth_service.py", "security_middleware.js"], "Cross-language concept"),
        ("error handling", vec!["error_handler.rs", "exceptions.py"], "Common patterns"),
    ];
    
    let mut search_performance = Vec::new();
    let mut relevance_scores = Vec::new();
    
    for (query, expected_files, test_description) in retrieval_test_cases {
        println!("  Testing: {} - '{}'", test_description, query);
        
        let start = std::time::Instant::now();
        let results = timeout(
            Duration::from_secs(30),
            searcher.search(query)
        ).await.expect(&format!("Search for '{}' timed out", query))
            .expect(&format!("Search for '{}' failed", query));
        
        let search_time = start.elapsed();
        search_performance.push(search_time);
        
        assert!(!results.is_empty(), "No results for query: {}", query);
        
        // Check if expected files are in results
        let mut found_expected = 0;
        for expected_file in &expected_files {
            if results.iter().any(|r| r.file.contains(expected_file)) {
                found_expected += 1;
            }
        }
        
        let relevance = found_expected as f32 / expected_files.len() as f32;
        relevance_scores.push(relevance);
        
        println!("    ✅ Found {}/{} expected files, {:?}, top score: {:.3}", 
                 found_expected, expected_files.len(), search_time, results[0].score);
        
        // Show top results for debugging
        for (i, result) in results.iter().take(3).enumerate() {
            println!("      {}. {} (score: {:.3})", i + 1, result.file, result.score);
        }
    }
    
    // Calculate performance metrics
    let avg_search_time = search_performance.iter().sum::<std::time::Duration>() / search_performance.len() as u32;
    let max_search_time = *search_performance.iter().max().unwrap();
    let avg_relevance = relevance_scores.iter().sum::<f32>() / relevance_scores.len() as f32;
    
    println!("\n  📊 Retrieval Performance Summary:");
    println!("    Average search time: {:?}", avg_search_time);
    println!("    Maximum search time: {:?}", max_search_time);
    println!("    Average relevance score: {:.2}", avg_relevance);
    
    // Performance and relevance assertions
    assert!(avg_search_time.as_secs() < 5, "Average search too slow: {:?}", avg_search_time);
    assert!(max_search_time.as_secs() < 15, "Maximum search too slow: {:?}", max_search_time);
    assert!(avg_relevance > 0.6, "Average relevance too low: {:.2}", avg_relevance);
    
    println!("  ✅ Retrieval validation passed\n");
}

/// Test ranking and result quality
async fn validate_ranking_phase(searcher: &UnifiedSearcher) {
    println!("🏆 Phase 5: Ranking Validation");
    println!("=============================");
    
    // Test ranking quality with specific scenarios
    let ranking_tests = vec![
        ("exact match test", "UserController", |results: &[_]| {
            // Exact matches should rank highest
            results[0].score > 0.8
        }),
        ("semantic similarity", "user authentication", |results: &[_]| {
            // Should find auth-related files with reasonable scores
            results.iter().take(3).any(|r| r.score > 0.5 && 
                (r.file.contains("auth") || r.file.contains("user")))
        }),
        ("code structure query", "async function definition", |results: &[_]| {
            // Should find JavaScript/TypeScript files with async functions
            results.iter().take(5).any(|r| r.score > 0.4 && 
                (r.file.contains(".js") || r.file.contains(".ts")))
        }),
    ];
    
    let mut ranking_quality_scores = Vec::new();
    
    for (test_name, query, quality_check) in ranking_tests {
        println!("  Testing ranking: {} - '{}'", test_name, query);
        
        let results = searcher.search(query).await.expect(&format!("Search failed for {}", query));
        assert!(!results.is_empty(), "No results for ranking test: {}", query);
        
        // Check that results are properly sorted (descending by score)
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score, 
                    "Results not properly sorted at position {}: {} vs {}", 
                    i, results[i-1].score, results[i].score);
        }
        
        // Apply quality check
        let quality_passed = quality_check(&results);
        ranking_quality_scores.push(if quality_passed { 1.0 } else { 0.0 });
        
        println!("    ✅ Ranking quality: {}, top score: {:.3}", 
                 if quality_passed { "PASS" } else { "FAIL" }, results[0].score);
        
        // Show ranking distribution
        let score_ranges = [(0.8, 1.0, "excellent"), (0.6, 0.8, "good"), (0.4, 0.6, "fair"), (0.0, 0.4, "poor")];
        for (min_score, max_score, label) in score_ranges {
            let count = results.iter().filter(|r| r.score >= min_score && r.score < max_score).count();
            if count > 0 {
                println!("      {} results with {} scores ({:.1}-{:.1})", count, label, min_score, max_score);
            }
        }
    }
    
    let ranking_quality = ranking_quality_scores.iter().sum::<f32>() / ranking_quality_scores.len() as f32;
    
    println!("\n  📊 Ranking Performance Summary:");
    println!("    Overall ranking quality: {:.1}%", ranking_quality * 100.0);
    
    assert!(ranking_quality > 0.7, "Ranking quality too low: {:.1}%", ranking_quality * 100.0);
    
    println!("  ✅ Ranking validation passed\n");
}

/// Test complete end-to-end workflows
async fn validate_end_to_end_workflows(searcher: &UnifiedSearcher) {
    println!("🔄 Phase 6: End-to-End Workflow Validation");
    println!("===========================================");
    
    // Workflow 1: New file addition and immediate search
    println!("  Testing workflow: New file → Index → Search");
    
    // Create a new test file
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let new_file_path = temp_dir.path().join("new_feature.py");
    let new_file_content = "class NewFeatureManager:\n    def __init__(self, config):\n        self.config = config\n    \n    def process_new_feature(self, data):\n        return self.transform_data(data)\n    \n    def transform_data(self, data):\n        return {\"processed\": data, \"timestamp\": time.now()}";
    
    std::fs::write(&new_file_path, new_file_content).expect("Failed to write new file");
    
    // Index the new file
    let start = std::time::Instant::now();
    searcher.index_file(&new_file_path).await.expect("Failed to index new file");
    let index_time = start.elapsed();
    
    // Search for content from the new file
    let search_results = searcher.search("NewFeatureManager").await.expect("Search failed");
    
    assert!(!search_results.is_empty(), "New file content not found in search results");
    let found_new_file = search_results.iter().any(|r| r.file.contains("new_feature.py"));
    assert!(found_new_file, "New file not found in search results");
    
    println!("    ✅ New file workflow: indexed in {:?}, found in search", index_time);
    
    // Workflow 2: Batch processing and search accuracy
    println!("  Testing workflow: Batch Index → Multiple Searches → Result Consistency");
    
    let batch_queries = vec![
        "class definition",
        "function implementation", 
        "error handling",
        "configuration",
        "data processing",
    ];
    
    let mut consistency_scores = Vec::new();
    
    for query in &batch_queries {
        // Run the same search multiple times
        let mut results_sets = Vec::new();
        
        for _ in 0..3 {
            let results = searcher.search(query).await.expect(&format!("Search failed for {}", query));
            results_sets.push(results);
        }
        
        // Check consistency between runs
        let first_results = &results_sets[0];
        let mut consistency_score = 0.0;
        
        for other_results in &results_sets[1..] {
            if first_results.len() == other_results.len() {
                let matching = first_results.iter().zip(other_results.iter())
                    .filter(|(a, b)| a.file == b.file && (a.score - b.score).abs() < 0.01)
                    .count();
                
                consistency_score += matching as f32 / first_results.len() as f32;
            }
        }
        
        consistency_score /= (results_sets.len() - 1) as f32;
        consistency_scores.push(consistency_score);
        
        println!("    ✅ Query '{}': {:.1}% consistent", query, consistency_score * 100.0);
    }
    
    let avg_consistency = consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32;
    
    println!("\n  📊 End-to-End Performance Summary:");
    println!("    New file indexing time: {:?}", index_time);
    println!("    Search result consistency: {:.1}%", avg_consistency * 100.0);
    
    // Workflow assertions
    assert!(index_time.as_secs() < 60, "New file indexing too slow: {:?}", index_time);
    assert!(avg_consistency > 0.95, "Search results not consistent enough: {:.1}%", avg_consistency * 100.0);
    
    println!("  ✅ End-to-end workflow validation passed\n");
}

// Helper functions

async fn create_test_codebase(base_path: &std::path::Path) {
    let test_files = vec![
        ("user_controller.py", "class UserController:\n    def __init__(self, db_connection):\n        self.db = db_connection\n    \n    def create_user(self, user_data):\n        return self.db.insert('users', user_data)\n    \n    def authenticate(self, username, password):\n        user = self.db.find_user(username)\n        return self.verify_password(user, password)"),
        
        ("auth_service.py", "import jwt\nimport bcrypt\n\nclass AuthenticationService:\n    def __init__(self, secret_key):\n        self.secret_key = secret_key\n    \n    def generate_token(self, user_id):\n        payload = {'user_id': user_id, 'exp': time.time() + 3600}\n        return jwt.encode(payload, self.secret_key, algorithm='HS256')\n    \n    def validate_token(self, token):\n        try:\n            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])\n            return payload['user_id']\n        except jwt.ExpiredSignatureError:\n            raise AuthenticationError('Token expired')"),
        
        ("async_utils.js", "export class AsyncUtilities {\n  static async delay(ms) {\n    return new Promise(resolve => setTimeout(resolve, ms));\n  }\n\n  static async fetchWithRetry(url, options = {}, maxRetries = 3) {\n    for (let i = 0; i < maxRetries; i++) {\n      try {\n        const response = await fetch(url, options);\n        if (response.ok) return response;\n      } catch (error) {\n        if (i === maxRetries - 1) throw error;\n        await this.delay(1000 * Math.pow(2, i));\n      }\n    }\n  }\n}"),
        
        ("db_manager.py", "import asyncpg\nimport asyncio\nfrom contextlib import asynccontextmanager\n\nclass DatabaseManager:\n    def __init__(self, connection_string):\n        self.connection_string = connection_string\n        self.pool = None\n    \n    async def initialize(self):\n        self.pool = await asyncpg.create_pool(\n            self.connection_string,\n            min_size=1,\n            max_size=10\n        )\n    \n    @asynccontextmanager\n    async def get_connection(self):\n        async with self.pool.acquire() as connection:\n            yield connection\n    \n    async def execute_query(self, query, *args):\n        async with self.get_connection() as conn:\n            return await conn.fetch(query, *args)"),
        
        ("crypto_utils.rs", "use sha2::{Sha256, Digest};\nuse rand::{thread_rng, Rng};\n\npub struct CryptoUtils;\n\nimpl CryptoUtils {\n    pub fn calculate_hash(data: &[u8]) -> String {\n        let mut hasher = Sha256::new();\n        hasher.update(data);\n        format!(\"{:x}\", hasher.finalize())\n    }\n    \n    pub fn generate_salt() -> Vec<u8> {\n        let mut salt = vec![0u8; 32];\n        thread_rng().fill(&mut salt[..]);\n        salt\n    }\n    \n    pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {\n        if a.len() != b.len() {\n            return false;\n        }\n        let mut result = 0u8;\n        for (x, y) in a.iter().zip(b.iter()) {\n            result |= x ^ y;\n        }\n        result == 0\n    }\n}"),
        
        ("error_handler.rs", "use thiserror::Error;\nuse std::fmt;\n\n#[derive(Error, Debug)]\npub enum AppError {\n    #[error(\"Database error: {message}\")]\n    Database { message: String },\n    \n    #[error(\"Authentication failed: {reason}\")]\n    Authentication { reason: String },\n    \n    #[error(\"Validation error: {field} is invalid\")]\n    Validation { field: String },\n    \n    #[error(\"Network error: {0}\")]\n    Network(#[from] reqwest::Error),\n}\n\nimpl AppError {\n    pub fn database(message: impl Into<String>) -> Self {\n        Self::Database { message: message.into() }\n    }\n    \n    pub fn auth(reason: impl Into<String>) -> Self {\n        Self::Authentication { reason: reason.into() }\n    }\n}"),
        
        ("exceptions.py", "class BaseApplicationError(Exception):\n    \"\"\"Base exception for all application errors.\"\"\"\n    def __init__(self, message, error_code=None, details=None):\n        super().__init__(message)\n        self.message = message\n        self.error_code = error_code\n        self.details = details or {}\n\nclass DatabaseError(BaseApplicationError):\n    \"\"\"Database-related errors.\"\"\"\n    def __init__(self, message, query=None, **kwargs):\n        super().__init__(message, **kwargs)\n        self.query = query\n\nclass AuthenticationError(BaseApplicationError):\n    \"\"\"Authentication and authorization errors.\"\"\"\n    pass\n\nclass ValidationError(BaseApplicationError):\n    \"\"\"Data validation errors.\"\"\"\n    def __init__(self, message, field=None, **kwargs):\n        super().__init__(message, **kwargs)\n        self.field = field"),
        
        ("security_middleware.js", "import jwt from 'jsonwebtoken';\n\nexport class SecurityMiddleware {\n  constructor(secretKey) {\n    this.secretKey = secretKey;\n  }\n\n  authenticate = (req, res, next) => {\n    const token = this.extractToken(req);\n    if (!token) {\n      return res.status(401).json({ error: 'No token provided' });\n    }\n\n    try {\n      const payload = jwt.verify(token, this.secretKey);\n      req.user = payload;\n      next();\n    } catch (error) {\n      return res.status(401).json({ error: 'Invalid token' });\n    }\n  };\n\n  extractToken(req) {\n    const authHeader = req.headers.authorization;\n    if (authHeader && authHeader.startsWith('Bearer ')) {\n      return authHeader.substring(7);\n    }\n    return null;\n  }\n}"),
        
        ("config.toml", "[database]\nhost = \"localhost\"\nport = 5432\nname = \"myapp\"\nuser = \"admin\"\npassword = \"secure123\"\nmax_connections = 20\nconnection_timeout = 30\n\n[cache]\nredis_url = \"redis://localhost:6379\"\nttl = 3600\nmax_size = 1000\n\n[logging]\nlevel = \"info\"\nfile = \"/var/log/myapp.log\"\nrotate = true\nmax_size = \"10MB\"\n\n[security]\njwt_secret = \"your-secret-key-here\"\ntoken_expiry = 3600\npassword_min_length = 8\n"),
    ];
    
    for (filename, content) in test_files {
        let file_path = base_path.join(filename);
        std::fs::write(file_path, content).expect(&format!("Failed to create {}", filename));
    }
}

fn calculate_embedding_quality_score(embedding: &[f32]) -> f32 {
    // Simple heuristic for embedding quality
    let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
    let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
    let std_dev = variance.sqrt();
    
    // Good embeddings should have:
    // - Mean close to 0 (normalized)
    // - Reasonable variance (not too sparse, not too dense)
    // - No extreme values
    
    let mean_score = (1.0 - mean.abs() * 10.0).max(0.0); // Penalize large means
    let variance_score = if variance > 0.01 && variance < 0.5 { 1.0 } else { 0.5 }; // Good variance range
    let range_score = if std_dev > 0.05 && std_dev < 2.0 { 1.0 } else { 0.7 }; // Reasonable spread
    
    (mean_score + variance_score + range_score) / 3.0
}

fn create_long_test_code() -> String {
    "pub struct ComplexDataProcessor {\n    config: ProcessorConfig,\n    cache: HashMap<String, ProcessedData>,\n    metrics: Arc<RwLock<ProcessingMetrics>>,\n}\n\nimpl ComplexDataProcessor {\n    pub fn new(config: ProcessorConfig) -> Result<Self, ProcessorError> {\n        Ok(Self {\n            config,\n            cache: HashMap::with_capacity(1000),\n            metrics: Arc::new(RwLock::new(ProcessingMetrics::default())),\n        })\n    }\n\n    pub async fn process_batch(&mut self, items: Vec<InputData>) -> Result<Vec<OutputData>, ProcessorError> {\n        let mut results = Vec::with_capacity(items.len());\n        let start_time = Instant::now();\n        \n        for item in items {\n            match self.process_single_item(&item).await {\n                Ok(processed) => results.push(processed),\n                Err(e) => {\n                    self.record_error(&e);\n                    return Err(e);\n                }\n            }\n        }\n        \n        self.update_metrics(start_time.elapsed(), results.len());\n        Ok(results)\n    }\n}".to_string()
}