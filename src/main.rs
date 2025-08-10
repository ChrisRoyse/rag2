use std::path::PathBuf;
use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(feature = "vectordb")]
use std::sync::Arc;
#[cfg(feature = "vectordb")]
use tokio::sync::RwLock;

use embed_search::{
    search::BM25Searcher,
    config::Config,
};

#[cfg(feature = "vectordb")]
use embed_search::{
    storage::lancedb_storage::LanceDBStorage,
    git::watcher::WatchCommand,
};

#[derive(Parser)]
#[command(name = "embed-search")]
#[command(about = "Intelligent embedding-based search system for codebases")]
#[command(version)]
struct Cli {
    /// Custom configuration file path
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index files in the specified path
    Index {
        /// Path to index (required - no default)
        path: PathBuf,
    },
    /// Search for a query in the indexed files
    Search {
        /// Search query
        query: String,
    },
    /// Start watching for file changes
    Watch,
    /// Run a single update for changed files
    Update,
    /// Clear all indexed data
    Clear,
    /// Show index statistics
    Stats,
    /// Run comprehensive tests on vectortest directory
    Test,
    /// Show current configuration
    Config {
        /// Show configuration as JSON
        #[arg(long)]
        json: bool,
    },
    /// Validate configuration file
    ValidateConfig {
        /// Configuration file to validate
        file: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize configuration
    if let Some(config_path) = &cli.config {
        Config::init_with_file(config_path)?;
        println!("📝 Loaded configuration from: {:?}", config_path);
    } else {
        Config::init()?;
    }
    
    // Validate configuration
    let config = Config::get()?;
    config.validate()?;
    
    println!("🚀 Embed Search System - Production Ready");
    println!("=========================================\n");
    
    // Setup paths from configuration
    let project_path = std::env::current_dir()?;
    let db_path = project_path.join(&config.vector_db_path);
    
    match cli.command {
        Commands::Index { path } => {
            index_command(path, project_path, db_path).await?;
        },
        Commands::Search { query } => {
            search_command(&query, project_path, db_path).await?;
        },
        Commands::Watch => {
            watch_command(project_path.clone(), db_path).await?;
        },
        Commands::Update => {
            update_command(project_path.clone(), db_path).await?;
        },
        Commands::Clear => {
            clear_command(project_path, db_path).await?;
        },
        Commands::Stats => {
            stats_command(project_path, db_path).await?;
        },
        Commands::Test => {
            test_command(project_path, db_path).await?;
        },
        Commands::Config { json } => {
            config_command(json).await?;
        },
        Commands::ValidateConfig { file } => {
            validate_config_command(file).await?;
        },
    }
    
    Ok(())
}


async fn index_command(path: PathBuf, project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("📂 Indexing path: {:?}", path);
    
    let mut searcher = BM25Searcher::new();
    
    let full_path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    };
    
    if full_path.is_file() {
        // BM25Engine doesn't have index_file method, use index_directory for single files
        return Err(anyhow::anyhow!("Individual file indexing not supported, use directory indexing"));
    } else if full_path.is_dir() {
        let stats = searcher.index_directory(&full_path).await?;
        println!("✅ {:?}", stats);
    } else {
        return Err(anyhow::anyhow!("Path does not exist: {:?}", full_path));
    }
    
    Ok(())
}

async fn search_command(query: &str, project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("🔍 Searching for: \"{}\"", query);
    
    let searcher = BM25Searcher::new();
    let results = searcher.search(query, 10)?;
    
    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }
    
    println!("\n📊 Found {} results:\n", results.len());
    
    let max_display = std::cmp::min(5, Config::max_search_results()?);
    for (idx, result) in results.iter().take(max_display).enumerate() {
        println!("{}. {} (score: {:.2})", idx + 1, result.doc_id, result.score);
        println!("   Terms: {:?}", result.matched_terms);
        
        // Show matched terms and their scores
        if !result.term_scores.is_empty() {
            println!("   Term scores:");
            for (term, score) in &result.term_scores {
                println!("     {}: {:.3}", term, score);
            }
        }
        
        println!();
    }
    
    if results.len() > max_display {
        println!("... and {} more results", results.len() - max_display);
    }
    
    Ok(())
}

#[cfg(feature = "vectordb")]
async fn watch_command(project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("👁️  Starting file watch mode...");
    
    let searcher = Arc::new(BM25Searcher::new());
    let storage = Arc::new(RwLock::new(LanceDBStorage::new(db_path).await?));
    
    let watch = WatchCommand::new(project_path, searcher, storage)?;
    watch.start().await;
    
    println!("Watching for file changes. Press Ctrl+C to stop.");
    
    // Keep the program running
    tokio::signal::ctrl_c().await?;
    
    watch.stop();
    println!("\n👋 Stopped watching");
    
    Ok(())
}

#[cfg(not(feature = "vectordb"))]
async fn watch_command(_project_path: PathBuf, _db_path: PathBuf) -> Result<()> {
    println!("❌ Watch functionality requires 'vectordb' feature to be enabled");
    std::process::exit(1);
}

#[cfg(feature = "vectordb")]
async fn update_command(project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("🔄 Checking for file changes...");
    
    let searcher = Arc::new(BM25Searcher::new());
    let storage = Arc::new(RwLock::new(LanceDBStorage::new(db_path).await?));
    
    let watch = WatchCommand::new(project_path, searcher, storage)?;
    let stats = watch.run_once().await?;
    
    println!("✅ {}", stats);
    
    Ok(())
}

#[cfg(not(feature = "vectordb"))]
async fn update_command(_project_path: PathBuf, _db_path: PathBuf) -> Result<()> {
    println!("❌ Update functionality requires 'vectordb' feature to be enabled");
    std::process::exit(1);
}

async fn clear_command(project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("🧹 Clearing all indexed data...");
    
    let searcher = BM25Searcher::new();
    // BM25Engine doesn't have clear_index method, create new instance to clear
    return Err(anyhow::anyhow!("Index clearing not implemented for BM25Engine"));
    
    println!("✅ Index cleared");
    
    Ok(())
}

async fn stats_command(project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("📊 Index Statistics");
    println!("==================");
    
    let searcher = BM25Searcher::new();
    let stats = searcher.get_stats();
    
    println!("Total documents: {}", stats.total_documents);
    println!("Total terms: {}", stats.total_terms);
    println!("Average document length: {:.2}", stats.avg_document_length);
    
    Ok(())
}

async fn test_command(project_path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("🧪 Running comprehensive tests on vectortest directory");
    println!("=====================================================\n");
    
    let vectortest_path = project_path.join("vectortest");
    
    if !vectortest_path.exists() {
        return Err(anyhow::anyhow!("vectortest directory not found: {}. Cannot run tests without test data.", vectortest_path.display()));
    }
    
    // Clear index first
    println!("1️⃣  Clearing existing index...");
    let mut searcher = BM25Searcher::new();
    // BM25Engine doesn't have clear_index method, create new instance instead
    
    // Index the vectortest directory
    println!("2️⃣  Indexing vectortest directory...");
    let stats = searcher.index_directory(&vectortest_path).await?;
    println!("   {:?}", stats);
    
    // Test searches
    println!("\n3️⃣  Running test searches...");
    
    let test_queries = vec![
        ("authentication", vec!["auth_service.py", "user_controller.js"]),
        ("database migration", vec!["database_migration.sql"]),
        ("websocket", vec!["websocket_server.cpp"]),
        ("payment", vec!["payment_gateway.ts"]),
        ("cache", vec!["memory_cache.rs"]),
        ("analytics", vec!["analytics_dashboard.go"]),
        ("def authenticate", vec!["auth_service.py"]),
        ("OrderService", vec!["OrderService.java"]),
        ("troubleshooting", vec!["TROUBLESHOOTING.md"]),
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (query, expected_files) in test_queries {
        print!("   Testing query \"{}\"... ", query);
        
        let results = searcher.search(query, 10)?;
        
        let mut found_expected = false;
        for expected_file in &expected_files {
            if results.iter().any(|r| r.doc_id.contains(expected_file)) {
                found_expected = true;
                break;
            }
        }
        
        if found_expected {
            println!("✅ PASS");
            passed += 1;
        } else {
            println!("❌ FAIL (expected files not in top results)");
            failed += 1;
        }
    }
    
    println!("\n📊 Test Results:");
    println!("   Passed: {}", passed);
    println!("   Failed: {}", failed);
    println!("   Success Rate: {:.1}%", (passed as f32 / (passed + failed) as f32) * 100.0);
    
    if failed == 0 {
        println!("\n🎉 All tests passed!");
    } else {
        println!("\n⚠️  Some tests failed. Please review the implementation.");
    }
    
    Ok(())
}

async fn config_command(json: bool) -> Result<()> {
    let config = Config::get()?;
    
    if json {
        println!("{}", serde_json::to_string_pretty(&config)?);
    } else {
        println!("{}", config.summary());
    }
    
    Ok(())
}

async fn validate_config_command(file: Option<PathBuf>) -> Result<()> {
    let (config, is_from_file) = if let Some(path) = file {
        println!("📋 Validating configuration file: {:?}", path);
        (Config::load_from_file(&path)?, true)
    } else {
        println!("📋 Validating current configuration...");
        (Config::get()?, false)
    };
    
    match config.validate() {
        Ok(()) => {
            println!("✅ Configuration is valid!");
            if !is_from_file {
                println!("\n{}", config.summary());
            }
        },
        Err(e) => {
            println!("❌ Configuration validation failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
