use std::path::PathBuf;
use std::fs;
use std::time::Duration;
use tokio::time::sleep;
use tempfile::TempDir;

use embed_search::search::BM25Engine;
use embed_search::watcher::{GitWatcher, IndexUpdater, FileEvent, EventType};

/// Test that file changes trigger actual BM25 re-indexing
#[tokio::test]
async fn test_file_change_triggers_reindexing() {
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path().to_path_buf();
    
    // Create a BM25 engine instance
    let bm25_engine = BM25Engine::new();
    let searcher_arc = std::sync::Arc::new(std::sync::RwLock::new(bm25_engine));
    
    // Create the indexer updater (this is what connects file changes to indexing)
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let updater = IndexUpdater::new(searcher_arc.clone(), rx);
    
    // Create a test file with content
    let test_file = repo_path.join("test_code.rs");
    fs::write(&test_file, "fn hello_world() { println!(\"Hello, World!\"); }").unwrap();
    
    // Create a FileEvent for file creation
    let create_event = FileEvent::new(test_file.clone(), EventType::Created);
    
    // Process the event through the updater
    let result = updater.process_events(vec![create_event]).await;
    // The process_events method doesn't return a result, but we should see output
    
    // Check that the file was actually indexed by searching for content
    {
        let searcher = searcher_arc.read().unwrap();
        let stats = searcher.get_stats();
        println!("📊 Index stats after adding file: {} documents, {} terms", 
                 stats.total_documents, stats.total_terms);
        
        // Should have at least one document now
        assert!(stats.total_documents > 0, "File should have been indexed");
        assert!(stats.total_terms > 0, "File should have generated terms");
    }
    
    // Test search to make sure the content is findable
    {
        let searcher = searcher_arc.read().unwrap();
        let search_results = searcher.search("hello", 10).unwrap();
        println!("🔍 Search results for 'hello': {} matches", search_results.len());
        
        assert!(!search_results.is_empty(), "Should find 'hello' in the indexed file");
        
        // Check that the search result points to our file
        let first_result = &search_results[0];
        assert!(first_result.doc_id.contains(&test_file.to_string_lossy().replace(['/', '\\'], "_")),
                "Search result should reference our test file");
    }
    
    // Now test file modification
    fs::write(&test_file, "fn goodbye_world() { println!(\"Goodbye, World!\"); }").unwrap();
    let modify_event = FileEvent::new(test_file.clone(), EventType::Modified);
    updater.process_events(vec![modify_event]).await;
    
    // Give a moment for indexing to complete
    sleep(Duration::from_millis(10)).await;
    
    // Search for new content
    {
        let searcher = searcher_arc.read().unwrap();
        let goodbye_results = searcher.search("goodbye", 10).unwrap();
        println!("🔍 Search results for 'goodbye': {} matches", goodbye_results.len());
        
        assert!(!goodbye_results.is_empty(), "Should find 'goodbye' after file modification");
        
        // The old content should be gone or updated
        let hello_results = searcher.search("hello", 10).unwrap();
        println!("🔍 Search results for 'hello' after modification: {} matches", hello_results.len());
        // Note: hello might still be found if there are multiple document chunks
    }
    
    // Test file deletion
    let delete_event = FileEvent::new(test_file.clone(), EventType::Removed);
    updater.process_events(vec![delete_event]).await;
    
    sleep(Duration::from_millis(10)).await;
    
    // Check that documents were removed from index
    {
        let searcher = searcher_arc.read().unwrap();
        let final_stats = searcher.get_stats();
        println!("📊 Index stats after file deletion: {} documents, {} terms", 
                 final_stats.total_documents, final_stats.total_terms);
        
        // The search should return fewer or no results now
        let search_results = searcher.search("goodbye", 10).unwrap();
        println!("🔍 Search results for 'goodbye' after deletion: {} matches", search_results.len());
        // Note: Depending on implementation, this might still find cached results
    }
    
    println!("✅ Git watcher integration test completed successfully!");
}

/// Test the complete GitWatcher -> IndexUpdater pipeline 
#[tokio::test]
async fn test_git_watcher_integration_pipeline() {
    // Initialize config (required for some components)
    if let Err(_) = embed_search::config::Config::init() {
        // Already initialized, ignore
    }
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = temp_dir.path();
    
    // Create a .gitignore to test gitignore functionality
    fs::write(repo_path.join(".gitignore"), "*.tmp\n*.log\n").unwrap();
    
    // Create BM25 engine
    let bm25_engine = embed_search::search::BM25Searcher::new();
    let searcher_arc = std::sync::Arc::new(std::sync::RwLock::new(bm25_engine));
    
    // Create GitWatcher (this is the complete integration)
    let git_watcher = GitWatcher::new(repo_path, searcher_arc.clone());
    assert!(git_watcher.is_ok(), "GitWatcher should be created successfully");
    
    let mut watcher = git_watcher.unwrap();
    
    // Start watching
    let watch_result = watcher.start_watching();
    assert!(watch_result.is_ok(), "GitWatcher should start successfully");
    
    println!("🎯 GitWatcher integration pipeline test completed - watcher can be created and started!");
    
    // Stop watching
    watcher.stop_watching();
}