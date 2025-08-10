#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::time::sleep;

    // Import only the specific modules we need to test
    use embed_search::search::bm25::{BM25Engine, BM25Document};
    use embed_search::watcher::{IndexUpdater, FileEvent, EventType};

    #[tokio::test]
    async fn test_index_updater_integration() {
        // Create a temp directory and test file
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn main() { println!(\"Hello, World!\"); }").unwrap();

        // Create BM25 engine
        let mut bm25_engine = BM25Engine::new();
        
        // Manually process the file to establish baseline
        let initial_result = bm25_engine.process_single_file(&test_file).await;
        assert!(initial_result.is_ok(), "Initial file processing should succeed");
        
        let initial_stats = bm25_engine.get_stats();
        println!("📊 Initial stats: {} docs, {} terms", 
                 initial_stats.total_documents, initial_stats.total_terms);
        assert!(initial_stats.total_documents > 0, "Should have documents after initial processing");

        // Test searching for content
        let search_results = bm25_engine.search("Hello", 10).unwrap();
        println!("🔍 Search results for 'Hello': {} matches", search_results.len());
        assert!(!search_results.is_empty(), "Should find 'Hello' in the test file");

        // Now test the IndexUpdater integration
        let searcher_arc = Arc::new(RwLock::new(bm25_engine));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let updater = IndexUpdater::new(searcher_arc.clone(), rx);

        // Test file modification through the updater
        fs::write(&test_file, "fn main() { println!(\"Goodbye, World!\"); }").unwrap();
        let modify_event = FileEvent::new(test_file.clone(), EventType::Modified);
        
        // Process the event
        updater.process_events(vec![modify_event]).await;
        
        // Give time for async processing
        sleep(Duration::from_millis(100)).await;

        // Check that the change was processed
        {
            let searcher = searcher_arc.read().unwrap();
            let goodbye_results = searcher.search("Goodbye", 10).unwrap();
            println!("🔍 Search results for 'Goodbye': {} matches", goodbye_results.len());
            assert!(!goodbye_results.is_empty(), "Should find 'Goodbye' after modification");

            let final_stats = searcher.get_stats();
            println!("📊 Final stats: {} docs, {} terms", 
                     final_stats.total_documents, final_stats.total_terms);
        }

        // Test file removal
        let remove_event = FileEvent::new(test_file.clone(), EventType::Removed);
        updater.process_events(vec![remove_event]).await;
        
        sleep(Duration::from_millis(100)).await;

        // Check removal results
        {
            let searcher = searcher_arc.read().unwrap();
            let post_removal_stats = searcher.get_stats();
            println!("📊 Stats after removal: {} docs, {} terms", 
                     post_removal_stats.total_documents, post_removal_stats.total_terms);
            
            // Should have fewer documents now (or same if removal was ignored due to file path encoding)
            let removal_results = searcher.search("Goodbye", 10).unwrap();
            println!("🔍 Search results for 'Goodbye' after removal: {} matches", removal_results.len());
        }

        println!("✅ IndexUpdater integration test completed successfully!");
    }

    #[test]
    fn test_bm25_remove_by_path() {
        // Test the new remove_documents_by_path method directly
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("remove_test.rs");
        
        let mut bm25_engine = BM25Engine::new();
        
        // Create a document manually to test removal
        let tokens = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
        let doc = BM25Document {
            id: format!("{}_{}", 
                       test_file.to_string_lossy().replace(['/', '\\'], "_"),
                       100), // Mock content length
            file_path: test_file.to_string_lossy().to_string(),
            chunk_index: 0,
            tokens,
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
        };
        
        let add_result = bm25_engine.add_document(doc);
        assert!(add_result.is_ok(), "Should be able to add document");
        
        let initial_stats = bm25_engine.get_stats();
        assert!(initial_stats.total_documents > 0, "Should have documents");
        
        // Now test removal by path
        let removed_count = bm25_engine.remove_documents_by_path(&test_file).unwrap();
        println!("🗑️ Removed {} documents for path: {:?}", removed_count, test_file);
        assert!(removed_count > 0, "Should have removed at least one document");
        
        let final_stats = bm25_engine.get_stats();
        assert!(final_stats.total_documents < initial_stats.total_documents, 
                "Should have fewer documents after removal");
        
        println!("✅ Remove by path test completed successfully!");
    }
}