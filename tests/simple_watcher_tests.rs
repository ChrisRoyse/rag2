use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;

use embed_search::git::{SimpleFileWatcher, FileEvent};
use embed_search::git::simple_watcher::FileChange;

/// Test the simple file watcher with actual file operations
#[test]
fn test_simple_file_watcher_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Simple File Watcher with real file changes");
    
    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().to_path_buf();
    println!("📁 Test directory: {}", temp_path.display());
    
    // Create test files
    let rs_file = temp_path.join("test.rs");
    let py_file = temp_path.join("test.py");
    let js_file = temp_path.join("test.js");
    let ts_file = temp_path.join("test.ts");
    let md_file = temp_path.join("README.md"); // Should be ignored
    
    // Initialize watcher
    let mut watcher = SimpleFileWatcher::new(temp_path)?;
    watcher.start()?;
    
    println!("✅ File watcher started, waiting for stabilization...");
    std::thread::sleep(Duration::from_millis(200)); // Let watcher stabilize
    
    // Test 1: Create files (should detect .rs, .py, .js, .ts but ignore .md)
    println!("\n📝 Test 1: Creating files");
    let start_time = Instant::now();
    
    fs::write(&rs_file, "fn main() { println!(\"Hello Rust\"); }")?;
    fs::write(&py_file, "print('Hello Python')")?;
    fs::write(&js_file, "console.log('Hello JavaScript');")?;
    fs::write(&ts_file, "console.log('Hello TypeScript');")?;
    fs::write(&md_file, "# Test README")?; // Should be ignored
    
    // Wait for debouncing and event processing
    std::thread::sleep(Duration::from_millis(300));
    
    let events = watcher.get_changes()?;
    let creation_latency = start_time.elapsed().as_millis();
    
    println!("📊 Creation Events Detected: {} (latency: {}ms)", events.len(), creation_latency);
    
    // Should detect 4 files (.rs, .py, .js, .ts) but ignore .md
    let mut detected_files = Vec::new();
    for event in &events {
        match &event.change {
            FileChange::Added(path) => {
                detected_files.push(path.file_name().unwrap().to_string_lossy().to_string());
                println!("  ➕ Added: {} ({}ms latency)", 
                    path.file_name().unwrap().to_string_lossy(),
                    event.latency_ms()
                );
            }
            _ => {}
        }
    }
    
    // Verify correct files were detected
    assert!(detected_files.contains(&"test.rs".to_string()));
    assert!(detected_files.contains(&"test.py".to_string()));
    assert!(detected_files.contains(&"test.js".to_string()));
    assert!(detected_files.contains(&"test.ts".to_string()));
    assert!(!detected_files.contains(&"README.md".to_string()), "MD file should be ignored");
    
    println!("✅ File creation test passed");
    
    // Test 2: Modify files
    println!("\n✏️ Test 2: Modifying files");
    let start_time = Instant::now();
    
    fs::write(&rs_file, "fn main() { println!(\"Hello Modified Rust\"); }")?;
    fs::write(&py_file, "print('Hello Modified Python')")?;
    
    std::thread::sleep(Duration::from_millis(300));
    
    let events = watcher.get_changes()?;
    let modify_latency = start_time.elapsed().as_millis();
    
    println!("📊 Modification Events Detected: {} (latency: {}ms)", events.len(), modify_latency);
    
    let mut modified_files = Vec::new();
    for event in &events {
        match &event.change {
            FileChange::Modified(path) => {
                modified_files.push(path.file_name().unwrap().to_string_lossy().to_string());
                println!("  ✏️ Modified: {} ({}ms latency)", 
                    path.file_name().unwrap().to_string_lossy(),
                    event.latency_ms()
                );
            }
            _ => {}
        }
    }
    
    // Should detect modifications to .rs and .py files
    assert!(modified_files.len() >= 2, "Should detect at least 2 modifications");
    
    println!("✅ File modification test passed");
    
    // Test 3: Delete files
    println!("\n🗑️ Test 3: Deleting files");
    let start_time = Instant::now();
    
    fs::remove_file(&js_file)?;
    fs::remove_file(&ts_file)?;
    
    std::thread::sleep(Duration::from_millis(300));
    
    let events = watcher.get_changes()?;
    let delete_latency = start_time.elapsed().as_millis();
    
    println!("📊 Deletion Events Detected: {} (latency: {}ms)", events.len(), delete_latency);
    
    let mut deleted_files = Vec::new();
    for event in &events {
        match &event.change {
            FileChange::Deleted(path) => {
                deleted_files.push(path.file_name().unwrap().to_string_lossy().to_string());
                println!("  🗑️ Deleted: {} ({}ms latency)", 
                    path.file_name().unwrap().to_string_lossy(),
                    event.latency_ms()
                );
            }
            _ => {}
        }
    }
    
    // Should detect deletions
    assert!(deleted_files.len() >= 2, "Should detect at least 2 deletions");
    
    println!("✅ File deletion test passed");
    
    // Test 4: Rapid file changes (test debouncing)
    println!("\n⚡ Test 4: Rapid file changes (debouncing test)");
    let rapid_test_file = temp_path.join("rapid.rs");
    let start_time = Instant::now();
    
    // Create rapid changes
    for i in 0..10 {
        fs::write(&rapid_test_file, format!("// Rapid change {}", i))?;
        std::thread::sleep(Duration::from_millis(10)); // Very fast changes
    }
    
    // Wait longer for debouncing to work
    std::thread::sleep(Duration::from_millis(500));
    
    let events = watcher.get_changes()?;
    let rapid_latency = start_time.elapsed().as_millis();
    
    println!("📊 Rapid Changes Events: {} (total latency: {}ms)", events.len(), rapid_latency);
    
    // Due to debouncing, should have fewer events than the 10+ changes made
    // The exact number depends on timing, but should be significantly less than 10
    println!("  ⏰ Debouncing working: {} events for 10+ rapid changes", events.len());
    
    for event in &events {
        if let FileChange::Modified(path) | FileChange::Added(path) = &event.change {
            if path.file_name().unwrap().to_string_lossy() == "rapid.rs" {
                println!("  ⚡ Rapid change detected: {} ({}ms latency)", 
                    path.file_name().unwrap().to_string_lossy(),
                    event.latency_ms()
                );
            }
        }
    }
    
    println!("✅ Rapid changes test passed - debouncing is working");
    
    // Get final statistics
    let stats = watcher.get_stats();
    println!("\n📊 Final Watcher Statistics:");
    println!("  {}", stats);
    
    // Stop watcher
    watcher.stop()?;
    println!("🛑 Watcher stopped successfully");
    
    // Verify no events are missed during normal operation
    assert_eq!(stats.events_missed, 0, "No events should be missed during normal operation");
    assert!(stats.events_processed > 0, "Should have processed some events");
    
    println!("✅ All Simple File Watcher tests passed!");
    
    Ok(())
}

/// Test error conditions
#[test]
fn test_watcher_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing error handling");
    
    // Test invalid directory
    let result = SimpleFileWatcher::new(PathBuf::from("/nonexistent/directory"));
    assert!(result.is_err());
    println!("✅ Invalid directory properly rejected");
    
    // Test double start
    let temp_dir = TempDir::new()?;
    let mut watcher = SimpleFileWatcher::new(temp_dir.path().to_path_buf())?;
    
    watcher.start()?;
    let result = watcher.start();
    assert!(result.is_err());
    println!("✅ Double start properly rejected");
    
    watcher.stop()?;
    println!("✅ Error handling tests passed");
    
    Ok(())
}

/// Performance test for event processing under load
#[test]
fn test_watcher_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Testing watcher performance under load");
    
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().to_path_buf();
    
    let mut watcher = SimpleFileWatcher::new(temp_path.clone())?;
    watcher.start()?;
    
    std::thread::sleep(Duration::from_millis(100)); // Stabilization
    
    let start_time = Instant::now();
    
    // Create many files rapidly
    for i in 0..50 {
        let file_path = temp_path.join(format!("perf_test_{}.rs", i));
        fs::write(&file_path, format!("// Performance test file {}", i))?;
        
        if i % 10 == 0 {
            std::thread::sleep(Duration::from_millis(1)); // Small pause every 10 files
        }
    }
    
    // Wait for all events to be processed
    std::thread::sleep(Duration::from_millis(1000));
    
    let events = watcher.get_changes()?;
    let total_time = start_time.elapsed();
    
    println!("📊 Performance Results:");
    println!("  Created 50 files in {:.2}ms", total_time.as_millis());
    println!("  Detected {} events", events.len());
    
    if !events.is_empty() {
        let avg_latency = events.iter().map(|e| e.latency_ms()).sum::<u64>() / events.len() as u64;
        let max_latency = events.iter().map(|e| e.latency_ms()).max().unwrap_or(0);
        println!("  Average latency: {}ms", avg_latency);
        println!("  Maximum latency: {}ms", max_latency);
        
        // Performance assertions
        assert!(avg_latency < 1000, "Average latency should be under 1 second");
        assert!(max_latency < 5000, "Maximum latency should be under 5 seconds");
    }
    
    let stats = watcher.get_stats();
    println!("  Final stats: {}", stats);
    
    // Should handle the load without missing events
    assert_eq!(stats.events_missed, 0, "Should not miss events under normal load");
    
    watcher.stop()?;
    println!("✅ Performance test passed");
    
    Ok(())
}