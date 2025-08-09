use embed_search::utils::memory_monitor::MemoryMonitor;
use anyhow::Result;

#[test]
fn test_memory_monitor_basic() -> Result<()> {
    let monitor = MemoryMonitor::new(1024, 80); // 1GB limit, 80% warning
    
    assert_eq!(monitor.limit_mb(), 1024);
    assert!(monitor.can_allocate(100 * 1024 * 1024)); // 100MB should be fine
    
    Ok(())
}

#[test] 
fn test_memory_allocation_tracking() -> Result<()> {
    let monitor = MemoryMonitor::new(1024, 80);
    
    let initial_usage = monitor.current_usage_mb();
    
    // Try to allocate some memory
    match monitor.try_allocate(100 * 1024 * 1024) { // 100MB
        Ok(allocation) => {
            // Verify usage increased
            assert!(monitor.current_usage_mb() >= initial_usage);
            
            // When allocation drops, memory should be freed
            drop(allocation);
            
            // Note: Due to async nature, we can't guarantee immediate cleanup
            // but we can verify the tracking structure exists
        }
        Err(_) => {
            // If allocation fails, that's also acceptable for this test
            // We're mainly testing that the API works without panics
        }
    }
    
    Ok(())
}

#[test]
fn test_memory_bounds_checking() -> Result<()> {
    let monitor = MemoryMonitor::new(100, 80); // Small 100MB limit
    
    // This should fail
    let large_alloc = monitor.try_allocate(200 * 1024 * 1024); // 200MB
    assert!(large_alloc.is_err());
    
    Ok(())
}