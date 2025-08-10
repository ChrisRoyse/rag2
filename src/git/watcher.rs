use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use crossbeam_channel::{Receiver, Sender, bounded, select};

#[cfg(feature = "vectordb")]
use std::sync::RwLock;
#[cfg(feature = "vectordb")]
use crate::storage::lancedb_storage::LanceDBStorage;
#[cfg(feature = "vectordb")]
use crate::search::bm25::BM25Engine as BM25Searcher;

#[derive(Debug, Clone)]
pub enum FileChange {
    Modified(PathBuf),
    Added(PathBuf),
    Deleted(PathBuf),
}

#[derive(Debug, Clone)]
pub struct FileEvent {
    pub change: FileChange,
    pub timestamp: SystemTime,
    pub detected_at: Instant,
}

impl FileEvent {
    pub fn latency_ms(&self) -> u64 {
        self.detected_at.elapsed().as_millis() as u64
    }
}

/// Simple, robust file watcher using notify crate with proper debouncing
pub struct SimpleFileWatcher {
    repo_path: PathBuf,
    event_sender: Sender<Vec<FileEvent>>,
    event_receiver: Receiver<Vec<FileEvent>>,
    watcher_handle: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
    events_processed: Arc<AtomicU64>,
    events_missed: Arc<AtomicU64>,
}

impl SimpleFileWatcher {
    pub fn new(repo_path: PathBuf) -> Result<Self> {
        if !repo_path.exists() {
            return Err(anyhow!("Repository path does not exist: {}", repo_path.display()));
        }
        if !repo_path.is_dir() {
            return Err(anyhow!("Repository path is not a directory: {}", repo_path.display()));
        }

        let (event_sender, event_receiver) = bounded::<Vec<FileEvent>>(1000); // Large queue to prevent blocking
        
        Ok(Self {
            repo_path,
            event_sender,
            event_receiver,
            watcher_handle: None,
            running: Arc::new(AtomicBool::new(false)),
            events_processed: Arc::new(AtomicU64::new(0)),
            events_missed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Start the file watcher with proper debouncing and error handling
    pub fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(anyhow!("File watcher is already running"));
        }

        self.running.store(true, Ordering::Relaxed);
        let repo_path = self.repo_path.clone();
        let event_sender = self.event_sender.clone();
        let running = self.running.clone();
        let events_processed = self.events_processed.clone();
        let events_missed = self.events_missed.clone();

        let handle = std::thread::spawn(move || {
            if let Err(e) = Self::watch_loop(repo_path, event_sender, running, events_processed, events_missed) {
                eprintln!("CRITICAL FILE WATCHER ERROR: {}", e);
                eprintln!("File watching has stopped. Manual restart required.");
            }
        });

        self.watcher_handle = Some(handle);
        println!("✅ Simple file watcher started for: {}", self.repo_path.display());
        Ok(())
    }

    fn watch_loop(
        repo_path: PathBuf,
        event_sender: Sender<Vec<FileEvent>>,
        running: Arc<AtomicBool>,
        events_processed: Arc<AtomicU64>,
        events_missed: Arc<AtomicU64>,
    ) -> Result<()> {
        // Create debounced watcher with 100ms minimum debounce time
        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer(Duration::from_millis(100), tx)
            .map_err(|e| anyhow!("Failed to create file system debouncer: {}", e))?;

        // Add the repository path to watch
        debouncer.watcher()
            .watch(&repo_path, RecursiveMode::Recursive)
            .map_err(|e| anyhow!("Failed to start watching directory {}: {}", repo_path.display(), e))?;

        println!("🔍 Watching for changes in: {} (debounce: 100ms)", repo_path.display());

        let mut event_buffer: HashMap<PathBuf, FileEvent> = HashMap::new();
        let mut last_flush = Instant::now();
        let flush_interval = Duration::from_millis(150); // Batch events together

        while running.load(Ordering::Relaxed) {
            // Use select to handle both file events and flush timing
            let should_flush = last_flush.elapsed() >= flush_interval && !event_buffer.is_empty();
            
            if should_flush {
                Self::flush_events(&mut event_buffer, &event_sender, &events_processed, &events_missed)?;
                last_flush = Instant::now();
                continue;
            }

            // Wait for file system events with timeout
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(result) => {
                    match result {
                        Ok(events) => {
                            let detected_at = Instant::now();
                            for debounced_event in events {
                                if let Some(file_event) = Self::process_debounced_event(debounced_event, detected_at)? {
                                    // Only keep the latest event for each file to handle rapid changes
                                    let path = match &file_event.change {
                                        FileChange::Modified(p) | FileChange::Added(p) | FileChange::Deleted(p) => p.clone(),
                                    };
                                    event_buffer.insert(path, file_event);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("File watcher event processing error: {}", e);
                            events_missed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Normal timeout, continue loop
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("CRITICAL: File watcher channel disconnected");
                    break;
                }
            }
        }

        // Final flush on shutdown
        if !event_buffer.is_empty() {
            let _ = Self::flush_events(&mut event_buffer, &event_sender, &events_processed, &events_missed);
        }

        println!("🛑 File watcher stopped");
        Ok(())
    }

    fn process_debounced_event(event: notify_debouncer_mini::DebouncedEvent, detected_at: Instant) -> Result<Option<FileEvent>> {
        let path = event.path;
        
        // Only process code files as specified
        if !Self::is_tracked_file(&path) {
            return Ok(None);
        }

        let timestamp = SystemTime::now();
        let change = match event.kind {
            DebouncedEventKind::Create => FileChange::Added(path.clone()),
            DebouncedEventKind::Modify => FileChange::Modified(path.clone()),
            DebouncedEventKind::Remove => FileChange::Deleted(path.clone()),
            _ => return Ok(None), // Ignore other events
        };

        Ok(Some(FileEvent {
            change,
            timestamp,
            detected_at,
        }))
    }

    fn flush_events(
        event_buffer: &mut HashMap<PathBuf, FileEvent>,
        event_sender: &Sender<Vec<FileEvent>>,
        events_processed: &Arc<AtomicU64>,
        events_missed: &Arc<AtomicU64>,
    ) -> Result<()> {
        if event_buffer.is_empty() {
            return Ok(());
        }

        let events: Vec<FileEvent> = event_buffer.drain().map(|(_, event)| event).collect();
        let count = events.len() as u64;

        match event_sender.try_send(events) {
            Ok(()) => {
                events_processed.fetch_add(count, Ordering::Relaxed);
            }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                eprintln!("WARNING: Event queue full, {} events dropped", count);
                events_missed.fetch_add(count, Ordering::Relaxed);
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                return Err(anyhow!("Event receiver disconnected"));
            }
        }

        Ok(())
    }

    /// Check if file should be tracked based on extension
    fn is_tracked_file(path: &Path) -> bool {
        match path.extension().and_then(|s| s.to_str()) {
            Some(ext) => matches!(ext, "rs" | "py" | "js" | "ts"),
            None => false,
        }
    }

    /// Get pending file change events (non-blocking)
    pub fn get_changes(&self) -> Result<Vec<FileEvent>> {
        let mut all_events = Vec::new();
        
        // Drain all pending events from the queue
        while let Ok(events) = self.event_receiver.try_recv() {
            all_events.extend(events);
        }

        if !all_events.is_empty() {
            let total = all_events.len();
            let avg_latency = if total > 0 {
                all_events.iter().map(|e| e.latency_ms()).sum::<u64>() / total as u64
            } else {
                0
            };
            println!("📊 Processing {} file events (avg latency: {}ms)", total, avg_latency);
        }

        Ok(all_events)
    }

    /// Get pending changes with timeout
    pub fn get_changes_blocking(&self, timeout: Duration) -> Result<Vec<FileEvent>> {
        match self.event_receiver.recv_timeout(timeout) {
            Ok(events) => {
                let mut all_events = events;
                
                // Get any additional pending events
                while let Ok(more_events) = self.event_receiver.try_recv() {
                    all_events.extend(more_events);
                }

                if !all_events.is_empty() {
                    let total = all_events.len();
                    let avg_latency = if total > 0 {
                        all_events.iter().map(|e| e.latency_ms()).sum::<u64>() / total as u64
                    } else {
                        0
                    };
                    println!("📊 Processing {} file events (avg latency: {}ms)", total, avg_latency);
                }

                Ok(all_events)
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(Vec::new()),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow!("File watcher channel disconnected"))
            }
        }
    }

    /// Stop the file watcher
    pub fn stop(&mut self) -> Result<()> {
        if !self.running.load(Ordering::Relaxed) {
            return Ok(()); // Already stopped
        }

        self.running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.watcher_handle.take() {
            // Give the thread a moment to stop gracefully
            std::thread::sleep(Duration::from_millis(200));
            
            // Force join if still running
            match handle.join() {
                Ok(()) => println!("✅ File watcher stopped gracefully"),
                Err(_) => eprintln!("⚠️ File watcher thread panicked during shutdown"),
            }
        }

        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get watcher statistics
    pub fn get_stats(&self) -> WatcherStats {
        WatcherStats {
            events_processed: self.events_processed.load(Ordering::Relaxed),
            events_missed: self.events_missed.load(Ordering::Relaxed),
            is_running: self.is_running(),
            queue_size: self.event_receiver.len(),
        }
    }
}

impl Drop for SimpleFileWatcher {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            eprintln!("Error stopping file watcher: {}", e);
        }
    }
}

#[derive(Debug, Clone)]
pub struct WatcherStats {
    pub events_processed: u64,
    pub events_missed: u64,
    pub is_running: bool,
    pub queue_size: usize,
}

impl std::fmt::Display for WatcherStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Watcher Stats: {} events processed, {} missed, running: {}, queue: {}",
            self.events_processed,
            self.events_missed,
            self.is_running,
            self.queue_size
        )
    }
}

#[cfg(feature = "vectordb")]
pub struct SimpleVectorUpdater {
    searcher: Arc<BM25Searcher>,
    storage: Arc<RwLock<LanceDBStorage>>,
    stats: Arc<RwLock<UpdateStats>>,
}

#[cfg(feature = "vectordb")]
impl SimpleVectorUpdater {
    pub fn new(searcher: Arc<BM25Searcher>, storage: Arc<RwLock<LanceDBStorage>>) -> Self {
        Self {
            searcher,
            storage,
            stats: Arc::new(RwLock::new(UpdateStats::new())),
        }
    }

    pub async fn process_events(&self, events: Vec<FileEvent>) -> Result<UpdateStats> {
        if events.is_empty() {
            return Ok(UpdateStats::new());
        }

        let start_time = Instant::now();
        let mut stats = UpdateStats::new();

        // Process events by type for efficiency
        let mut deletions = Vec::new();
        let mut modifications = Vec::new();

        for event in events {
            match event.change {
                FileChange::Deleted(path) => deletions.push(path),
                FileChange::Modified(path) | FileChange::Added(path) => modifications.push(path),
            }
        }

        // Process deletions first (fast and must succeed)
        for path in deletions {
            match self.delete_file_embeddings(&path).await {
                Ok(()) => stats.deleted_files += 1,
                Err(e) => {
                    eprintln!("Failed to delete embeddings for {}: {}", path.display(), e);
                    stats.failed_files += 1;
                }
            }
        }

        // Process modifications/additions
        for path in modifications {
            // Delete old embeddings first, then re-index
            if let Err(e) = self.delete_file_embeddings(&path).await {
                eprintln!("Failed to delete old embeddings for {}: {}", path.display(), e);
            }

            match self.searcher.index_file(&path).await {
                Ok(()) => stats.updated_files += 1,
                Err(e) => {
                    eprintln!("Failed to re-index {}: {}", path.display(), e);
                    stats.failed_files += 1;
                }
            }
        }

        stats.total_time = start_time.elapsed();

        // Update global stats
        {
            let mut global_stats = self.stats.write().unwrap();
            global_stats.updated_files += stats.updated_files;
            global_stats.deleted_files += stats.deleted_files;
            global_stats.failed_files += stats.failed_files;
            global_stats.total_time += stats.total_time;
        }

        println!("📊 Vector update completed: {}", stats);
        Ok(stats)
    }

    async fn delete_file_embeddings(&self, file_path: &Path) -> Result<()> {
        let storage = self.storage.write().unwrap();
        storage.delete_by_file(&file_path.to_string_lossy()).await?;
        Ok(())
    }

    pub fn get_stats(&self) -> UpdateStats {
        self.stats.read().unwrap().clone()
    }
}

/// Combined watcher and updater for easy usage
#[cfg(feature = "vectordb")]
pub struct SimpleWatchCommand {
    watcher: SimpleFileWatcher,
    updater: SimpleVectorUpdater,
    update_handle: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

#[cfg(feature = "vectordb")]
impl SimpleWatchCommand {
    pub fn new(
        repo_path: PathBuf,
        searcher: Arc<BM25Searcher>,
        storage: Arc<RwLock<LanceDBStorage>>,
    ) -> Result<Self> {
        let watcher = SimpleFileWatcher::new(repo_path)?;
        let updater = SimpleVectorUpdater::new(searcher, storage);

        Ok(Self {
            watcher,
            updater,
            update_handle: None,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        // Start file watcher
        self.watcher.start()?;
        
        self.running.store(true, Ordering::Relaxed);
        
        // Start update processor
        let updater = self.updater.clone();
        let watcher_receiver = self.watcher.event_receiver.clone();
        let running = self.running.clone();

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create async runtime");
            
            while running.load(Ordering::Relaxed) {
                match watcher_receiver.recv_timeout(Duration::from_millis(500)) {
                    Ok(events) => {
                        // Process all available events
                        let mut all_events = events;
                        while let Ok(more_events) = watcher_receiver.try_recv() {
                            all_events.extend(more_events);
                        }

                        if !all_events.is_empty() {
                            if let Err(e) = rt.block_on(updater.process_events(all_events)) {
                                eprintln!("Vector update error: {}", e);
                            }
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Normal timeout, continue
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        println!("Update processor: watcher disconnected");
                        break;
                    }
                }
            }
        });

        self.update_handle = Some(handle);
        println!("✅ Simple watch command started with vector updates");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.update_handle.take() {
            let _ = handle.join();
        }
        
        self.watcher.stop()?;
        println!("✅ Simple watch command stopped");
        Ok(())
    }

    pub fn get_combined_stats(&self) -> (WatcherStats, UpdateStats) {
        (self.watcher.get_stats(), self.updater.get_stats())
    }
}

/// Statistics for vector updates
#[derive(Debug, Clone)]
pub struct UpdateStats {
    pub updated_files: usize,
    pub deleted_files: usize,
    pub failed_files: usize,
    pub total_time: Duration,
}

impl UpdateStats {
    pub fn new() -> Self {
        Self {
            updated_files: 0,
            deleted_files: 0,
            failed_files: 0,
            total_time: Duration::from_secs(0),
        }
    }
}

impl std::fmt::Display for UpdateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Updated {} files, deleted {} files, {} failures in {:.2}ms",
            self.updated_files,
            self.deleted_files,
            self.failed_files,
            self.total_time.as_millis()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_file_tracking() {
        // Test that only specified extensions are tracked
        assert!(SimpleFileWatcher::is_tracked_file(Path::new("test.rs")));
        assert!(SimpleFileWatcher::is_tracked_file(Path::new("test.py")));
        assert!(SimpleFileWatcher::is_tracked_file(Path::new("test.js")));
        assert!(SimpleFileWatcher::is_tracked_file(Path::new("test.ts")));
        
        // These should NOT be tracked
        assert!(!SimpleFileWatcher::is_tracked_file(Path::new("test.md")));
        assert!(!SimpleFileWatcher::is_tracked_file(Path::new("test.txt")));
        assert!(!SimpleFileWatcher::is_tracked_file(Path::new("test.go")));
        assert!(!SimpleFileWatcher::is_tracked_file(Path::new("Cargo.toml")));
    }

    #[test]
    fn test_watcher_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let watcher = SimpleFileWatcher::new(temp_dir.path().to_path_buf())?;
        
        assert!(!watcher.is_running());
        assert_eq!(watcher.get_stats().events_processed, 0);
        
        Ok(())
    }

    #[test]
    fn test_invalid_path_rejection() {
        let result = SimpleFileWatcher::new(PathBuf::from("/nonexistent/path"));
        assert!(result.is_err());
    }
}