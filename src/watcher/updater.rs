use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

use crate::search::BM25Searcher;
use super::events::{FileEvent, EventType};

pub struct IndexUpdater {
    searcher: Arc<RwLock<BM25Searcher>>,
}

impl IndexUpdater {
    pub fn new(
        searcher: Arc<RwLock<BM25Searcher>>,
        _rx: mpsc::UnboundedReceiver<FileEvent>,
    ) -> Self {
        Self {
            searcher,
        }
    }

    pub fn start(&self) {
        // This is a placeholder for starting the updater
        // In production, this would spawn a background task
        println!("Index updater started");
    }
    
    pub async fn process_events(&self, events: Vec<FileEvent>) {
        for event in events {
            if let Err(e) = self.process_event(event).await {
                eprintln!("Error processing event: {}", e);
            }
        }
    }

    async fn process_event(&self, event: FileEvent) -> Result<(), anyhow::Error> {
        // Clone the Arc to avoid holding the lock during async operations
        let searcher_arc = Arc::clone(&self.searcher);
        
        // BM25Engine methods need &mut self, so we need a write lock
        let mut searcher = searcher_arc.write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire searcher write lock: {}", e))?;
        
        match event.event_type {
            EventType::Created | EventType::Modified => {
                // Index the file using BM25Engine's process_single_file method
                searcher.process_single_file(&event.path).await?;
                println!("✅ Indexed {:?}", event.path);
            }
            EventType::Removed => {
                // Remove documents associated with this file path
                let removed_count = searcher.remove_documents_by_path(&event.path)?;
                
                if removed_count > 0 {
                    println!("🗑️ Removed {} documents for file: {:?}", removed_count, event.path);
                } else {
                    println!("ℹ️ No documents found to remove for file: {:?}", event.path);
                }
            }
        }
        
        Ok(())
    }
}