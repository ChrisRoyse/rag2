// LanceDB disabled
// pub mod watcher;
pub mod simple_watcher;

// LanceDB disabled
// pub use watcher::{GitWatcher, UpdateStats};
#[cfg(feature = "vectordb")]
// LanceDB disabled
// pub use watcher::WatchCommand;

// New simple file watcher exports - preferred implementation
pub use simple_watcher::{SimpleFileWatcher, FileEvent, FileChange, WatcherStats};
#[cfg(feature = "vectordb")]
pub use simple_watcher::{SimpleVectorUpdater, SimpleWatchCommand};