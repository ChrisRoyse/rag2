// LanceDB Vector Storage Implementation - BRUTAL TRUTH: FUNCTIONAL IMPLEMENTATION ONLY
// This module provides real LanceDB vector storage for production use

#[cfg(feature = "vectordb")]
use std::path::PathBuf;
#[cfg(feature = "vectordb")]
use std::sync::Arc;
#[cfg(feature = "vectordb")]
use anyhow::Result;
#[cfg(feature = "vectordb")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "vectordb")]
use lancedb::{Connection, Table};
#[cfg(feature = "vectordb")]
use arrow_array::{Float32Array, RecordBatch, StringArray, UInt64Array, UInt32Array};
#[cfg(feature = "vectordb")]
use arrow_schema::{DataType, Field, Schema};
#[cfg(feature = "vectordb")]
use futures::stream::StreamExt;
#[cfg(feature = "vectordb")]
use crate::chunking::Chunk;

#[cfg(feature = "vectordb")]
#[derive(Debug)]
pub enum LanceDBError {
    ConnectionError(String),
    TableError(String),
    InsertError(String),
    SearchError(String),
    SchemaError(String),
    InvalidDimension { expected: usize, actual: usize },
    NoResultsFound,
}

#[cfg(feature = "vectordb")]
impl std::fmt::Display for LanceDBError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LanceDBError::ConnectionError(msg) => write!(f, "LanceDB connection error: {}", msg),
            LanceDBError::TableError(msg) => write!(f, "LanceDB table error: {}", msg),
            LanceDBError::InsertError(msg) => write!(f, "LanceDB insert error: {}", msg),
            LanceDBError::SearchError(msg) => write!(f, "LanceDB search error: {}", msg),
            LanceDBError::SchemaError(msg) => write!(f, "LanceDB schema error: {}", msg),
            LanceDBError::InvalidDimension { expected, actual } => {
                write!(f, "Invalid embedding dimension: expected {}, got {}", expected, actual)
            }
            LanceDBError::NoResultsFound => write!(f, "No results found"),
        }
    }
}

#[cfg(feature = "vectordb")]
impl std::error::Error for LanceDBError {}

#[cfg(feature = "vectordb")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanceDBRecord {
    pub id: String,
    pub file_path: String,
    pub chunk_index: u32,
    pub content: String,
    pub embedding: Vec<f32>,
    pub start_line: u32,
    pub end_line: u32,
    pub timestamp: u64,
}

#[cfg(feature = "vectordb")]
#[derive(Debug, Clone)]
pub struct LanceDBConfig {
    pub db_path: PathBuf,
    pub table_name: String,
    pub embedding_dimension: usize,
    pub max_connections: u32,
}

#[cfg(feature = "vectordb")]
impl LanceDBConfig {
    pub fn new(db_path: PathBuf, embedding_dimension: usize) -> Self {
        Self {
            db_path,
            table_name: "embeddings".to_string(),
            embedding_dimension,
            max_connections: 10,
        }
    }

    pub fn with_table_name(mut self, name: String) -> Self {
        self.table_name = name;
        self
    }
}

#[cfg(feature = "vectordb")]
pub struct LanceDBStorage {
    connection: Arc<Connection>,
    table: Option<Table>,
    config: LanceDBConfig,
}

#[cfg(feature = "vectordb")]
impl LanceDBStorage {
    /// Create new LanceDB storage instance
    pub async fn new(config: LanceDBConfig) -> Result<Self, LanceDBError> {
        // Ensure parent directory exists
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| LanceDBError::ConnectionError(format!("Failed to create directory: {}", e)))?;
        }

        // Connect to LanceDB
        let connection = lancedb::connect(&config.db_path).await
            .map_err(|e| LanceDBError::ConnectionError(format!("Failed to connect to LanceDB: {}", e)))?;

        Ok(Self {
            connection: Arc::new(connection),
            table: None,
            config,
        })
    }

    /// Initialize or open the embeddings table
    pub async fn init_table(&mut self) -> Result<(), LanceDBError> {
        let schema = self.create_schema();
        
        // Try to open existing table first
        match self.connection.open_table(&self.config.table_name).await {
            Ok(table) => {
                self.table = Some(table);
                Ok(())
            }
            Err(_) => {
                // Table doesn't exist, create it with empty data
                let empty_batch = self.create_empty_record_batch(&schema)?;
                let table = self.connection
                    .create_table(&self.config.table_name, vec![empty_batch])
                    .await
                    .map_err(|e| LanceDBError::TableError(format!("Failed to create table: {}", e)))?;
                
                self.table = Some(table);
                Ok(())
            }
        }
    }

    /// Create Arrow schema for embeddings table
    fn create_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.config.embedding_dimension as i32,
                ),
                false,
            ),
            Field::new("start_line", DataType::UInt32, false),
            Field::new("end_line", DataType::UInt32, false),
            Field::new("timestamp", DataType::UInt64, false),
        ])
    }

    /// Create empty record batch for table initialization
    fn create_empty_record_batch(&self, schema: &Schema) -> Result<RecordBatch, LanceDBError> {
        use arrow_array::FixedSizeListArray;
        
        let id_array = StringArray::from(Vec::<String>::new());
        let file_path_array = StringArray::from(Vec::<String>::new());
        let chunk_index_array = UInt32Array::from(Vec::<u32>::new());
        let content_array = StringArray::from(Vec::<String>::new());
        let start_line_array = UInt32Array::from(Vec::<u32>::new());
        let end_line_array = UInt32Array::from(Vec::<u32>::new());
        let timestamp_array = UInt64Array::from(Vec::<u64>::new());

        // Create empty fixed-size list array for embeddings
        let values = Float32Array::from(Vec::<f32>::new());
        let embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.config.embedding_dimension as i32,
            Arc::new(values),
            None,
        );

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(id_array),
                Arc::new(file_path_array),
                Arc::new(chunk_index_array),
                Arc::new(content_array),
                Arc::new(embedding_array),
                Arc::new(start_line_array),
                Arc::new(end_line_array),
                Arc::new(timestamp_array),
            ],
        )
        .map_err(|e| LanceDBError::SchemaError(format!("Failed to create empty record batch: {}", e)))
    }

    /// Insert a single embedding record
    pub async fn insert_embedding(
        &self,
        file_path: &str,
        chunk_index: usize,
        chunk: &Chunk,
        embedding: Vec<f32>,
    ) -> Result<(), LanceDBError> {
        if embedding.len() != self.config.embedding_dimension {
            return Err(LanceDBError::InvalidDimension {
                expected: self.config.embedding_dimension,
                actual: embedding.len(),
            });
        }

        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        let record = LanceDBRecord {
            id: format!("{}-{}", file_path, chunk_index),
            file_path: file_path.to_string(),
            chunk_index: chunk_index as u32,
            content: chunk.content.clone(),
            embedding,
            start_line: chunk.start_line as u32,
            end_line: chunk.end_line as u32,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };

        self.insert_records(vec![record]).await
    }

    /// Insert multiple embedding records in batch
    pub async fn insert_batch(
        &self,
        embeddings_data: Vec<(&str, usize, Chunk, Vec<f32>)>,
    ) -> Result<(), LanceDBError> {
        if embeddings_data.is_empty() {
            return Ok(());
        }

        // Validate all embeddings have correct dimensions
        for (_, _, _, embedding) in &embeddings_data {
            if embedding.len() != self.config.embedding_dimension {
                return Err(LanceDBError::InvalidDimension {
                    expected: self.config.embedding_dimension,
                    actual: embedding.len(),
                });
            }
        }

        let records: Vec<LanceDBRecord> = embeddings_data
            .into_iter()
            .map(|(file_path, chunk_index, chunk, embedding)| LanceDBRecord {
                id: format!("{}-{}", file_path, chunk_index),
                file_path: file_path.to_string(),
                chunk_index: chunk_index as u32,
                content: chunk.content,
                embedding,
                start_line: chunk.start_line as u32,
                end_line: chunk.end_line as u32,
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
            })
            .collect();

        self.insert_records(records).await
    }

    /// Internal method to insert records
    async fn insert_records(&self, records: Vec<LanceDBRecord>) -> Result<(), LanceDBError> {
        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        let record_batch = self.create_record_batch(&records)?;
        
        table
            .add(vec![record_batch])
            .await
            .map_err(|e| LanceDBError::InsertError(format!("Failed to insert records: {}", e)))?;

        Ok(())
    }

    /// Create Arrow RecordBatch from LanceDBRecord vector
    fn create_record_batch(&self, records: &[LanceDBRecord]) -> Result<RecordBatch, LanceDBError> {
        use arrow_array::FixedSizeListArray;
        
        let mut ids = Vec::new();
        let mut file_paths = Vec::new();
        let mut chunk_indices = Vec::new();
        let mut contents = Vec::new();
        let mut embeddings_flat = Vec::new();
        let mut start_lines = Vec::new();
        let mut end_lines = Vec::new();
        let mut timestamps = Vec::new();

        for record in records {
            ids.push(record.id.clone());
            file_paths.push(record.file_path.clone());
            chunk_indices.push(record.chunk_index);
            contents.push(record.content.clone());
            embeddings_flat.extend_from_slice(&record.embedding);
            start_lines.push(record.start_line);
            end_lines.push(record.end_line);
            timestamps.push(record.timestamp);
        }

        let id_array = StringArray::from(ids);
        let file_path_array = StringArray::from(file_paths);
        let chunk_index_array = UInt32Array::from(chunk_indices);
        let content_array = StringArray::from(contents);
        let start_line_array = UInt32Array::from(start_lines);
        let end_line_array = UInt32Array::from(end_lines);
        let timestamp_array = UInt64Array::from(timestamps);

        // Create fixed-size list array for embeddings
        let values = Float32Array::from(embeddings_flat);
        let embedding_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.config.embedding_dimension as i32,
            Arc::new(values),
            None,
        );

        let schema = self.create_schema();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(id_array),
                Arc::new(file_path_array),
                Arc::new(chunk_index_array),
                Arc::new(content_array),
                Arc::new(embedding_array),
                Arc::new(start_line_array),
                Arc::new(end_line_array),
                Arc::new(timestamp_array),
            ],
        )
        .map_err(|e| LanceDBError::SchemaError(format!("Failed to create record batch: {}", e)))
    }

    /// Search for similar embeddings using vector similarity
    pub async fn search_similar(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<LanceDBRecord>, LanceDBError> {
        if query_embedding.len() != self.config.embedding_dimension {
            return Err(LanceDBError::InvalidDimension {
                expected: self.config.embedding_dimension,
                actual: query_embedding.len(),
            });
        }

        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        // Perform vector search using LanceDB's vector search capabilities
        let query_result = table
            .vector_search(query_embedding)
            .map_err(|e| LanceDBError::SearchError(format!("Vector search failed: {}", e)))?
            .limit(limit)
            .execute()
            .await
            .map_err(|e| LanceDBError::SearchError(format!("Search execution failed: {}", e)))?;

        // Convert results to LanceDBRecord
        let mut results = Vec::new();
        let mut stream = query_result.try_collect::<Vec<_>>().await
            .map_err(|e| LanceDBError::SearchError(format!("Failed to collect results: {}", e)))?;

        for batch in stream {
            results.extend(self.extract_records_from_batch(&batch)?);
        }

        Ok(results)
    }

    /// Extract LanceDBRecord instances from Arrow RecordBatch
    fn extract_records_from_batch(&self, batch: &RecordBatch) -> Result<Vec<LanceDBRecord>, LanceDBError> {
        use arrow_array::FixedSizeListArray;
        
        let id_array = batch.column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid id column type".to_string()))?;

        let file_path_array = batch.column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid file_path column type".to_string()))?;

        let chunk_index_array = batch.column(2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid chunk_index column type".to_string()))?;

        let content_array = batch.column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid content column type".to_string()))?;

        let embedding_array = batch.column(4)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid embedding column type".to_string()))?;

        let start_line_array = batch.column(5)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid start_line column type".to_string()))?;

        let end_line_array = batch.column(6)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid end_line column type".to_string()))?;

        let timestamp_array = batch.column(7)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| LanceDBError::SchemaError("Invalid timestamp column type".to_string()))?;

        let mut records = Vec::new();

        for row in 0..batch.num_rows() {
            let id = id_array.value(row).to_string();
            let file_path = file_path_array.value(row).to_string();
            let chunk_index = chunk_index_array.value(row);
            let content = content_array.value(row).to_string();
            let start_line = start_line_array.value(row);
            let end_line = end_line_array.value(row);
            let timestamp = timestamp_array.value(row);

            // Extract embedding from fixed-size list
            let embedding_list = embedding_array.value(row);
            let embedding_values = embedding_list
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| LanceDBError::SchemaError("Invalid embedding values type".to_string()))?;

            let embedding: Vec<f32> = (0..embedding_values.len())
                .map(|i| embedding_values.value(i))
                .collect();

            records.push(LanceDBRecord {
                id,
                file_path,
                chunk_index,
                content,
                embedding,
                start_line,
                end_line,
                timestamp,
            });
        }

        Ok(records)
    }

    /// Delete embeddings by file path
    pub async fn delete_by_file(&self, file_path: &str) -> Result<(), LanceDBError> {
        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        let filter = format!("file_path = '{}'", file_path);
        table
            .delete(&filter)
            .await
            .map_err(|e| LanceDBError::TableError(format!("Failed to delete records: {}", e)))?;

        Ok(())
    }

    /// Get count of stored embeddings
    pub async fn count(&self) -> Result<usize, LanceDBError> {
        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        let count_result = table
            .query()
            .execute()
            .await
            .map_err(|e| LanceDBError::SearchError(format!("Count query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| LanceDBError::SearchError(format!("Failed to collect count results: {}", e)))?;

        let total_rows = count_result.iter().map(|batch| batch.num_rows()).sum();
        Ok(total_rows)
    }

    /// Clear all embeddings from table
    pub async fn clear_all(&self) -> Result<(), LanceDBError> {
        let table = self.table.as_ref()
            .ok_or_else(|| LanceDBError::TableError("Table not initialized".to_string()))?;

        // Delete all records
        table
            .delete("TRUE")  // LanceDB SQL-like syntax to delete all
            .await
            .map_err(|e| LanceDBError::TableError(format!("Failed to clear table: {}", e)))?;

        Ok(())
    }

    /// Get table statistics
    pub async fn get_stats(&self) -> Result<LanceDBStats, LanceDBError> {
        let count = self.count().await?;
        
        Ok(LanceDBStats {
            total_embeddings: count,
            embedding_dimension: self.config.embedding_dimension,
            table_name: self.config.table_name.clone(),
        })
    }
}

#[cfg(feature = "vectordb")]
#[derive(Debug, Clone)]
pub struct LanceDBStats {
    pub total_embeddings: usize,
    pub embedding_dimension: usize,
    pub table_name: String,
}

// Implement conversion from simple_vectordb::EmbeddingRecord for compatibility
#[cfg(feature = "vectordb")]
impl From<crate::storage::simple_vectordb::EmbeddingRecord> for LanceDBRecord {
    fn from(record: crate::storage::simple_vectordb::EmbeddingRecord) -> Self {
        Self {
            id: record.id,
            file_path: record.file_path,
            chunk_index: record.chunk_index,
            content: record.content,
            embedding: record.embedding,
            start_line: record.start_line as u32,
            end_line: record.end_line as u32,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        }
    }
}

#[cfg(all(test, feature = "vectordb"))]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::chunking::Chunk;

    #[tokio::test]
    async fn test_lancedb_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_lance.db");

        let config = LanceDBConfig::new(db_path, 768);
        let mut storage = LanceDBStorage::new(config).await.unwrap();
        storage.init_table().await.unwrap();

        // Test insertion
        let chunk = Chunk {
            content: "fn test() {}".to_string(),
            start_line: 1,
            end_line: 1,
        };

        let embedding = vec![0.1f32; 768];
        storage
            .insert_embedding("test.rs", 0, &chunk, embedding.clone())
            .await
            .unwrap();

        // Test count
        let count = storage.count().await.unwrap();
        assert_eq!(count, 1);

        // Test search
        let query = vec![0.1f32; 768];
        let results = storage.search_similar(query, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "fn test() {}");
        assert_eq!(results[0].file_path, "test.rs");
    }

    #[tokio::test]
    async fn test_lancedb_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_batch.db");

        let config = LanceDBConfig::new(db_path, 768);
        let mut storage = LanceDBStorage::new(config).await.unwrap();
        storage.init_table().await.unwrap();

        // Prepare batch data
        let chunk1 = Chunk {
            content: "fn test1() {}".to_string(),
            start_line: 1,
            end_line: 1,
        };
        let chunk2 = Chunk {
            content: "fn test2() {}".to_string(),
            start_line: 3,
            end_line: 3,
        };

        let embedding1 = vec![1.0f32; 768];
        let embedding2 = vec![0.5f32; 768];

        let batch_data = vec![
            ("test.rs", 0, chunk1, embedding1),
            ("test.rs", 1, chunk2, embedding2),
        ];

        // Test batch insertion
        storage.insert_batch(batch_data).await.unwrap();

        // Test count
        let count = storage.count().await.unwrap();
        assert_eq!(count, 2);

        // Test search returns multiple results
        let query = vec![0.8f32; 768];
        let results = storage.search_similar(query, 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_lancedb_dimension_validation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_validation.db");

        let config = LanceDBConfig::new(db_path, 768);
        let mut storage = LanceDBStorage::new(config).await.unwrap();
        storage.init_table().await.unwrap();

        let chunk = Chunk {
            content: "test".to_string(),
            start_line: 1,
            end_line: 1,
        };

        // Test invalid embedding dimension
        let wrong_embedding = vec![0.1f32; 512]; // Wrong dimension
        let result = storage
            .insert_embedding("test.rs", 0, &chunk, wrong_embedding)
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            LanceDBError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, 768);
                assert_eq!(actual, 512);
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }
}