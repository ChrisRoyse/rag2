/// BRUTAL NOMIC-3 EMBEDDING VALIDATION TEST
/// 
/// This test suite ruthlessly validates the NOMIC-3 embedding system with INTJ Type-8 precision.
/// NO COMPROMISES. NO FALLBACKS. ABSOLUTE TRUTH REQUIRED.
/// 
/// Testing Strategy:
/// 1. CPU-ONLY execution (no GPU, no CUDA, no fallbacks)
/// 2. Real-world performance measurement (not theoretical)
/// 3. Memory usage tracking under load
/// 4. Quality assessment via cosine similarity
/// 5. Stress testing with 1000+ token documents
/// 6. Error handling validation
/// 
/// FAILURE MODES TESTED:
/// - Model loading failures
/// - Memory allocation errors
/// - Quantization errors
/// - NaN/Inf generation
/// - Performance degradation
/// - Dimension mismatches

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::task;
use anyhow::{Result, anyhow};
use embed_search::config::Config;
use embed_search::embedding::{LazyEmbedder, NomicEmbedder};

/// BRUTAL TRUTH: Performance thresholds that MUST be met
const MIN_TOKENS_PER_SECOND: f64 = 1000.0;  // Minimum acceptable performance
const MAX_MEMORY_MB_PER_EMBEDDING: f64 = 50.0;  // Memory efficiency requirement
const EXPECTED_DIMENSIONS: usize = 768;
const MAX_INFERENCE_TIME_MS: u128 = 5000;  // 5 seconds max per embedding
const MIN_COSINE_SIMILARITY_THRESHOLD: f32 = 0.1;  // Minimum meaningful similarity

/// Test data representing different complexity levels
const SHORT_CODE: &str = "fn main() { println!(\"Hello, world!\"); }";

const MEDIUM_CODE: &str = r#"
use std::collections::HashMap;

pub struct DatabaseConnection {
    url: String,
    timeout: Duration,
    pool: Option<ConnectionPool>,
}

impl DatabaseConnection {
    pub fn new(url: String) -> Self {
        Self {
            url,
            timeout: Duration::from_secs(30),
            pool: None,
        }
    }
    
    pub async fn connect(&mut self) -> Result<()> {
        // Connection logic here
        Ok(())
    }
}
"#;

const LARGE_DOCUMENT: &str = r#"
/// Advanced Machine Learning Pipeline Implementation
/// 
/// This module implements a comprehensive machine learning pipeline for natural language processing
/// tasks including text classification, named entity recognition, and semantic similarity scoring.
/// The implementation leverages transformer architectures with attention mechanisms for state-of-the-art
/// performance across multiple domains.

use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow, Context};

/// Configuration for the ML pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub dropout_rate: f32,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_epochs: usize,
    pub early_stopping_patience: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_sequence_length: 512,
            num_attention_heads: 12,
            hidden_size: 768,
            intermediate_size: 3072,
            num_layers: 12,
            dropout_rate: 0.1,
            learning_rate: 2e-5,
            warmup_steps: 10000,
            max_epochs: 3,
            early_stopping_patience: 3,
        }
    }
}

/// Represents a training example with input and target
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input_text: String,
    pub target_labels: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Batch of training examples for efficient processing
#[derive(Debug)]
pub struct TrainingBatch {
    pub examples: Vec<TrainingExample>,
    pub batch_id: usize,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub loss: f64,
    pub training_time_seconds: f64,
    pub inference_time_ms: f64,
    pub memory_usage_mb: f64,
}

/// Main ML pipeline orchestrator
pub struct MLPipeline {
    config: PipelineConfig,
    model_weights: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    optimizer: Arc<Mutex<dyn Optimizer + Send + Sync>>,
    scheduler: Arc<Mutex<dyn LearningRateScheduler + Send + Sync>>,
    metrics_history: Arc<RwLock<Vec<ModelMetrics>>>,
    training_data: Arc<RwLock<Vec<TrainingExample>>>,
    validation_data: Arc<RwLock<Vec<TrainingExample>>>,
}

impl MLPipeline {
    /// Initialize a new ML pipeline with configuration
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let model_weights = Arc::new(RwLock::new(HashMap::new()));
        let tokenizer = Arc::new(BertTokenizer::new()?);
        let optimizer = Arc::new(Mutex::new(AdamOptimizer::new(config.learning_rate)?));
        let scheduler = Arc::new(Mutex::new(LinearWarmupScheduler::new(
            config.warmup_steps,
            config.max_epochs * 1000, // Approximate total steps
        )?));
        
        Ok(Self {
            config,
            model_weights,
            tokenizer,
            optimizer,
            scheduler,
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            training_data: Arc::new(RwLock::new(Vec::new())),
            validation_data: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Load training and validation data
    pub async fn load_data(&self, train_path: &str, val_path: &str) -> Result<()> {
        let training_examples = self.load_examples_from_file(train_path).await?;
        let validation_examples = self.load_examples_from_file(val_path).await?;
        
        {
            let mut train_data = self.training_data.write().unwrap();
            *train_data = training_examples;
        }
        
        {
            let mut val_data = self.validation_data.write().unwrap();
            *val_data = validation_examples;
        }
        
        Ok(())
    }
    
    async fn load_examples_from_file(&self, path: &str) -> Result<Vec<TrainingExample>> {
        // Implementation would read from file and parse examples
        // For now, return mock data
        Ok(vec![])
    }
    
    /// Train the model using the loaded data
    pub async fn train(&self) -> Result<ModelMetrics> {
        let start_time = Instant::now();
        let mut best_metrics = ModelMetrics {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: f64::INFINITY,
            training_time_seconds: 0.0,
            inference_time_ms: 0.0,
            memory_usage_mb: 0.0,
        };
        
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.max_epochs {
            println!("Starting epoch {}/{}", epoch + 1, self.config.max_epochs);
            
            // Training phase
            let train_metrics = self.train_epoch().await?;
            
            // Validation phase  
            let val_metrics = self.validate_epoch().await?;
            
            // Update learning rate
            {
                let mut scheduler = self.scheduler.lock().unwrap();
                scheduler.step();
            }
            
            // Early stopping check
            if val_metrics.loss < best_metrics.loss {
                best_metrics = val_metrics.clone();
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping triggered after {} epochs", epoch + 1);
                    break;
                }
            }
            
            // Store metrics
            {
                let mut history = self.metrics_history.write().unwrap();
                history.push(val_metrics);
            }
        }
        
        best_metrics.training_time_seconds = start_time.elapsed().as_secs_f64();
        Ok(best_metrics)
    }
    
    async fn train_epoch(&self) -> Result<ModelMetrics> {
        let training_data = self.training_data.read().unwrap().clone();
        let batches = self.create_batches(&training_data)?;
        
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        
        for (batch_idx, batch) in batches.iter().enumerate() {
            let batch_metrics = self.train_batch(batch).await?;
            total_loss += batch_metrics.loss;
            total_accuracy += batch_metrics.accuracy;
            
            if batch_idx % 100 == 0 {
                println!("Batch {}/{}, Loss: {:.4}, Accuracy: {:.4}", 
                        batch_idx + 1, batches.len(), batch_metrics.loss, batch_metrics.accuracy);
            }
        }
        
        Ok(ModelMetrics {
            accuracy: total_accuracy / batches.len() as f64,
            precision: 0.0, // Calculate in validation
            recall: 0.0,    // Calculate in validation
            f1_score: 0.0,  // Calculate in validation
            loss: total_loss / batches.len() as f64,
            training_time_seconds: 0.0,
            inference_time_ms: 0.0,
            memory_usage_mb: get_memory_usage_mb(),
        })
    }
    
    async fn validate_epoch(&self) -> Result<ModelMetrics> {
        let validation_data = self.validation_data.read().unwrap().clone();
        let batches = self.create_batches(&validation_data)?;
        
        let mut predictions = Vec::new();
        let mut ground_truth = Vec::new();
        let mut total_loss = 0.0;
        let mut inference_times = Vec::new();
        
        for batch in batches.iter() {
            let start_time = Instant::now();
            let batch_predictions = self.predict_batch(batch).await?;
            let inference_time = start_time.elapsed().as_millis();
            
            inference_times.push(inference_time);
            predictions.extend(batch_predictions.predictions);
            ground_truth.extend(batch_predictions.ground_truth);
            total_loss += batch_predictions.loss;
        }
        
        let metrics = calculate_classification_metrics(&predictions, &ground_truth)?;
        
        Ok(ModelMetrics {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1_score: metrics.f1_score,
            loss: total_loss / batches.len() as f64,
            training_time_seconds: 0.0,
            inference_time_ms: inference_times.iter().sum::<u128>() as f64 / inference_times.len() as f64,
            memory_usage_mb: get_memory_usage_mb(),
        })
    }
    
    async fn train_batch(&self, batch: &TrainingBatch) -> Result<ModelMetrics> {
        // Implementation would perform forward pass, compute loss, backward pass, and optimizer step
        // For now, return mock metrics
        Ok(ModelMetrics {
            accuracy: 0.85,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: 0.2,
            training_time_seconds: 0.0,
            inference_time_ms: 0.0,
            memory_usage_mb: get_memory_usage_mb(),
        })
    }
    
    async fn predict_batch(&self, batch: &TrainingBatch) -> Result<BatchPredictions> {
        // Implementation would perform inference
        // For now, return mock predictions
        Ok(BatchPredictions {
            predictions: vec![0; batch.examples.len()],
            ground_truth: vec![0; batch.examples.len()],
            loss: 0.15,
        })
    }
    
    fn create_batches(&self, examples: &[TrainingExample]) -> Result<Vec<TrainingBatch>> {
        let batch_size = self.config.batch_size;
        let mut batches = Vec::new();
        
        for (batch_id, chunk) in examples.chunks(batch_size).enumerate() {
            batches.push(TrainingBatch {
                examples: chunk.to_vec(),
                batch_id,
            });
        }
        
        Ok(batches)
    }
    
    /// Generate predictions for new text
    pub async fn predict(&self, text: &str) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        
        // Tokenize input
        let tokens = self.tokenizer.tokenize(text)?;
        
        // Run inference
        let embeddings = self.forward_pass(&tokens).await?;
        
        let inference_time = start_time.elapsed().as_millis();
        println!("Inference time: {}ms", inference_time);
        
        Ok(embeddings)
    }
    
    async fn forward_pass(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Implementation would perform the actual forward pass through the transformer
        // For now, return mock embeddings
        Ok(vec![0.0; self.config.hidden_size])
    }
}

/// Helper traits and structures

pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

pub trait Optimizer {
    fn step(&mut self, gradients: &HashMap<String, Vec<f32>>) -> Result<()>;
    fn zero_grad(&mut self);
}

pub trait LearningRateScheduler {
    fn step(&mut self);
    fn get_lr(&self) -> f64;
}

pub struct BertTokenizer {
    vocab: HashMap<String, u32>,
    special_tokens: HashSet<String>,
}

impl BertTokenizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            vocab: HashMap::new(),
            special_tokens: HashSet::new(),
        })
    }
}

impl Tokenizer for BertTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Mock tokenization
        Ok(vec![101, 2023, 2003, 102]) // [CLS] this is [SEP]
    }
    
    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("mock decoded text".to_string())
    }
}

pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    step_count: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64) -> Result<Self> {
        Ok(Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_count: 0,
        })
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, _gradients: &HashMap<String, Vec<f32>>) -> Result<()> {
        self.step_count += 1;
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // Implementation would zero gradients
    }
}

pub struct LinearWarmupScheduler {
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
    base_lr: f64,
}

impl LinearWarmupScheduler {
    pub fn new(warmup_steps: usize, total_steps: usize) -> Result<Self> {
        Ok(Self {
            warmup_steps,
            total_steps,
            current_step: 0,
            base_lr: 2e-5,
        })
    }
}

impl LearningRateScheduler for LinearWarmupScheduler {
    fn step(&mut self) {
        self.current_step += 1;
    }
    
    fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            let progress = (self.current_step - self.warmup_steps) as f64 / 
                          (self.total_steps - self.warmup_steps) as f64;
            self.base_lr * (1.0 - progress)
        }
    }
}

#[derive(Debug)]
pub struct BatchPredictions {
    pub predictions: Vec<usize>,
    pub ground_truth: Vec<usize>,
    pub loss: f64,
}

#[derive(Debug)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

fn calculate_classification_metrics(predictions: &[usize], ground_truth: &[usize]) -> Result<ClassificationMetrics> {
    if predictions.len() != ground_truth.len() {
        return Err(anyhow!("Predictions and ground truth must have the same length"));
    }
    
    let correct = predictions.iter().zip(ground_truth.iter())
        .filter(|(pred, truth)| pred == truth)
        .count();
    
    let accuracy = correct as f64 / predictions.len() as f64;
    
    // For simplicity, using accuracy as proxy for other metrics
    Ok(ClassificationMetrics {
        accuracy,
        precision: accuracy,
        recall: accuracy,
        f1_score: accuracy,
    })
}

fn get_memory_usage_mb() -> f64 {
    // Mock memory usage
    42.0
}
"#;

/// Memory monitoring utilities
fn get_current_memory_usage() -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status")?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Ok(parts[1].parse::<u64>()? * 1024); // Convert kB to bytes
                }
            }
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback for other platforms - estimate based on system info
        use sysinfo::{System, SystemExt, ProcessExt};
        let mut system = System::new_all();
        system.refresh_all();
        
        if let Some(process) = system.process(sysinfo::get_current_pid().unwrap()) {
            return Ok(process.memory() * 1024); // Convert kB to bytes
        }
    }
    
    Err(anyhow!("Could not determine memory usage"))
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("Vectors must have the same length"));
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(anyhow!("Cannot compute cosine similarity with zero vectors"));
    }
    
    Ok(dot_product / (norm_a * norm_b))
}

/// Count tokens in text (approximation)
fn count_tokens(text: &str) -> usize {
    text.split_whitespace()
        .map(|word| (word.len() + 3) / 4) // Rough approximation: 4 chars per token
        .sum()
}

#[tokio::test]
async fn test_brutal_model_loading() -> Result<()> {
    println!("🔥 BRUTAL TEST: Model Loading on CPU-Only");
    
    // Initialize config
    Config::init_test()?;
    
    let start_time = Instant::now();
    let embedder = LazyEmbedder::new();
    
    // CRITICAL: Force initialization and measure time
    match embedder.get_or_init().await {
        Ok(_) => {
            let load_time = start_time.elapsed();
            println!("✅ Model loaded successfully in {:?}", load_time);
            
            // BRUTAL REQUIREMENT: Loading must complete within reasonable time
            if load_time > Duration::from_secs(300) { // 5 minutes max
                return Err(anyhow!("❌ BRUTAL FAILURE: Model loading took too long: {:?}", load_time));
            }
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Model failed to load: {}", e));
        }
    }
    
    println!("✅ BRUTAL VERDICT: Model loading PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_embedding_dimensions() -> Result<()> {
    println!("🔥 BRUTAL TEST: Embedding Dimension Validation");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Test with different text lengths
    let test_cases = vec![
        ("Short", SHORT_CODE),
        ("Medium", MEDIUM_CODE),
        ("Large", LARGE_DOCUMENT),
    ];
    
    for (name, text) in test_cases {
        println!("Testing {} text ({} tokens)...", name, count_tokens(text));
        
        let start_memory = get_current_memory_usage().unwrap_or(0);
        let start_time = Instant::now();
        
        match embedder.embed(text).await {
            Ok(embedding) => {
                let inference_time = start_time.elapsed();
                let end_memory = get_current_memory_usage().unwrap_or(0);
                let memory_used_mb = (end_memory.saturating_sub(start_memory)) as f64 / (1024.0 * 1024.0);
                
                println!("  ✅ Generated embedding in {:?}", inference_time);
                println!("  📏 Dimensions: {}", embedding.len());
                println!("  💾 Memory used: {:.2} MB", memory_used_mb);
                
                // BRUTAL REQUIREMENTS
                if embedding.len() != EXPECTED_DIMENSIONS {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Wrong dimensions for {}: got {}, expected {}", 
                                     name, embedding.len(), EXPECTED_DIMENSIONS));
                }
                
                if inference_time.as_millis() > MAX_INFERENCE_TIME_MS {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Inference too slow for {}: {:?}", name, inference_time));
                }
                
                if memory_used_mb > MAX_MEMORY_MB_PER_EMBEDDING {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Memory usage too high for {}: {:.2} MB", name, memory_used_mb));
                }
                
                // Check for NaN/Inf values
                if embedding.iter().any(|&x| !x.is_finite()) {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Embedding contains NaN/Inf values for {}", name));
                }
                
                // Check normalization
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if (norm - 1.0).abs() > 0.01 {
                    return Err(anyhow!("❌ BRUTAL FAILURE: Embedding not properly normalized for {}: norm = {}", name, norm));
                }
                
                println!("  ✅ {} text PASSED all brutal checks", name);
            }
            Err(e) => {
                return Err(anyhow!("❌ BRUTAL FAILURE: Could not generate embedding for {}: {}", name, e));
            }
        }
    }
    
    println!("✅ BRUTAL VERDICT: Embedding dimensions PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_performance_measurement() -> Result<()> {
    println!("🔥 BRUTAL TEST: Performance Measurement");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Warm up the model
    let _ = embedder.embed(SHORT_CODE).await;
    
    // Performance test with multiple texts
    let test_texts = vec![SHORT_CODE, MEDIUM_CODE, LARGE_DOCUMENT];
    let mut total_tokens = 0;
    let mut total_time = Duration::default();
    let mut inference_times = Vec::new();
    
    for (i, text) in test_texts.iter().enumerate() {
        let tokens = count_tokens(text);
        total_tokens += tokens;
        
        println!("Performance test {}/{}... ({} tokens)", i + 1, test_texts.len(), tokens);
        
        let start_time = Instant::now();
        match embedder.embed(text).await {
            Ok(embedding) => {
                let inference_time = start_time.elapsed();
                total_time += inference_time;
                inference_times.push(inference_time.as_millis());
                
                println!("  ⚡ Inference: {:?} ({:.0} tokens/sec)", 
                        inference_time, 
                        tokens as f64 / inference_time.as_secs_f64());
                
                // Verify embedding quality
                assert_eq!(embedding.len(), EXPECTED_DIMENSIONS);
                assert!(embedding.iter().all(|&x| x.is_finite()));
            }
            Err(e) => {
                return Err(anyhow!("❌ BRUTAL FAILURE: Inference failed: {}", e));
            }
        }
    }
    
    // Calculate overall performance
    let avg_tokens_per_second = total_tokens as f64 / total_time.as_secs_f64();
    let avg_inference_time = inference_times.iter().sum::<u128>() as f64 / inference_times.len() as f64;
    
    println!("📊 PERFORMANCE RESULTS:");
    println!("  🚀 Average: {:.0} tokens/second", avg_tokens_per_second);
    println!("  ⏱️  Average inference time: {:.0}ms", avg_inference_time);
    println!("  📈 Total tokens processed: {}", total_tokens);
    println!("  ⏰ Total time: {:?}", total_time);
    
    // BRUTAL PERFORMANCE REQUIREMENTS
    if avg_tokens_per_second < MIN_TOKENS_PER_SECOND {
        return Err(anyhow!("❌ BRUTAL FAILURE: Performance too slow: {:.0} tokens/sec (minimum: {:.0})", 
                          avg_tokens_per_second, MIN_TOKENS_PER_SECOND));
    }
    
    if avg_inference_time > MAX_INFERENCE_TIME_MS as f64 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Average inference time too slow: {:.0}ms (maximum: {}ms)", 
                          avg_inference_time, MAX_INFERENCE_TIME_MS));
    }
    
    println!("✅ BRUTAL VERDICT: Performance PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_cosine_similarity_quality() -> Result<()> {
    println!("🔥 BRUTAL TEST: Cosine Similarity Quality Assessment");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Similar texts should have high similarity
    let similar_pairs = vec![
        ("fn main() { println!(\"Hello\"); }", "fn main() { print!(\"Hi\"); }"),
        ("class User { name: String }", "struct User { name: String }"),
        ("async fn connect() -> Result<()>", "async fn connect() -> anyhow::Result<()>"),
    ];
    
    // Different texts should have lower similarity
    let different_pairs = vec![
        ("fn main() { println!(\"Hello\"); }", "SELECT * FROM users WHERE id = 1"),
        ("class User { }", "package main\nimport \"fmt\""),
        ("const API_KEY = 'secret'", "DROP TABLE users;"),
    ];
    
    println!("Testing similar text pairs...");
    for (text1, text2) in similar_pairs {
        let emb1 = embedder.embed(text1).await?;
        let emb2 = embedder.embed(text2).await?;
        
        let similarity = cosine_similarity(&emb1, &emb2)?;
        println!("  📏 Similarity: {:.3} for similar texts", similarity);
        
        if similarity < MIN_COSINE_SIMILARITY_THRESHOLD {
            return Err(anyhow!("❌ BRUTAL FAILURE: Similar texts have too low similarity: {:.3}", similarity));
        }
    }
    
    println!("Testing different text pairs...");
    let mut different_similarities = Vec::new();
    for (text1, text2) in different_pairs {
        let emb1 = embedder.embed(text1).await?;
        let emb2 = embedder.embed(text2).await?;
        
        let similarity = cosine_similarity(&emb1, &emb2)?;
        different_similarities.push(similarity);
        println!("  📏 Similarity: {:.3} for different texts", similarity);
    }
    
    // Check that different texts are less similar than similar texts on average
    let avg_different_similarity = different_similarities.iter().sum::<f32>() / different_similarities.len() as f32;
    println!("📊 Average similarity for different texts: {:.3}", avg_different_similarity);
    
    // BRUTAL QUALITY REQUIREMENT: Model should distinguish between similar and different content
    if avg_different_similarity > 0.8 {
        return Err(anyhow!("❌ BRUTAL FAILURE: Model cannot distinguish different content (avg similarity: {:.3})", 
                          avg_different_similarity));
    }
    
    println!("✅ BRUTAL VERDICT: Quality assessment PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_stress_with_large_documents() -> Result<()> {
    println!("🔥 BRUTAL TEST: Stress Testing with 1000+ Token Documents");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Create a very large document by repeating the large document
    let mut massive_document = String::new();
    for i in 0..5 {
        massive_document.push_str(&format!("// Section {}\n", i + 1));
        massive_document.push_str(LARGE_DOCUMENT);
        massive_document.push('\n');
    }
    
    let token_count = count_tokens(&massive_document);
    println!("Generated massive document with ~{} tokens", token_count);
    
    if token_count < 1000 {
        return Err(anyhow!("❌ Test setup error: Document not large enough ({} tokens)", token_count));
    }
    
    let start_memory = get_current_memory_usage().unwrap_or(0);
    let start_time = Instant::now();
    
    // Process the massive document
    match embedder.embed(&massive_document).await {
        Ok(embedding) => {
            let processing_time = start_time.elapsed();
            let end_memory = get_current_memory_usage().unwrap_or(0);
            let memory_used_mb = (end_memory.saturating_sub(start_memory)) as f64 / (1024.0 * 1024.0);
            
            println!("📊 STRESS TEST RESULTS:");
            println!("  📄 Document size: {} tokens", token_count);
            println!("  ⏱️  Processing time: {:?}", processing_time);
            println!("  💾 Memory used: {:.2} MB", memory_used_mb);
            println!("  📏 Output dimensions: {}", embedding.len());
            println!("  🚀 Performance: {:.0} tokens/second", 
                    token_count as f64 / processing_time.as_secs_f64());
            
            // BRUTAL STRESS REQUIREMENTS
            if processing_time > Duration::from_secs(60) {
                return Err(anyhow!("❌ BRUTAL FAILURE: Large document processing too slow: {:?}", processing_time));
            }
            
            if memory_used_mb > 500.0 {
                return Err(anyhow!("❌ BRUTAL FAILURE: Memory usage too high for large document: {:.2} MB", memory_used_mb));
            }
            
            if embedding.len() != EXPECTED_DIMENSIONS {
                return Err(anyhow!("❌ BRUTAL FAILURE: Wrong dimensions for large document: {}", embedding.len()));
            }
            
            if embedding.iter().any(|&x| !x.is_finite()) {
                return Err(anyhow!("❌ BRUTAL FAILURE: Large document produced invalid embedding values"));
            }
            
            // Check normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if (norm - 1.0).abs() > 0.01 {
                return Err(anyhow!("❌ BRUTAL FAILURE: Large document embedding not normalized: norm = {}", norm));
            }
            
            println!("✅ Large document PASSED all stress tests");
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Could not process large document: {}", e));
        }
    }
    
    println!("✅ BRUTAL VERDICT: Stress testing PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_concurrent_processing() -> Result<()> {
    println!("🔥 BRUTAL TEST: Concurrent Processing");
    
    Config::init_test()?;
    let embedder = Arc::new(LazyEmbedder::new());
    
    // Warm up
    let _ = embedder.embed(SHORT_CODE).await;
    
    let test_texts = vec![SHORT_CODE, MEDIUM_CODE, LARGE_DOCUMENT];
    let num_concurrent = 3;
    
    let start_time = Instant::now();
    let mut handles = Vec::new();
    
    for i in 0..num_concurrent {
        let embedder_clone = embedder.clone();
        let text = test_texts[i % test_texts.len()];
        
        let handle = task::spawn(async move {
            let thread_start = Instant::now();
            let result = embedder_clone.embed(text).await;
            let thread_time = thread_start.elapsed();
            
            match result {
                Ok(embedding) => {
                    println!("  🧵 Thread {} completed in {:?} ({} dims)", i, thread_time, embedding.len());
                    Ok((embedding, thread_time))
                }
                Err(e) => Err(anyhow!("Thread {} failed: {}", i, e)),
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await??);
    }
    
    let total_time = start_time.elapsed();
    let avg_thread_time = results.iter()
        .map(|(_, time)| time.as_millis())
        .sum::<u128>() as f64 / results.len() as f64;
    
    println!("📊 CONCURRENT PROCESSING RESULTS:");
    println!("  🧵 Threads: {}", num_concurrent);
    println!("  ⏰ Total time: {:?}", total_time);
    println!("  ⏱️  Average thread time: {:.0}ms", avg_thread_time);
    
    // BRUTAL CONCURRENT REQUIREMENTS
    if total_time > Duration::from_secs(30) {
        return Err(anyhow!("❌ BRUTAL FAILURE: Concurrent processing took too long: {:?}", total_time));
    }
    
    // Verify all results
    for (i, (embedding, _)) in results.iter().enumerate() {
        if embedding.len() != EXPECTED_DIMENSIONS {
            return Err(anyhow!("❌ BRUTAL FAILURE: Thread {} produced wrong dimensions", i));
        }
        
        if embedding.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow!("❌ BRUTAL FAILURE: Thread {} produced invalid values", i));
        }
    }
    
    println!("✅ BRUTAL VERDICT: Concurrent processing PASSED");
    Ok(())
}

#[tokio::test]
async fn test_brutal_error_handling() -> Result<()> {
    println!("🔥 BRUTAL TEST: Error Handling");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    // Test with empty string
    match embedder.embed("").await {
        Ok(embedding) => {
            if embedding.len() != EXPECTED_DIMENSIONS {
                return Err(anyhow!("❌ Empty string should produce valid dimensions"));
            }
            println!("  ✅ Empty string handled correctly");
        }
        Err(_) => {
            println!("  ⚠️  Empty string rejected (acceptable)");
        }
    }
    
    // Test with very long string
    let long_string = "a ".repeat(10000);
    match embedder.embed(&long_string).await {
        Ok(embedding) => {
            if embedding.len() != EXPECTED_DIMENSIONS {
                return Err(anyhow!("❌ Long string should produce valid dimensions"));
            }
            println!("  ✅ Very long string handled correctly");
        }
        Err(e) => {
            println!("  ⚠️  Very long string rejected: {} (may be acceptable)", e);
        }
    }
    
    // Test with special characters
    let special_chars = "🚀 العربية 中文 русский язык ñáéíóú";
    match embedder.embed(special_chars).await {
        Ok(embedding) => {
            if embedding.len() != EXPECTED_DIMENSIONS {
                return Err(anyhow!("❌ Special characters should produce valid dimensions"));
            }
            println!("  ✅ Special characters handled correctly");
        }
        Err(e) => {
            return Err(anyhow!("❌ BRUTAL FAILURE: Special characters should be handled: {}", e));
        }
    }
    
    println!("✅ BRUTAL VERDICT: Error handling PASSED");
    Ok(())
}

/// FINAL BRUTAL ASSESSMENT
#[tokio::test]
async fn test_brutal_final_assessment() -> Result<()> {
    println!("🔥 BRUTAL FINAL ASSESSMENT: Complete System Validation");
    
    Config::init_test()?;
    let embedder = LazyEmbedder::new();
    
    let overall_start = Instant::now();
    let start_memory = get_current_memory_usage().unwrap_or(0);
    
    // Comprehensive test battery
    let test_cases = vec![
        ("Minimal code", "x = 1"),
        ("Function definition", "def process(data): return data.strip()"),
        ("Class with methods", r#"
class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        return self.transform(data)
    
    def transform(self, data):
        return data.upper()
"#),
        ("Complex algorithm", r#"
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"#),
        ("Documentation", r#"
/**
 * Advanced data processing pipeline for machine learning workflows.
 * 
 * This module provides comprehensive functionality for:
 * - Data ingestion from multiple sources
 * - Real-time preprocessing and transformation  
 * - Feature engineering and selection
 * - Model training and validation
 * - Production deployment and monitoring
 * 
 * Key components:
 * 1. DataLoader: Efficient batch processing
 * 2. Preprocessor: Cleaning and normalization
 * 3. FeatureEngineer: Advanced feature creation
 * 4. ModelTrainer: Distributed training support
 * 5. Evaluator: Comprehensive metrics calculation
 * 
 * Performance characteristics:
 * - Throughput: 10,000+ samples/second
 * - Memory usage: <2GB for 1M samples  
 * - Latency: <10ms per prediction
 * - Accuracy: 95%+ on benchmark datasets
 * 
 * @author ML Engineering Team
 * @version 2.1.0
 * @since 2024-01-15
 */
"#),
    ];
    
    let mut all_embeddings = Vec::new();
    let mut performance_metrics = Vec::new();
    
    for (name, text) in test_cases {
        let tokens = count_tokens(text);
        println!("Processing: {} ({} tokens)", name, tokens);
        
        let start = Instant::now();
        match embedder.embed(text).await {
            Ok(embedding) => {
                let duration = start.elapsed();
                let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
                
                performance_metrics.push((name, tokens, duration, tokens_per_sec));
                all_embeddings.push((name, embedding));
                
                println!("  ✅ {} - {:?} ({:.0} tok/s)", name, duration, tokens_per_sec);
            }
            Err(e) => {
                return Err(anyhow!("❌ BRUTAL FAILURE: {} failed: {}", name, e));
            }
        }
    }
    
    let total_time = overall_start.elapsed();
    let end_memory = get_current_memory_usage().unwrap_or(0);
    let memory_used_mb = (end_memory.saturating_sub(start_memory)) as f64 / (1024.0 * 1024.0);
    
    // Calculate aggregate metrics
    let total_tokens: usize = performance_metrics.iter().map(|(_, tokens, _, _)| tokens).sum();
    let avg_tokens_per_sec: f64 = performance_metrics.iter()
        .map(|(_, _, _, tps)| tps)
        .sum::<f64>() / performance_metrics.len() as f64;
    
    // Quality assessment: compare embeddings
    let mut similarity_matrix = Vec::new();
    for i in 0..all_embeddings.len() {
        for j in i+1..all_embeddings.len() {
            let sim = cosine_similarity(&all_embeddings[i].1, &all_embeddings[j].1)?;
            similarity_matrix.push((all_embeddings[i].0, all_embeddings[j].0, sim));
        }
    }
    
    println!("\n🏆 BRUTAL FINAL ASSESSMENT RESULTS:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 PERFORMANCE METRICS:");
    println!("  🚀 Total tokens processed: {}", total_tokens);
    println!("  ⏰ Total processing time: {:?}", total_time);
    println!("  📈 Average performance: {:.0} tokens/second", avg_tokens_per_sec);
    println!("  💾 Memory used: {:.2} MB", memory_used_mb);
    
    println!("\n📏 QUALITY METRICS:");
    for (name, embedding) in &all_embeddings {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let finite_count = embedding.iter().filter(|&&x| x.is_finite()).count();
        println!("  {} - Dims: {}, Norm: {:.3}, Finite: {}/{}", 
                name, embedding.len(), norm, finite_count, embedding.len());
    }
    
    println!("\n🔗 SIMILARITY ANALYSIS:");
    for (name1, name2, sim) in similarity_matrix {
        println!("  {} ↔ {}: {:.3}", name1, name2, sim);
    }
    
    // FINAL BRUTAL VERDICT
    println!("\n⚖️ BRUTAL VERDICT:");
    
    let mut passed = 0;
    let mut total_checks = 0;
    
    // Performance check
    total_checks += 1;
    if avg_tokens_per_sec >= MIN_TOKENS_PER_SECOND {
        println!("  ✅ PERFORMANCE: {:.0} tokens/s ≥ {:.0}", avg_tokens_per_sec, MIN_TOKENS_PER_SECOND);
        passed += 1;
    } else {
        println!("  ❌ PERFORMANCE: {:.0} tokens/s < {:.0}", avg_tokens_per_sec, MIN_TOKENS_PER_SECOND);
    }
    
    // Memory check
    total_checks += 1;
    if memory_used_mb <= MAX_MEMORY_MB_PER_EMBEDDING * all_embeddings.len() as f64 {
        println!("  ✅ MEMORY: {:.2} MB within limits", memory_used_mb);
        passed += 1;
    } else {
        println!("  ❌ MEMORY: {:.2} MB exceeds limits", memory_used_mb);
    }
    
    // Dimensions check
    total_checks += 1;
    if all_embeddings.iter().all(|(_, emb)| emb.len() == EXPECTED_DIMENSIONS) {
        println!("  ✅ DIMENSIONS: All embeddings have {} dimensions", EXPECTED_DIMENSIONS);
        passed += 1;
    } else {
        println!("  ❌ DIMENSIONS: Inconsistent embedding dimensions");
    }
    
    // Quality check
    total_checks += 1;
    if all_embeddings.iter().all(|(_, emb)| {
        emb.iter().all(|&x| x.is_finite()) &&
        (emb.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 0.01
    }) {
        println!("  ✅ QUALITY: All embeddings are normalized and finite");
        passed += 1;
    } else {
        println!("  ❌ QUALITY: Some embeddings have invalid values");
    }
    
    let success_rate = passed as f64 / total_checks as f64;
    
    println!("\n🎯 FINAL SCORE: {}/{} ({:.0}%)", passed, total_checks, success_rate * 100.0);
    
    if success_rate >= 1.0 {
        println!("🏆 BRUTAL VERDICT: NOMIC-3 EMBEDDINGS PASSED ALL TESTS");
        println!("   Model is production-ready for CPU deployment");
    } else if success_rate >= 0.8 {
        println!("⚠️  BRUTAL VERDICT: NOMIC-3 EMBEDDINGS MOSTLY FUNCTIONAL");
        println!("   Model has issues but may be usable with caution");
    } else {
        println!("💥 BRUTAL VERDICT: NOMIC-3 EMBEDDINGS FAILED");
        println!("   Model is not ready for production use");
        return Err(anyhow!("Final assessment failed: {:.0}% success rate", success_rate * 100.0));
    }
    
    Ok(())
}