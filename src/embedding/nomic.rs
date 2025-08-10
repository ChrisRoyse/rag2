#[cfg(feature = "ml")]
use once_cell::sync::OnceCell;
#[cfg(feature = "ml")]
use std::sync::Arc;
#[cfg(feature = "ml")]
use std::path::PathBuf;
#[cfg(feature = "ml")]
use anyhow::{Result, anyhow};
#[cfg(feature = "ml")]
use std::fs;
#[cfg(feature = "ml")]
use std::io::{Write, Read, Seek};
#[cfg(feature = "ml")]
use candle_core::{Device, Tensor};
#[cfg(feature = "ml")]
use candle_core::quantized::{gguf_file, GgmlDType};
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;
#[cfg(feature = "ml")]
// Removed memmap2::Mmap to prevent V8 heap issues in Node.js
#[cfg(feature = "ml")]
#[cfg(feature = "ml")]
use byteorder::{LittleEndian, ReadBytesExt};

#[cfg(feature = "ml")]
static GLOBAL_EMBEDDER: OnceCell<Arc<NomicEmbedder>> = OnceCell::new();

/// Simplified GGUF-based Nomic Embed model - token embeddings with L2 normalization only
#[cfg(feature = "ml")]
pub struct NomicEmbedder {
    tokenizer: Tokenizer,
    device: Device,
    dimensions: usize,
    cache: Option<Arc<crate::embedding::EmbeddingCache>>,
    token_embeddings: Tensor,
}

#[cfg(feature = "ml")]
impl NomicEmbedder {

    const MODEL_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code-v1-Q4_K_M-GGUF/resolve/main/nomic-embed-code.Q4_K_M.gguf";
    const TOKENIZER_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code-v1/resolve/main/tokenizer.json";
    const MODEL_SIZE: u64 = 4_378_000_000;
    const MODEL_FILENAME: &'static str = "nomic-embed-code.Q4_K_M.gguf";
    const TOKENIZER_FILENAME: &'static str = "tokenizer.json";
    const MAX_SEQUENCE_LENGTH: usize = 2048;
    const HIDDEN_SIZE: usize = 768;
    
    pub async fn get_global() -> Result<Arc<Self>, crate::error::EmbedError> {
        if let Some(embedder) = GLOBAL_EMBEDDER.get() {
            return Ok(embedder.clone());
        }
        
        let embedder = Arc::new(Self::new().await.map_err(|e| crate::error::EmbedError::Internal {
            message: format!("Failed to initialize NomicEmbedder: {}", e),
            backtrace: None,
        })?);
        match GLOBAL_EMBEDDER.set(embedder.clone()) {
            Ok(_) => Ok(embedder),
            Err(_) => Err(crate::error::EmbedError::Internal {
                message: "Global embedder was already initialized by another thread".to_string(),
                backtrace: None,
            }),
        }
    }
    
    pub async fn new() -> Result<Self> {
        // Ensure files are cached using async directly
        let (model_path, tokenizer_path) = Self::ensure_files_cached().await?;
        
        // Setup device (CPU for GGUF)
        let device = Device::Cpu;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Load GGUF model - simplified to extract only token embeddings
        let token_embeddings = Self::load_token_embeddings(&model_path, &device)?;
        
        // Initialize cache
        let cache = Some(Arc::new(
            crate::embedding::EmbeddingCache::new(100_000)
                .map_err(|e| anyhow!("Failed to initialize embedding cache: {}", e))?
        ));
        
        #[cfg(debug_assertions)]
        println!("✅ Token embeddings loaded: {:?}", token_embeddings.shape());
        
        Ok(Self {
            tokenizer,
            device,
            dimensions: Self::HIDDEN_SIZE,
            cache,
            token_embeddings,
        })
    }
    
    fn load_token_embeddings(model_path: &PathBuf, device: &Device) -> Result<Tensor> {
        let mut file = fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Find token embeddings tensor
        let (_, tensor_info) = content.tensor_infos.iter()
            .find(|(name, _)| *name == "token_embd.weight")
            .ok_or_else(|| anyhow!("Token embeddings not found in GGUF file"))?;
            
        // Load tensor data  
        let tensor_size = Self::calculate_tensor_size(tensor_info)?;
        file.seek(std::io::SeekFrom::Start(content.tensor_data_offset as u64))?;
        
        let mut tensor_data = vec![0u8; tensor_size];
        file.read_exact(&mut tensor_data)?;
        
        Self::dequantize_tensor(&tensor_data, tensor_info, device)
    }
    
    
    fn calculate_tensor_size(tensor_info: &gguf_file::TensorInfo) -> Result<usize> {
        let total_elements = tensor_info.shape.elem_count();
        
        let size = match tensor_info.ggml_dtype {
            GgmlDType::F32 => total_elements * 4,
            GgmlDType::F16 => total_elements * 2,
            GgmlDType::Q4K => (total_elements / 256) * 144,
            _ => return Err(anyhow!("Unsupported quantization: {:?}", tensor_info.ggml_dtype)),
        };
        
        Ok(size)
    }
    
    fn dequantize_tensor(data: &[u8], tensor_info: &gguf_file::TensorInfo, device: &Device) -> Result<Tensor> {
        let shape = &tensor_info.shape;
        let total_elements = shape.elem_count();
        
        let values = match tensor_info.ggml_dtype {
            GgmlDType::F32 => {
                let mut values = Vec::with_capacity(total_elements);
                let mut cursor = std::io::Cursor::new(data);
                for _ in 0..total_elements {
                    values.push(cursor.read_f32::<LittleEndian>()?);
                }
                values
            },
            GgmlDType::F16 => {
                let mut values = Vec::with_capacity(total_elements);
                let mut cursor = std::io::Cursor::new(data);
                for _ in 0..total_elements {
                    let f16_bits = cursor.read_u16::<LittleEndian>()?;
                    values.push(Self::f16_to_f32(f16_bits));
                }
                values
            },
            GgmlDType::Q4K => Self::dequantize_q4_k_m(data, total_elements)?,
            _ => return Err(anyhow!("Unsupported quantization: {:?}", tensor_info.ggml_dtype)),
        };
        
        // Create tensor from dequantized values
        Ok(Tensor::from_vec(values, shape.dims(), device)
            .map_err(|e| anyhow!("Failed to create tensor from values: {}", e))?)
    }
    
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;
        
        if exp == 0 {
            if frac == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Subnormal
                let val = (frac as f32) / 1024.0 / 16384.0;
                if sign == 1 { -val } else { val }
            }
        } else if exp == 0x1f {
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            let val = f32::from_bits(
                ((sign as u32) << 31) |
                (((exp as u32) + 127 - 15) << 23) |
                ((frac as u32) << 13)
            );
            val
        }
    }

    /// Extract a 6-bit value from the packed scales array at the specified index
    fn extract_6bit_value(scales: &[u8; 12], index: usize) -> u8 {
        // Q4K scales are packed as 6-bit values in 12 bytes
        // 8 scales and 8 mins = 16 6-bit values = 96 bits = 12 bytes
        let bit_offset = index * 6;
        let byte_start = bit_offset / 8;
        let bit_start = bit_offset % 8;
        
        if byte_start >= 11 {
            return 0;
        }
        
        if bit_start + 6 <= 8 {
            // Value fits within one byte
            (scales[byte_start] >> bit_start) & 0x3F
        } else if byte_start + 1 < 12 {
            // Value spans two bytes
            let low_bits = 8 - bit_start;
            let high_bits = 6 - low_bits;
            let low_part = (scales[byte_start] >> bit_start) & ((1 << low_bits) - 1);
            let high_part = scales[byte_start + 1] & ((1 << high_bits) - 1);
            low_part | (high_part << low_bits)
        } else {
            0
        }
    }

    /// Correct Q4_K_M dequantization using 256-element superblocks
    fn dequantize_q4_k_m(data: &[u8], total_elements: usize) -> Result<Vec<f32>> {
        const QK_K: usize = 256;  // Superblock size
        const K_SCALE_SIZE: usize = 12;  // Size of scales array
        const BLOCK_Q4_K_SIZE: usize = 2 + 2 + K_SCALE_SIZE + (QK_K / 2);  // 144 bytes
        
        let superblocks = (total_elements + QK_K - 1) / QK_K;
        let mut values = Vec::with_capacity(total_elements);
        
        for superblock_idx in 0..superblocks {
            let block_offset = superblock_idx * BLOCK_Q4_K_SIZE;
            
            // Bounds check for the entire superblock
            if block_offset + BLOCK_Q4_K_SIZE > data.len() {
                return Err(anyhow!("Insufficient data for Q4_K_M superblock {}: need {} bytes but only {} available. Model file is truncated or corrupted.", superblock_idx, BLOCK_Q4_K_SIZE, data.len() - block_offset));
            }
            
            // Extract superblock components
            let d_bits = u16::from_le_bytes([
                data[block_offset], 
                data[block_offset + 1]
            ]);
            let dmin_bits = u16::from_le_bytes([
                data[block_offset + 2], 
                data[block_offset + 3]
            ]);
            
            let d = Self::f16_to_f32(d_bits);
            let dmin = Self::f16_to_f32(dmin_bits);
            
            // Validate scales for NaN prevention
            if !d.is_finite() || !dmin.is_finite() {
                return Err(anyhow!("Invalid scales in Q4_K_M superblock {}: d={}, dmin={}. Model data is corrupted.", superblock_idx, d, dmin));
            }
            
            // Extract scales array (12 bytes)
            let scales_start = block_offset + 4;
            if scales_start + K_SCALE_SIZE > data.len() {
                return Err(anyhow!("Insufficient data for Q4_K_M scales array: need {} bytes but only {} available.", K_SCALE_SIZE, data.len() - scales_start));
            }
            let mut scales_array = [0u8; K_SCALE_SIZE];
            scales_array.copy_from_slice(&data[scales_start..scales_start + K_SCALE_SIZE]);
            
            // Extract quantized values array (128 bytes)
            let qs_start = scales_start + K_SCALE_SIZE;
            if qs_start + QK_K / 2 > data.len() {
                return Err(anyhow!("Insufficient data for Q4_K_M quantized values: need {} bytes but only {} available.", QK_K / 2, data.len() - qs_start));
            }
            let qs = &data[qs_start..qs_start + QK_K / 2];
            
            // Process each of the 8 blocks within this superblock
            for block_idx in 0..8 {
                // Extract 6-bit scale and min for this block
                let scale_bits = Self::extract_6bit_value(&scales_array, block_idx);
                let min_bits = Self::extract_6bit_value(&scales_array, block_idx + 8);
                
                let block_scale = d * (scale_bits as f32);
                let block_min = dmin * (min_bits as f32);
                
                // Validate block scale and min
                if !block_scale.is_finite() || !block_min.is_finite() {
                    return Err(anyhow!("Invalid block scales in Q4_K_M block {}: scale={}, min={}. Model computation failed.", block_idx, block_scale, block_min));
                }
                
                // Dequantize 32 weights in this block
                for weight_idx in 0..32 {
                    let global_idx = block_idx * 32 + weight_idx;
                    
                    if values.len() >= total_elements {
                        break;
                    }
                    
                    // Extract 4-bit quantized value
                    let byte_idx = global_idx / 2;
                    if byte_idx >= qs.len() {
                        return Err(anyhow!("Byte index {} out of bounds for quantized data length {}. Model data structure is invalid.", byte_idx, qs.len()));
                    }
                    
                    let is_high_nibble = (global_idx % 2) == 1;
                    let q4_value = if is_high_nibble {
                        (qs[byte_idx] >> 4) & 0x0F
                    } else {
                        qs[byte_idx] & 0x0F
                    };
                    
                    // Apply Q4_K_M dequantization formula: y = d * q + dmin * q_offset
                    let dequantized_weight = block_scale * (q4_value as f32) + block_min;
                    
                    // Validate the dequantized value
                    if !dequantized_weight.is_finite() {
                        return Err(anyhow!("Dequantized weight {} is not finite in Q4_K_M processing. Model computation failed.", dequantized_weight));
                    }
                    values.push(dequantized_weight);
                }
            }
        }
        
        // Ensure we have exactly the right number of elements
        if values.len() != total_elements {
            return Err(anyhow!("Q4_K_M dequantization produced {} elements but expected {}. Model processing failed.", values.len(), total_elements));
        }
        
        Ok(values)
    }
    
    
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Ok(Some(embedding)) = cache.get(text) {
                return Ok(embedding);
            }
        }
        
        // Tokenize and get embeddings
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        let input_ids = encoding.get_ids();
        let seq_len = input_ids.len().min(Self::MAX_SEQUENCE_LENGTH);
        let input_ids = &input_ids[..seq_len];
        
        if input_ids.is_empty() {
            return Err(anyhow!("Empty input after tokenization"));
        }
        
        // Get token embeddings and mean pool
        let input_tensor = Tensor::new(input_ids, &self.device)?;
        let embeddings = self.token_embeddings.index_select(&input_tensor, 0)?;
        let pooled = embeddings.mean(0)?;
        
        // L2 normalization  
        let output_vec = pooled.to_vec1::<f32>()?;
        
        // L2 normalize
        let norm = output_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding: Vec<f32> = if norm > 1e-9 {
            output_vec.iter().map(|x| x / norm).collect()
        } else {
            return Err(anyhow!("Zero norm embedding"));
        };
        
        // Cache result
        if let Some(cache) = &self.cache {
            cache.put(text, embedding.clone())?;
        }
        
        Ok(embedding)
    }
    
    
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter()
            .map(|text| self.embed(text))
            .collect()
    }
    
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    pub fn set_dimensions(&mut self, dims: usize) -> Result<()> {
        let valid_dims = [64, 128, 256, 512, 768];
        
        if !valid_dims.contains(&dims) {
            return Err(anyhow!("Invalid dimensions. Must be one of: {:?}", valid_dims));
        }
        
        self.dimensions = dims;
        Ok(())
    }
    
    
    async fn ensure_files_cached() -> Result<(PathBuf, PathBuf)> {
        let cache_dir = dirs::home_dir()
            .ok_or_else(|| anyhow!("Could not determine home directory. Set HOME environment variable."))?
            .join(".nomic");
        
        fs::create_dir_all(&cache_dir)?;
        
        let model_path = cache_dir.join(Self::MODEL_FILENAME);
        let tokenizer_path = cache_dir.join(Self::TOKENIZER_FILENAME);
        
        // Download model if needed
        if !model_path.exists() || fs::metadata(&model_path)?.len() < (Self::MODEL_SIZE as f64 * 0.95) as u64 {
            println!("📥 Downloading Nomic Embed Code v1 GGUF (Q4_K_M, ~4.3GB)...");
            println!("   Cache location: {:?}", model_path);
            Self::download_with_progress(Self::MODEL_URL, &model_path).await?;
            println!("✅ Model cached successfully at: {:?}", model_path);
        } else {
            println!("✅ Nomic model found in cache: {:?}", model_path);
            #[cfg(debug_assertions)]
            println!("   File size: {:.1}MB", fs::metadata(&model_path)?.len() as f64 / 1_048_576.0);
        }
        
        // Download tokenizer if needed
        if !tokenizer_path.exists() {
            println!("📥 Downloading tokenizer...");
            Self::download_file(Self::TOKENIZER_URL, &tokenizer_path).await?;
            println!("✅ Tokenizer cached successfully at: {:?}", tokenizer_path);
        } else {
            println!("✅ Tokenizer found in cache: {:?}", tokenizer_path);
        }
        
        Ok((model_path, tokenizer_path))
    }
    
    async fn download_file(url: &str, target: &PathBuf) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download file: {}", response.status()));
        }
        
        let content = response.bytes().await?;
        fs::write(target, content)?;
        
        Ok(())
    }
    
    async fn download_with_progress(url: &str, target: &PathBuf) -> Result<()> {
        use reqwest;
        use futures::StreamExt;
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()?;
            
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download model: {}", response.status()));
        }
        
        let total_size = response.content_length()
            .ok_or_else(|| anyhow!("Server did not provide content length for model download. Cannot track progress."))?;
        
        let mut file = fs::File::create(target)?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 {
                let progress = (downloaded as f64 / total_size as f64) * 100.0;
                print!("\r📥 Progress: {:.1}% ({:.1}MB / {:.1}MB)", 
                       progress, 
                       downloaded as f64 / 1_048_576.0,
                       total_size as f64 / 1_048_576.0);
                std::io::stdout().flush()?;
            }
        }
        println!();
        
        Ok(())
    }
}

#[cfg(all(test, feature = "ml"))]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_embedding_generation() {
        let embedder = NomicEmbedder::get_global().await.unwrap();
        
        let text = "def calculate_sum(a, b): return a + b";
        let embedding = embedder.embed(text).unwrap();
        
        // Check dimensions and normalization
        assert_eq!(embedding.len(), 768);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Should be L2 normalized");
    }
}

