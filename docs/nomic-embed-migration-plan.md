# Comprehensive Migration Plan: nomic-embed-text to nomic-embed-code

## Executive Summary

This document provides a detailed strategy for migrating from `nomic-embed-text` to `nomic-embed-code` embeddings across the entire codebase. The migration involves significant changes to URLs, filenames, model specifications, cache cleanup procedures, and potential compatibility risks.

## Critical Discovery: Major Architecture Differences

### Model Size and Architecture Changes
- **Current (nomic-embed-text)**: ~84MB Q4_K_M model with 768 dimensions
- **Target (nomic-embed-code)**: 4.38GB Q4_K_M model with 7B parameters
- **Impact**: 50x larger model requiring significant system resources

### Infrastructure Requirements
- **Storage**: Additional 4.3GB disk space required
- **Memory**: Significantly higher RAM usage during inference
- **Processing**: Longer loading times and inference latency
- **Compatibility**: Different tokenizer and model architecture

## 1. Code Location Analysis

### Files Requiring Changes

1. **`/home/cabdru/rag/src/embedding/nomic.rs`** (Lines 103-106)
   - Current URLs point to nomic-embed-text-v1.5
   - Model constants need complete update

2. **`/home/cabdru/rag/src/config/safe_config.rs`** (Line 96)
   - Test configuration references old model path

3. **`/home/cabdru/rag/src/config/mod.rs`** (Line 135)
   - Model name configuration string

4. **`/home/cabdru/rag/src/embedding/streaming_nomic_integration.rs`** (Line 538)
   - Hardcoded path to old model

5. **`/home/cabdru/rag/.gitignore`** (Line 60)
   - Ignore pattern needs updating

## 2. Required URL and Filename Changes

### Current Configuration
```rust
const MODEL_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf";
const TOKENIZER_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";
const MODEL_FILENAME: &'static str = "nomic-embed-text-v1.5.Q4_K_M.gguf";
```

### New Configuration Required
```rust
const MODEL_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code-GGUF/resolve/main/nomic-embed-code.Q4_K_M.gguf";
const TOKENIZER_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code/resolve/main/tokenizer.json";
const MODEL_FILENAME: &'static str = "nomic-embed-code.Q4_K_M.gguf";
const MODEL_SIZE: u64 = 4_380_000_000;  // ~4.38GB
```

### Model Specification Changes
```rust
// OLD
model_name: "nomic-ai/nomic-embed-text-v1.5".to_string(),

// NEW  
model_name: "nomic-ai/nomic-embed-code".to_string(),
```

## 3. Integration with Existing nomic-embed-code.Q4_K_M.gguf

### Current Status
- File exists at `/home/cabdru/rag/model/nomic-embed-code.Q4_K_M.gguf` (4.1GB)
- File size matches expected Q4_K_M quantization
- Ready for integration

### Path Integration Required
```rust
// Update model path references
let model_path = "model/nomic-embed-code.Q4_K_M.gguf";  // Use existing file
```

## 4. llama.cpp Integration Requirements

### Dependencies Analysis
The codebase uses **Candle framework**, not llama.cpp directly:
- `candle-core` for tensor operations
- `candle-nn` for neural network layers
- `candle-transformers` for transformer models

### Required Changes for 7B Model
1. **Memory Management**: Streaming loader implementation exists
2. **Quantization Support**: Q4_K_M dequantization implemented
3. **Architecture Updates**: Need transformer layer adjustments for larger model

## 5. Cache Cleanup and Migration Procedures

### Cache Directory Structure
```bash
~/.nomic/
├── nomic-embed-text-v1.5.Q4_K_M.gguf    # OLD - 84MB
├── tokenizer.json                        # OLD - for text model
└── [to be created]
    ├── nomic-embed-code.Q4_K_M.gguf     # NEW - 4.38GB
    └── tokenizer.json                    # NEW - for code model
```

### Cache Migration Strategy
1. **Keep Old Cache**: Preserve during transition for rollback
2. **Parallel Download**: Download new model alongside old
3. **Validation**: Verify new model before cleanup
4. **Cleanup**: Remove old files only after successful migration

### Cleanup Script
```bash
#!/bin/bash
# Cache cleanup after successful migration
CACHE_DIR="$HOME/.nomic"
OLD_MODEL="nomic-embed-text-v1.5.Q4_K_M.gguf"
OLD_TOKENIZER_BACKUP="tokenizer_text.json"

if [ -f "$CACHE_DIR/$OLD_MODEL" ]; then
    echo "Backing up old model..."
    mv "$CACHE_DIR/$OLD_MODEL" "$CACHE_DIR/${OLD_MODEL}.backup"
fi

if [ -f "$CACHE_DIR/tokenizer.json" ]; then
    echo "Backing up old tokenizer..."
    mv "$CACHE_DIR/tokenizer.json" "$CACHE_DIR/$OLD_TOKENIZER_BACKUP"
fi
```

## 6. Testing and Validation Requirements

### Pre-Migration Tests
1. **Model Loading Test**: Verify existing nomic-embed-code.Q4_K_M.gguf loads correctly
2. **Memory Usage Test**: Confirm system can handle 7B model
3. **Embedding Generation Test**: Validate output format and dimensions
4. **Performance Benchmark**: Measure inference time vs. old model

### Post-Migration Validation
1. **Functionality Tests**: All embedding operations work
2. **Integration Tests**: MCP server and streaming components function
3. **Regression Tests**: No breaking changes in API
4. **Performance Tests**: Acceptable latency for use case

### Test Script Template
```rust
#[tokio::test]
async fn test_nomic_code_migration() {
    // Test model loading
    let embedder = NomicEmbedder::new().await.expect("Model loading failed");
    
    // Test embedding generation
    let code_text = "def hello_world(): return 'Hello, World!'";
    let embedding = embedder.embed(code_text).expect("Embedding generation failed");
    
    // Validate dimensions (likely different from 768)
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    
    // Test code-specific functionality
    let query = "Represent this query for searching relevant code: function definition";
    let query_embedding = embedder.embed(query).expect("Query embedding failed");
    
    // Validate similarity
    let similarity = cosine_similarity(&embedding, &query_embedding);
    assert!(similarity > 0.0, "Should have positive similarity");
}
```

## 7. Compatibility Issues and Breaking Changes

### High-Risk Changes

#### 1. Model Architecture Differences
- **Risk**: 7B parameter model vs. smaller text model
- **Impact**: Different embedding dimensions, inference patterns
- **Mitigation**: Update all dimension constants and tensor operations

#### 2. Memory Requirements
- **Risk**: 50x larger model may cause OOM errors
- **Impact**: System instability, especially in constrained environments
- **Mitigation**: Implement memory monitoring and graceful degradation

#### 3. Tokenizer Changes
- **Risk**: Different tokenization for code vs. text
- **Impact**: Incompatible embeddings between models
- **Mitigation**: Separate tokenizer handling, clear migration path

#### 4. Inference Latency
- **Risk**: Significantly slower embedding generation
- **Impact**: User experience degradation, timeout issues
- **Mitigation**: Implement caching, batch processing, async operations

### Medium-Risk Changes

#### 1. Cache Invalidation
- **Risk**: All cached embeddings become invalid
- **Impact**: Performance hit during cache rebuild
- **Mitigation**: Progressive cache warming, background rebuilding

#### 2. API Compatibility
- **Risk**: Different output formats or dimensions
- **Impact**: Client applications may break
- **Mitigation**: Version API, provide migration tools

### Breaking Changes Requiring Code Updates

1. **Model Constants**: All URLs, filenames, and size constants
2. **Memory Limits**: Increase memory allocation limits
3. **Test Data**: Update expected embeddings in test cases
4. **Configuration**: All model name references
5. **Documentation**: Update all references to model specifications

## 8. Step-by-Step Migration Plan

### Phase 1: Preparation (Low Risk)
1. **Backup Current System**
   ```bash
   # Backup current embeddings cache
   cp -r ~/.nomic ~/.nomic_backup_$(date +%Y%m%d)
   
   # Backup configuration files
   git commit -am "Pre-migration backup"
   ```

2. **Update Dependencies** (if needed)
   ```bash
   cargo update candle-core candle-nn candle-transformers
   ```

3. **Create Feature Flag** (optional)
   ```rust
   #[cfg(feature = "nomic-code")]
   const MODEL_NAME: &str = "nomic-ai/nomic-embed-code";
   #[cfg(not(feature = "nomic-code"))]
   const MODEL_NAME: &str = "nomic-ai/nomic-embed-text-v1.5";
   ```

### Phase 2: Model File Updates (Medium Risk)
1. **Update nomic.rs constants**
2. **Update configuration files**
3. **Update streaming integration**
4. **Update test configurations**

### Phase 3: System Integration (High Risk)
1. **Test model loading with existing file**
2. **Validate memory usage**
3. **Test embedding generation**
4. **Verify MCP integration**

### Phase 4: Validation and Deployment
1. **Run comprehensive test suite**
2. **Performance benchmarking**
3. **Gradual rollout with monitoring**
4. **Cache cleanup after validation**

## 9. Implementation Commands and Scripts

### Migration Script
```bash
#!/bin/bash
set -e

echo "=== Nomic Embed Migration: text -> code ==="

# Phase 1: Backup
echo "Phase 1: Creating backups..."
git stash push -m "Pre-migration work in progress"
git commit -am "Pre-migration snapshot" || true

# Phase 2: Update code files
echo "Phase 2: Updating source files..."

# Update nomic.rs
sed -i 's/nomic-embed-text-v1.5-GGUF/nomic-embed-code-GGUF/g' src/embedding/nomic.rs
sed -i 's/nomic-embed-text-v1.5.Q4_K_M.gguf/nomic-embed-code.Q4_K_M.gguf/g' src/embedding/nomic.rs
sed -i 's/nomic-embed-text-v1.5/nomic-embed-code/g' src/embedding/nomic.rs
sed -i 's/84_000_000/4_380_000_000/g' src/embedding/nomic.rs

# Update config files
sed -i 's/nomic-embed-text/nomic-embed-code/g' src/config/mod.rs
sed -i 's/nomic-embed-text/nomic-embed-code/g' src/config/safe_config.rs
sed -i 's/nomic-embed-text/nomic-embed-code/g' src/embedding/streaming_nomic_integration.rs

# Update gitignore
sed -i 's/nomic-embed-text/nomic-embed-code/g' .gitignore

echo "Phase 3: Testing changes..."
cargo check --features=ml
cargo test --features=ml test_nomic_code_migration

echo "Phase 4: Build and validate..."
cargo build --features=ml --release

echo "=== Migration completed successfully ==="
echo "Next steps:"
echo "1. Run full test suite"
echo "2. Validate embedding generation"
echo "3. Monitor memory usage"
echo "4. Clean up old cache files"
```

### Rollback Script
```bash
#!/bin/bash
echo "=== Rolling back migration ==="

git reset --hard HEAD~1
git stash pop || true

echo "Rollback completed. System restored to pre-migration state."
```

### Validation Script
```bash
#!/bin/bash
echo "=== Migration Validation ==="

# Test model loading
echo "Testing model loading..."
timeout 30 cargo run --features=ml --bin test_model_loading

# Test embedding generation  
echo "Testing embedding generation..."
timeout 60 cargo test --features=ml test_embedding_generation

# Memory usage check
echo "Checking memory usage..."
cargo run --features=ml --bin memory_usage_test

echo "Validation completed."
```

## 10. Risk Assessment and Mitigation

### Critical Risks (High Priority)

1. **Out of Memory Errors**
   - **Probability**: High
   - **Impact**: System crash/instability
   - **Mitigation**: Implement memory monitoring, graceful fallbacks

2. **Model Loading Failures**
   - **Probability**: Medium
   - **Impact**: Complete functionality loss
   - **Mitigation**: Extensive testing, rollback procedures

3. **Performance Degradation**
   - **Probability**: High
   - **Impact**: Poor user experience
   - **Mitigation**: Performance optimization, caching strategies

### Medium Risks

1. **Cache Inconsistency**
   - **Mitigation**: Clear cache invalidation strategy

2. **Test Failures**
   - **Mitigation**: Update all test expectations

3. **Configuration Errors**
   - **Mitigation**: Automated configuration validation

## 11. Success Criteria

### Technical Success
- [ ] Model loads without memory issues
- [ ] Embedding generation works for code samples
- [ ] Performance within acceptable limits (< 2x slowdown)
- [ ] All tests pass with new model
- [ ] MCP server integration functional

### Operational Success  
- [ ] No system crashes or instability
- [ ] Cache migration completes cleanly
- [ ] Rollback procedure verified and ready
- [ ] Documentation updated

## 12. Conclusion and Recommendations

### Recommendation: **Proceed with Caution**

This migration involves substantial changes to core functionality with significant risks. The 50x increase in model size fundamentally changes the system's resource requirements.

### Suggested Approach
1. **Implement feature flags** for gradual rollout
2. **Extensive testing** on non-production systems first  
3. **Monitor resource usage** closely during migration
4. **Prepare rollback procedures** before starting

### Timeline Estimate
- **Preparation**: 2-3 hours
- **Implementation**: 4-6 hours  
- **Testing & Validation**: 8-12 hours
- **Total**: 14-21 hours for safe migration

### Next Steps
1. Execute Phase 1 (Preparation) of migration plan
2. Set up monitoring and rollback procedures
3. Begin implementation with extensive testing
4. Consider phased rollout with performance monitoring

---

*This migration plan is comprehensive but should be adapted based on specific system requirements and risk tolerance. Always test in non-production environments first.*