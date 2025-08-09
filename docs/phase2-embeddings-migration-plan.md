# Phase 2: Embeddings Migration Plan - Detailed Implementation

## 🎯 Migration Overview

**MIGRATION TYPE: CRITICAL SYSTEM OVERHAUL**
- **Complexity Level**: HIGH - Multi-language system with tight integration
- **Risk Level**: MEDIUM-HIGH - Potential for system-wide failure
- **Duration**: 8-12 weeks with proper testing
- **Impact**: 50x model size increase, 2-5x performance impact

## Critical Migration Requirements

### Pre-Migration Checklist
- [ ] **Full system backup** - All databases, configs, caches
- [ ] **Test environment setup** - Isolated environment with new model  
- [ ] **Monitoring infrastructure** - Memory, performance, error tracking
- [ ] **Rollback procedures** - Tested emergency recovery
- [ ] **Stakeholder approval** - Confirmed understanding of performance impact

## Files Requiring Changes

### 🔴 Critical Files (MUST CHANGE)

**1. `/src/embedding/nomic.rs` (Lines 103-106)**
```rust
// CURRENT (WRONG):
const MODEL_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf";
const TOKENIZER_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";
const MODEL_FILENAME: &'static str = "nomic-embed-text-v1.5.Q4_K_M.gguf";

// TARGET (CORRECT):  
const MODEL_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code-v1-Q4_K_M-GGUF/resolve/main/nomic-embed-code.Q4_K_M.gguf";
const TOKENIZER_URL: &'static str = "https://huggingface.co/nomic-ai/nomic-embed-code-v1/resolve/main/tokenizer.json";
const MODEL_FILENAME: &'static str = "nomic-embed-code.Q4_K_M.gguf";
```

**2. `/src/embedding/streaming_nomic_integration.rs` (Line 538)**
```rust
// CURRENT (WRONG):
let model_path = "models/nomic-embed-text-v1.5.Q4_K_M.gguf";

// TARGET (CORRECT):
let model_path = "models/nomic-embed-code.Q4_K_M.gguf";
```

**3. `/src/config/safe_config.rs` (Line 96)**
```rust
// CURRENT (WRONG):
model_path: PathBuf::from("./test_models/nomic-embed-text-v1.5.gguf"),

// TARGET (CORRECT):
model_path: PathBuf::from("./test_models/nomic-embed-code.Q4_K_M.gguf"),
```

**4. `/src/config/mod.rs` (Line 135)**
```rust
// CURRENT (WRONG):
model_name: "nomic-ai/nomic-embed-text-v1.5".to_string(),

// TARGET (CORRECT):
model_name: "nomic-ai/nomic-embed-code-v1".to_string(),
```

**5. Model Size Constants (Critical)**
```rust
// In /src/embedding/nomic.rs - Line 105
// CURRENT (WRONG):
const MODEL_SIZE: u64 = 84_000_000;  // ~84MB

// TARGET (CORRECT):
const MODEL_SIZE: u64 = 4_378_000_000;  // ~4.38GB
```

### 🟡 Configuration Files (SHOULD CHANGE)

**6. `.gitignore` - Update ignore patterns**
```gitignore
# CURRENT:
# Nomic model cache  
.nomic/nomic-embed-text-*

# TARGET:
# Nomic model cache
.nomic/nomic-embed-code-*
.nomic/nomic-embed-text-*  # Keep for cleanup
```

**7. Documentation Updates**
- `README.md` - System requirements update (memory, disk space)
- `DEPLOYMENT.md` - Model download instructions
- Configuration examples in docs/

## Model Integration Strategy

### Current Model Status
```bash
# Your existing model (CORRECT):
/home/cabdru/rag/model/nomic-embed-code.Q4_K_M.gguf (4.1GB)

# Status: ✅ Downloaded and ready
# Validation: File size matches expected 4.38GB (within 5% variance)
```

### Integration Approach

**Option 1: Direct Path Integration (RECOMMENDED)**
```rust
// Modify nomic.rs to check local path first
async fn ensure_files_cached() -> Result<(PathBuf, PathBuf)> {
    // Check project-local model first
    let project_model = PathBuf::from("./model/nomic-embed-code.Q4_K_M.gguf");
    if project_model.exists() {
        let tokenizer_path = download_tokenizer().await?;
        return Ok((project_model, tokenizer_path));
    }
    
    // Fallback to cache directory download
    let cache_dir = dirs::home_dir()
        .ok_or_else(|| anyhow!("Could not determine home directory"))?
        .join(".nomic");
    // ... existing download logic
}
```

**Option 2: Configuration Override**
```toml
# In config files
[embedding]
model_path = "./model/nomic-embed-code.Q4_K_M.gguf"
model_name = "nomic-embed-code-v1"
force_local_model = true
```

## Migration Commands & Scripts

### Automated Migration Script
```bash
#!/bin/bash
# File: /home/cabdru/rag/scripts/migrate-to-code-embeddings.sh

set -euo pipefail

echo "🚀 Starting nomic-embed-code migration..."

# 1. Backup current system
echo "📦 Creating backup..."
cp -r .embed .embed.backup.$(date +%Y%m%d_%H%M%S)
cp -r src/embedding src/embedding.backup.$(date +%Y%m%d_%H%M%S)

# 2. Update source files
echo "🔧 Updating source files..."

# Update nomic.rs constants
sed -i 's|nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf|nomic-embed-code-v1-Q4_K_M-GGUF/resolve/main/nomic-embed-code.Q4_K_M.gguf|g' src/embedding/nomic.rs
sed -i 's|nomic-embed-text-v1.5/resolve/main/tokenizer.json|nomic-embed-code-v1/resolve/main/tokenizer.json|g' src/embedding/nomic.rs
sed -i 's|nomic-embed-text-v1.5.Q4_K_M.gguf|nomic-embed-code.Q4_K_M.gguf|g' src/embedding/nomic.rs
sed -i 's|84_000_000|4_378_000_000|g' src/embedding/nomic.rs

# Update streaming integration
sed -i 's|nomic-embed-text-v1.5.Q4_K_M.gguf|nomic-embed-code.Q4_K_M.gguf|g' src/embedding/streaming_nomic_integration.rs

# Update config files
sed -i 's|nomic-embed-text-v1.5|nomic-embed-code|g' src/config/safe_config.rs
sed -i 's|nomic-ai/nomic-embed-text-v1.5|nomic-ai/nomic-embed-code-v1|g' src/config/mod.rs

# 3. Clear invalid caches
echo "🗑️  Clearing embedding caches..."
rm -rf .embed/cache/embeddings/*
rm -rf ~/.nomic/nomic-embed-text-*

# 4. Validate changes
echo "✅ Validating migration..."
grep -r "nomic-embed-text" src/ && echo "❌ Still found text references!" || echo "✅ All text references updated"
grep -r "nomic-embed-code" src/ && echo "✅ Code references found" || echo "❌ No code references found!"

# 5. Test compile
echo "🔨 Testing compilation..."
cargo check --features ml

echo "🎉 Migration complete! Run tests with: cargo test --features ml"
```

### Validation Commands
```bash
# 1. Verify model file
ls -lah /home/cabdru/rag/model/nomic-embed-code.Q4_K_M.gguf

# 2. Check for remaining text model references  
grep -r "nomic-embed-text" src/

# 3. Validate new code model references
grep -r "nomic-embed-code" src/

# 4. Test compilation
cargo check --features ml

# 5. Run embedding tests
cargo test --features ml test_embedding_generation

# 6. Performance benchmark
cargo test --release --features ml embedding_performance_benchmark
```

## Cache Management & Cleanup

### Cache Invalidation Strategy
```bash
# 1. Identify cache locations
find . -name "*embed*" -type d
find ~/.nomic -name "*text*" -type f

# 2. Clear all embedding caches
rm -rf .embed/cache/embeddings/*
rm -rf .embed/vector_cache/*
rm -rf ~/.cache/nomic-embed-*

# 3. Clear LanceDB vector storage
rm -rf .embed/lancedb/*

# 4. Clear any Tantivy indexes (optional - text search still works)
# rm -rf .embed/tantivy/*
```

### Cache Rebuild Process
```rust
// After migration, trigger full rebuild:
// 1. Re-process all files through embedding pipeline
// 2. Rebuild vector indices
// 3. Update search fusion weights
// 4. Validate embedding quality
```

## Testing & Validation Procedures

### Unit Test Validation
```bash
# 1. Core embedding functionality
cargo test --features ml test_nomic_embedder_initialization
cargo test --features ml test_embedding_generation  
cargo test --features ml test_batch_embedding
cargo test --features ml test_embedding_dimensions

# 2. Model loading tests
cargo test --features ml test_gguf_model_loading
cargo test --features ml test_tokenizer_integration
cargo test --features ml test_model_caching

# 3. Error handling tests  
cargo test --features ml test_invalid_model_path
cargo test --features ml test_corrupted_model_handling
cargo test --features ml test_memory_pressure_handling
```

### Integration Test Validation
```bash
# 1. End-to-end pipeline
cargo test --features full-system test_full_embedding_pipeline
cargo test --features full-system test_search_integration
cargo test --features full-system test_mcp_server_integration

# 2. Performance validation
cargo test --release --features full-system embedding_performance_benchmark
cargo test --release --features full-system memory_usage_benchmark
cargo test --release --features full-system search_accuracy_validation
```

### Manual Validation Steps
```bash
# 1. Start system and check logs
RUST_LOG=debug cargo run --features full-system

# 2. Test embedding generation
echo "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)" | \
  cargo run --features ml --bin test_embedding

# 3. Test search functionality
cargo run --features full-system --bin test_search -- "python function fibonacci"

# 4. Monitor memory usage
top -p $(pgrep embed-search)
```

## Performance Impact Assessment

### Expected Changes

**Model Size**: 84MB → 4.38GB (50.5x increase)
**Memory Usage**: 200MB → 1-4GB (5-20x increase) 
**Load Time**: 5-10 seconds → 30-60 seconds (6-12x increase)
**Inference Speed**: 100-500ms → 200-1000ms (2-5x increase)

### Mitigation Strategies

**1. Memory Management**
```rust
// Implement model quantization options
// Add streaming inference for memory-constrained environments  
// Implement model sharding for very large deployments
```

**2. Performance Optimization**
```rust
// Cache frequently used embeddings more aggressively
// Implement batch processing for multiple requests
// Use async loading to prevent blocking
```

**3. Resource Monitoring**
```rust
// Add memory usage alerts
// Implement automatic garbage collection triggers
// Monitor embedding quality degradation
```

## Rollback Procedures

### Emergency Rollback (< 5 minutes)
```bash
#!/bin/bash
# Emergency rollback script

echo "🚨 EMERGENCY ROLLBACK INITIATED"

# 1. Stop all services
pkill -f embed-search
pkill -f mcp-server

# 2. Restore backed up files
cp -r src/embedding.backup.* src/embedding/
cp -r .embed.backup.* .embed/

# 3. Clear new model caches
rm -rf ~/.nomic/nomic-embed-code-*

# 4. Restart with old configuration  
RUST_LOG=info cargo run --features ml &

echo "✅ Rollback complete - system restored to previous state"
```

### Gradual Rollback (Complete)
1. **Immediate**: Stop accepting new embedding requests
2. **5 minutes**: Restore source code from backup
3. **10 minutes**: Rebuild with old model references  
4. **15 minutes**: Restore cache from backup
5. **20 minutes**: Full system validation
6. **25 minutes**: Resume normal operations

## Risk Mitigation

### Critical Risks & Mitigations

**1. Memory Exhaustion**
```yaml
Risk: System OOM due to 50x model size increase
Mitigation: 
  - Pre-deployment memory testing
  - Automatic model unloading under pressure
  - Graceful degradation to text search only
Monitoring: Memory usage alerts at 70% threshold
```

**2. Performance Regression**
```yaml  
Risk: 2-10x slower inference affecting user experience
Mitigation:
  - Aggressive caching strategies
  - Async processing where possible
  - Fallback to faster search methods
Monitoring: Response time alerts at 2x baseline
```

**3. Data Corruption**
```yaml
Risk: Cache corruption during migration
Mitigation:
  - Complete cache invalidation and rebuild
  - Checksums for all cached data
  - Atomic operations for cache updates
Monitoring: Data integrity checks on startup
```

## Success Criteria

### ✅ Technical Success Criteria
- [ ] All embedding requests use nomic-embed-code model
- [ ] System memory usage remains stable under load
- [ ] Embedding quality improves for code-related queries  
- [ ] No data loss during migration
- [ ] All tests pass with new model
- [ ] Performance regression < 5x slower

### ✅ Operational Success Criteria  
- [ ] Migration completed within planned timeframe
- [ ] No unplanned downtime > 1 hour
- [ ] Rollback procedures tested and validated
- [ ] Monitoring and alerting functional
- [ ] Documentation updated and accurate
- [ ] Team trained on new system characteristics

## Timeline

### Week 1: Preparation
- [ ] Environment setup and testing infrastructure
- [ ] Backup procedures and rollback testing
- [ ] Code changes and validation in test environment

### Week 2: Implementation  
- [ ] Source code migration with thorough testing
- [ ] Cache invalidation and cleanup
- [ ] Initial deployment in staging environment

### Week 3: Validation
- [ ] Comprehensive testing suite execution
- [ ] Performance benchmarking and optimization  
- [ ] Integration testing with dependent systems

### Week 4: Deployment
- [ ] Production migration during maintenance window
- [ ] Real-time monitoring and performance validation
- [ ] User acceptance testing and feedback collection

---

**Migration Plan Status**: ✅ COMPLETE AND READY FOR EXECUTION

**Risk Assessment**: MEDIUM-HIGH - Complex but manageable with proper preparation

**Recommendation**: Proceed with caution and comprehensive testing at each stage

**Next Phase**: Implementation Strategy & Swarm Orchestration (Phase 3)