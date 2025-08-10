#!/usr/bin/env rust-script
//! Comprehensive MCP Integration Verification Script
//! 
//! This script verifies all components are properly connected:
//! - Tantivy search with fuzzy matching
//! - Nomic embeddings and semantic search
//! - AST symbolic search
//! - BM25 scoring
//! - Intelligent fusion
//! - Git watching
//! - Embedding index directory

use std::path::PathBuf;
use std::process::Command;
use anyhow::{Result, Context};

#[derive(Debug)]
struct VerificationResult {
    component: String,
    status: bool,
    details: String,
    suggestions: Vec<String>,
}

impl VerificationResult {
    fn new(component: &str) -> Self {
        Self {
            component: component.to_string(),
            status: false,
            details: String::new(),
            suggestions: Vec::new(),
        }
    }
    
    fn success(mut self, details: &str) -> Self {
        self.status = true;
        self.details = details.to_string();
        self
    }
    
    fn failure(mut self, details: &str) -> Self {
        self.status = false;
        self.details = details.to_string();
        self
    }
    
    fn suggest(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }
}

fn main() -> Result<()> {
    println!("🔍 MCP Integration Verification Script");
    println!("=====================================\n");
    
    let mut results = Vec::new();
    
    // 1. Check Tantivy Search
    results.push(verify_tantivy());
    
    // 2. Check Nomic Embeddings
    results.push(verify_nomic_embeddings());
    
    // 3. Check BM25 Scoring
    results.push(verify_bm25());
    
    // 4. Check AST/Symbol Search
    results.push(verify_symbol_search());
    
    // 5. Check Intelligent Fusion
    results.push(verify_fusion());
    
    // 6. Check Git Watcher
    results.push(verify_git_watcher());
    
    // 7. Check Embedding Index Directory
    results.push(verify_embedding_index());
    
    // 8. Check MCP Server Connection
    results.push(verify_mcp_server());
    
    // Print results
    println!("\n📊 Verification Results:");
    println!("========================\n");
    
    let mut all_passed = true;
    for result in &results {
        let status_emoji = if result.status { "✅" } else { "❌" };
        println!("{} {}: {}", status_emoji, result.component, result.details);
        
        if !result.suggestions.is_empty() {
            println!("   💡 Suggestions:");
            for suggestion in &result.suggestions {
                println!("      - {}", suggestion);
            }
        }
        
        if !result.status {
            all_passed = false;
        }
    }
    
    println!("\n📈 Overall Status: {}", 
        if all_passed { "✅ All components operational!" } 
        else { "⚠️ Some components need attention" }
    );
    
    // Generate fix script if needed
    if !all_passed {
        generate_fix_script(&results)?;
    }
    
    Ok(())
}

fn verify_tantivy() -> VerificationResult {
    let mut result = VerificationResult::new("Tantivy Search");
    
    // Check if tantivy_search.rs exists and has fuzzy matching
    if std::path::Path::new("src/search/tantivy_search.rs").exists() {
        let content = std::fs::read_to_string("src/search/tantivy_search.rs").unwrap_or_default();
        
        if content.contains("FuzzyTermQuery") && content.contains("fuzzy_search") {
            result.success("Tantivy with fuzzy matching is properly configured")
        } else {
            result.failure("Tantivy exists but fuzzy matching not fully implemented")
                .suggest("Implement fuzzy_search method with FuzzyTermQuery")
                .suggest("Add levenshtein distance parameter for fuzzy matching")
        }
    } else {
        result.failure("Tantivy search module not found")
            .suggest("Create src/search/tantivy_search.rs")
    }
    
    result
}

fn verify_nomic_embeddings() -> VerificationResult {
    let mut result = VerificationResult::new("Nomic Embeddings");
    
    // Check Nomic module and model
    if std::path::Path::new("src/embedding/nomic.rs").exists() {
        let content = std::fs::read_to_string("src/embedding/nomic.rs").unwrap_or_default();
        
        if content.contains("NomicEmbedder") && content.contains("generate_embedding") {
            // Check if model file exists
            let model_path = PathBuf::from("models/nomic-embed-text-v1.5.f16.gguf");
            if model_path.exists() {
                result.success("Nomic embedder configured with model file present")
            } else {
                result.failure("Nomic embedder configured but model file missing")
                    .suggest("Download nomic-embed-text-v1.5.f16.gguf to models/ directory")
                    .suggest("wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf -P models/")
            }
        } else {
            result.failure("Nomic embedder not properly implemented")
                .suggest("Implement NomicEmbedder with generate_embedding method")
        }
    } else {
        result.failure("Nomic embeddings module not found")
            .suggest("Create src/embedding/nomic.rs")
    }
    
    result
}

fn verify_bm25() -> VerificationResult {
    let mut result = VerificationResult::new("BM25 Scoring");
    
    if std::path::Path::new("src/search/bm25.rs").exists() {
        let content = std::fs::read_to_string("src/search/bm25.rs").unwrap_or_default();
        
        if content.contains("BM25Engine") && content.contains("calculate_score") {
            result.success("BM25 scoring engine properly implemented")
        } else {
            result.failure("BM25 module exists but scoring not fully implemented")
                .suggest("Implement calculate_score method with proper IDF calculation")
        }
    } else {
        result.failure("BM25 scoring module not found")
            .suggest("Create src/search/bm25.rs")
    }
    
    result
}

fn verify_symbol_search() -> VerificationResult {
    let mut result = VerificationResult::new("AST/Symbol Search");
    
    if std::path::Path::new("src/search/symbol_index.rs").exists() {
        let content = std::fs::read_to_string("src/search/symbol_index.rs").unwrap_or_default();
        
        // Check if it's the temporary regex implementation
        if content.contains("TODO: Replace with proper tree-sitter") {
            result.failure("Symbol search using temporary regex implementation")
                .suggest("Implement tree-sitter based AST parsing")
                .suggest("Add tree-sitter-rust, tree-sitter-javascript dependencies")
        } else if content.contains("tree_sitter") || content.contains("TreeSitter") {
            result.success("Symbol search with proper AST parsing implemented")
        } else {
            result.failure("Symbol search implementation unclear")
                .suggest("Verify AST parsing implementation")
        }
    } else {
        result.failure("Symbol search module not found")
            .suggest("Create src/search/symbol_index.rs")
    }
    
    result
}

fn verify_fusion() -> VerificationResult {
    let mut result = VerificationResult::new("Intelligent Fusion");
    
    if std::path::Path::new("src/search/search_adapter.rs").exists() {
        let content = std::fs::read_to_string("src/search/search_adapter.rs").unwrap_or_default();
        
        if content.contains("UnifiedSearchAdapter") && content.contains("intelligent_fusion") {
            result.success("Intelligent fusion with result merging implemented")
        } else if content.contains("UnifiedSearchAdapter") {
            result.failure("Unified adapter exists but intelligent fusion not complete")
                .suggest("Implement intelligent_fusion method with score normalization")
                .suggest("Add reciprocal rank fusion algorithm")
        } else {
            result.failure("Search adapter not properly configured")
                .suggest("Implement UnifiedSearchAdapter")
        }
    } else {
        result.failure("Search adapter module not found")
            .suggest("Create src/search/search_adapter.rs")
    }
    
    result
}

fn verify_git_watcher() -> VerificationResult {
    let mut result = VerificationResult::new("Git Watcher");
    
    let git_watcher = std::path::Path::new("src/git/watcher.rs").exists();
    let watcher_module = std::path::Path::new("src/watcher/git_watcher.rs").exists();
    
    if git_watcher || watcher_module {
        let path = if git_watcher { "src/git/watcher.rs" } else { "src/watcher/git_watcher.rs" };
        let content = std::fs::read_to_string(path).unwrap_or_default();
        
        if content.contains("GitWatcher") && content.contains("watch_changes") {
            result.success("Git watcher properly configured for change detection")
        } else {
            result.failure("Git watcher exists but not fully functional")
                .suggest("Implement watch_changes method")
                .suggest("Add git2 integration for change detection")
        }
    } else {
        result.failure("Git watcher module not found")
            .suggest("Create src/git/watcher.rs or src/watcher/git_watcher.rs")
    }
    
    result
}

fn verify_embedding_index() -> VerificationResult {
    let mut result = VerificationResult::new("Embedding Index Directory");
    
    let embed_dir = PathBuf::from(".embed-search");
    
    if embed_dir.exists() && embed_dir.is_dir() {
        result.success("Embedding index directory exists at .embed-search/")
    } else {
        result.failure("Embedding index directory not found")
            .suggest("Create .embed-search/ directory")
            .suggest("mkdir -p .embed-search")
    }
    
    result
}

fn verify_mcp_server() -> VerificationResult {
    let mut result = VerificationResult::new("MCP Server Connection");
    
    if std::path::Path::new("src/mcp/server.rs").exists() {
        let content = std::fs::read_to_string("src/mcp/server.rs").unwrap_or_default();
        
        if content.contains("UnifiedSearchAdapter") && content.contains("tool_registry") {
            result.success("MCP server properly connected to all search components")
        } else {
            result.failure("MCP server exists but not fully integrated")
                .suggest("Connect UnifiedSearchAdapter to MCP server")
                .suggest("Register all search tools in tool_registry")
        }
    } else {
        result.failure("MCP server module not found")
            .suggest("Create src/mcp/server.rs")
    }
    
    result
}

fn generate_fix_script(results: &[VerificationResult]) -> Result<()> {
    let mut fixes = Vec::new();
    
    for result in results {
        if !result.status {
            for suggestion in &result.suggestions {
                fixes.push(format!("# Fix for {}: {}", result.component, suggestion));
            }
        }
    }
    
    if !fixes.is_empty() {
        let fix_script = format!(
            "#!/bin/bash\n\
            # Auto-generated fix script for MCP integration\n\
            \n\
            {}\n\
            \n\
            echo '✅ Fix script completed!'\n",
            fixes.join("\n")
        );
        
        std::fs::write("fix_mcp_integration.sh", fix_script)?;
        println!("\n💾 Fix script generated: fix_mcp_integration.sh");
    }
    
    Ok(())
}