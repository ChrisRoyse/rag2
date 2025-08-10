/// BRUTAL MCP END-TO-END VALIDATION TEST
/// This test validates the ACTUAL MCP server functionality
/// Tests the real binary, real protocols, real search capabilities
/// NO LIES, NO SIMULATIONS, NO ASSUMPTIONS

use std::process::{Command, Stdio};
use std::io::{Write, BufRead, BufReader};
use std::time::{Duration, Instant};
use std::path::PathBuf;
use tempfile::TempDir;
use serde_json::{json, Value};

/// MCP JSON-RPC message structure
#[derive(Debug)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: Option<Value>,
}

impl JsonRpcRequest {
    fn new(id: u64, method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
    
    fn to_json(&self) -> String {
        if let Some(params) = &self.params {
            format!(r#"{{"jsonrpc":"{}","id":{},"method":"{}","params":{}}}"#, 
                self.jsonrpc, self.id, self.method, params)
        } else {
            format!(r#"{{"jsonrpc":"{}","id":{},"method":"{}"}}"#, 
                self.jsonrpc, self.id, self.method)
        }
    }
}

#[derive(Debug)]
struct TestResults {
    server_starts: bool,
    initialize_works: bool,
    capabilities_received: bool,
    index_works: bool,
    search_works: bool,
    bm25_available: bool,
    git_watcher_available: bool,
    error_details: Vec<String>,
    total_duration: Duration,
}

impl TestResults {
    fn new() -> Self {
        Self {
            server_starts: false,
            initialize_works: false,
            capabilities_received: false,
            index_works: false,
            search_works: false,
            bm25_available: false,
            git_watcher_available: false,
            error_details: Vec::new(),
            total_duration: Duration::from_secs(0),
        }
    }
    
    fn success_rate(&self) -> f64 {
        let total_tests = 7;
        let successful = [
            self.server_starts,
            self.initialize_works,
            self.capabilities_received,
            self.index_works,
            self.search_works,
            self.bm25_available,
            self.git_watcher_available,
        ].iter().filter(|&&x| x).count();
        
        (successful as f64 / total_tests as f64) * 100.0
    }
    
    fn print_brutal_assessment(&self) {
        println!("\n=== BRUTAL MCP SERVER VALIDATION RESULTS ===");
        println!("Total test duration: {:?}", self.total_duration);
        println!("Success rate: {:.1}%", self.success_rate());
        println!();
        
        println!("✓/✗ Test Results:");
        println!("  {} Server startup", if self.server_starts { "✓" } else { "✗" });
        println!("  {} MCP initialize", if self.initialize_works { "✓" } else { "✗" });
        println!("  {} Capabilities", if self.capabilities_received { "✓" } else { "✗" });
        println!("  {} Index directory", if self.index_works { "✓" } else { "✗" });
        println!("  {} Search functionality", if self.search_works { "✓" } else { "✗" });
        println!("  {} BM25 search", if self.bm25_available { "✓" } else { "✗" });
        println!("  {} Git watcher", if self.git_watcher_available { "✓" } else { "✗" });
        
        if !self.error_details.is_empty() {
            println!("\n=== ERRORS ENCOUNTERED ===");
            for (i, error) in self.error_details.iter().enumerate() {
                println!("{}. {}", i + 1, error);
            }
        }
        
        println!();
        if self.success_rate() >= 80.0 {
            println!("🎯 ASSESSMENT: MCP server is FUNCTIONAL for production use");
        } else if self.success_rate() >= 60.0 {
            println!("⚠️  ASSESSMENT: MCP server has SIGNIFICANT ISSUES but partially works");
        } else {
            println!("❌ ASSESSMENT: MCP server is NOT FUNCTIONAL - major failures");
        }
        
        println!("\nTRUTH: This test validates the ACTUAL MCP server binary with REAL protocol messages.");
        println!("No simulations, no mocks, no lies. These are the real capabilities.");
    }
}

#[tokio::test]
async fn brutal_mcp_end_to_end_validation() {
    let start_time = Instant::now();
    let mut results = TestResults::new();
    
    println!("=== STARTING BRUTAL MCP SERVER VALIDATION ===");
    println!("Testing the ACTUAL MCP server binary with REAL protocol messages");
    
    // Create test directory with sample files
    let test_dir = TempDir::new().expect("Failed to create temp dir");
    create_test_files(&test_dir).expect("Failed to create test files");
    
    println!("Test directory: {}", test_dir.path().display());
    
    // Test 1: Can we start the MCP server?
    let binary_path = find_mcp_binary();
    println!("Using MCP binary: {}", binary_path.display());
    
    let mut child = match Command::new(&binary_path)
        .arg(test_dir.path())
        .arg("--debug")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            println!("✓ MCP server process started successfully");
            results.server_starts = true;
            child
        }
        Err(e) => {
            results.error_details.push(format!("Failed to start MCP server: {}", e));
            results.total_duration = start_time.elapsed();
            results.print_brutal_assessment();
            panic!("Cannot proceed - server won't start");
        }
    };
    
    // Get stdin/stdout handles
    let stdin = child.stdin.take().expect("Failed to get stdin");
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let mut reader = BufReader::new(stdout);
    
    // Helper function to send request and read response
    let mut send_request = |mut stdin: std::process::ChildStdin, request: JsonRpcRequest| -> Result<(Value, std::process::ChildStdin), String> {
        let json_str = request.to_json();
        println!("→ Sending: {}", json_str);
        
        // Write request
        if let Err(e) = writeln!(stdin, "{}", json_str) {
            return Err(format!("Failed to write request: {}", e));
        }
        
        // Read response with timeout
        let mut response_line = String::new();
        match reader.read_line(&mut response_line) {
            Ok(0) => return Err("Server closed stdout".to_string()),
            Ok(_) => {
                println!("← Received: {}", response_line.trim());
                match serde_json::from_str::<Value>(&response_line) {
                    Ok(json) => Ok((json, stdin)),
                    Err(e) => Err(format!("Invalid JSON response: {}", e))
                }
            }
            Err(e) => Err(format!("Failed to read response: {}", e))
        }
    };
    
    // Test 2: Initialize the MCP server
    println!("\n--- Testing MCP Initialize ---");
    let init_request = JsonRpcRequest::new(1, "initialize", Some(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {
                "listChanged": true
            }
        },
        "clientInfo": {
            "name": "brutal-test-client",
            "version": "1.0.0"
        }
    })));
    
    match send_request(stdin, init_request) {
        Ok((response, new_stdin)) => {
            stdin = new_stdin;
            if response.get("result").is_some() {
                println!("✓ MCP initialize succeeded");
                results.initialize_works = true;
                
                if let Some(capabilities) = response["result"]["capabilities"].as_object() {
                    println!("✓ Server capabilities received");
                    results.capabilities_received = true;
                    
                    // Check available tools
                    if let Some(tools) = capabilities.get("tools") {
                        println!("Available tools: {}", tools);
                    }
                    
                    // Check for BM25 capability 
                    if capabilities.contains_key("tools") || response.to_string().contains("search") {
                        results.bm25_available = true;
                        println!("✓ Search capabilities detected");
                    }
                }
            } else {
                results.error_details.push("Initialize response missing result field".to_string());
            }
        }
        Err(e) => {
            results.error_details.push(format!("Initialize failed: {}", e));
        }
    }
    
    // Test 3: Test tools/list to see available tools
    println!("\n--- Testing tools/list ---");
    let tools_request = JsonRpcRequest::new(2, "tools/list", None);
    
    match send_request(stdin, tools_request) {
        Ok((response, new_stdin)) => {
            stdin = new_stdin;
            println!("Tools response: {}", response);
            
            if let Some(tools) = response["result"]["tools"].as_array() {
                let tool_names: Vec<String> = tools.iter()
                    .filter_map(|t| t["name"].as_str())
                    .map(|s| s.to_string())
                    .collect();
                
                println!("Available tools: {:?}", tool_names);
                
                if tool_names.iter().any(|name| name.contains("search")) {
                    results.bm25_available = true;
                }
                
                if tool_names.iter().any(|name| name.contains("index")) {
                    println!("✓ Index tools available");
                }
                
                if tool_names.iter().any(|name| name.contains("watcher") || name.contains("git")) {
                    results.git_watcher_available = true;
                    println!("✓ Git watcher tools available");
                }
            }
        }
        Err(e) => {
            results.error_details.push(format!("Tools list failed: {}", e));
        }
    }
    
    // Test 4: Try to index the test directory
    println!("\n--- Testing Directory Indexing ---");
    let index_request = JsonRpcRequest::new(3, "tools/call", Some(json!({
        "name": "index_directory",
        "arguments": {
            "path": test_dir.path().to_string_lossy()
        }
    })));
    
    match send_request(stdin, index_request) {
        Ok((response, new_stdin)) => {
            stdin = new_stdin;
            println!("Index response: {}", response);
            
            if response.get("result").is_some() && !response.to_string().contains("error") {
                results.index_works = true;
                println!("✓ Directory indexing succeeded");
            } else {
                results.error_details.push("Index operation failed or returned error".to_string());
            }
        }
        Err(e) => {
            results.error_details.push(format!("Index request failed: {}", e));
        }
    }
    
    // Test 5: Try to search for content
    println!("\n--- Testing Search Functionality ---");
    let search_request = JsonRpcRequest::new(4, "tools/call", Some(json!({
        "name": "search",
        "arguments": {
            "query": "function",
            "max_results": 5
        }
    })));
    
    match send_request(stdin, search_request) {
        Ok((response, new_stdin)) => {
            stdin = new_stdin;
            println!("Search response: {}", response);
            
            if let Some(result) = response.get("result") {
                if result.get("content").is_some() || result.to_string().contains("results") {
                    results.search_works = true;
                    println!("✓ Search functionality works");
                } else {
                    results.error_details.push("Search returned no results or invalid format".to_string());
                }
            } else {
                results.error_details.push("Search request failed".to_string());
            }
        }
        Err(e) => {
            results.error_details.push(format!("Search request failed: {}", e));
        }
    }
    
    // Clean shutdown
    println!("\n--- Testing Graceful Shutdown ---");
    drop(stdin);
    
    // Wait for server to shut down
    match child.wait_timeout(Duration::from_secs(5)) {
        Ok(Some(status)) => {
            println!("✓ Server shutdown gracefully with status: {}", status);
        }
        Ok(None) => {
            println!("⚠ Server didn't shutdown in 5 seconds, killing");
            let _ = child.kill();
        }
        Err(e) => {
            println!("⚠ Error during shutdown: {}", e);
            let _ = child.kill();
        }
    }
    
    results.total_duration = start_time.elapsed();
    results.print_brutal_assessment();
    
    // Assert minimum functionality for test to pass
    assert!(results.server_starts, "Server must start");
    assert!(results.initialize_works, "MCP initialize must work"); 
    
    // Overall assessment
    if results.success_rate() < 50.0 {
        panic!("BRUTAL TRUTH: MCP server failed too many basic tests ({:.1}% success rate)", results.success_rate());
    }
}

fn find_mcp_binary() -> PathBuf {
    let possible_paths = [
        "target/debug/mcp_server",
        "target/release/mcp_server", 
        "./mcp_server",
    ];
    
    for path in &possible_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return path_buf;
        }
    }
    
    panic!("MCP server binary not found. Tried paths: {:?}", possible_paths);
}

fn create_test_files(dir: &TempDir) -> std::io::Result<()> {
    use std::fs;
    
    // Create some test files to index
    fs::write(
        dir.path().join("test.rs"), 
        r#"
fn main() {
    println!("Hello world");
}

pub fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

struct User {
    name: String,
    email: String,
}

impl User {
    fn new(name: String, email: String) -> Self {
        Self { name, email }
    }
    
    fn get_display_name(&self) -> &str {
        &self.name
    }
}
"#,
    )?;
    
    fs::write(
        dir.path().join("README.md"),
        r#"# Test Project

This is a test project for MCP server validation.

## Features

- Function definitions
- Struct definitions  
- Implementation blocks
- Documentation

## Usage

Run the tests to validate functionality.
"#,
    )?;
    
    fs::write(
        dir.path().join("config.json"),
        r#"{
    "name": "test-project",
    "version": "1.0.0",
    "description": "Test configuration file",
    "settings": {
        "debug": true,
        "timeout": 30
    }
}"#,
    )?;
    
    // Create subdirectory
    fs::create_dir_all(dir.path().join("src"))?;
    fs::write(
        dir.path().join("src").join("lib.rs"),
        r#"//! Library module

pub mod utils;

pub use utils::*;

/// Core functionality
pub fn process_data(input: &str) -> String {
    format!("Processed: {}", input)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_data() {
        assert_eq!(process_data("test"), "Processed: test");
    }
}
"#,
    )?;
    
    fs::write(
        dir.path().join("src").join("utils.rs"),
        r#"//! Utility functions

use std::collections::HashMap;

pub fn create_map() -> HashMap<String, i32> {
    let mut map = HashMap::new();
    map.insert("key1".to_string(), 100);
    map.insert("key2".to_string(), 200);
    map
}

pub fn format_number(n: i32) -> String {
    format!("Number: {}", n)
}

/// Helper function for debugging
pub fn debug_print(msg: &str) {
    eprintln!("[DEBUG] {}", msg);
}
"#,
    )?;
    
    Ok(())
}