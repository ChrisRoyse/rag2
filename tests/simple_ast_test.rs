use embed_search::ast::{SimpleRustParser, SymbolDatabase, SymbolKind};
use std::fs;
use std::time::Instant;

// Simple integration test for the AST parser
#[test]
fn test_ast_parser_on_lib_rs() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    
    // Test on the main library file
    let lib_path = "src/lib.rs";
    if let Ok(content) = fs::read_to_string(lib_path) {
        let line_count = content.lines().count();
        
        let start = Instant::now();
        let symbols = parser.extract_symbols(&content, lib_path).expect("Failed to parse lib.rs");
        let elapsed = start.elapsed();
        
        println!("✅ Parsed {}: {} symbols from {} lines in {:.2}ms", 
                 lib_path, symbols.len(), line_count, elapsed.as_millis());
        
        // Basic verification
        assert!(!symbols.is_empty(), "Should find some symbols in lib.rs");
        
        // Check for expected module declarations
        let modules: Vec<_> = symbols.iter()
            .filter(|s| s.kind == SymbolKind::Module)
            .collect();
        println!("Found {} modules: {:?}", modules.len(), 
                 modules.iter().map(|s| &s.name).collect::<Vec<_>>());
        
        // Performance check
        let symbols_per_sec = symbols.len() as f64 / elapsed.as_secs_f64();
        println!("Performance: {:.1} symbols/second", symbols_per_sec);
        
        // Should be reasonably fast
        assert!(symbols_per_sec > 100.0, "Should process at least 100 symbols per second");
    } else {
        panic!("Could not read src/lib.rs");
    }
}

#[test] 
fn test_ast_parser_on_simple_parser_rs() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    
    // Test on our own parser code
    let parser_path = "src/ast/simple_parser.rs";
    if let Ok(content) = fs::read_to_string(parser_path) {
        let line_count = content.lines().count();
        
        let start = Instant::now();
        let symbols = parser.extract_symbols(&content, parser_path)
            .expect("Failed to parse simple_parser.rs");
        let elapsed = start.elapsed();
        
        println!("✅ Parsed {}: {} symbols from {} lines in {:.2}ms", 
                 parser_path, symbols.len(), line_count, elapsed.as_millis());
        
        // Should find our main structures
        let structs: Vec<_> = symbols.iter()
            .filter(|s| s.kind == SymbolKind::Struct)
            .map(|s| &s.name)
            .collect();
        
        println!("Found structs: {:?}", structs);
        assert!(structs.contains(&&"SimpleRustParser".to_string()), "Should find SimpleRustParser struct");
        assert!(structs.contains(&&"SymbolDatabase".to_string()), "Should find SymbolDatabase struct");
        
        // Should find methods
        let methods: Vec<_> = symbols.iter()
            .filter(|s| s.kind == SymbolKind::Method)
            .collect();
        
        println!("Found {} methods", methods.len());
        assert!(!methods.is_empty(), "Should find some methods");
        
        // Check for extract_symbols method
        assert!(methods.iter().any(|m| m.name == "extract_symbols"), 
                "Should find extract_symbols method");
        
        let symbols_per_sec = symbols.len() as f64 / elapsed.as_secs_f64();
        println!("Performance: {:.1} symbols/second", symbols_per_sec);
    } else {
        println!("Warning: Could not read {}", parser_path);
    }
}

#[test]
fn test_symbol_database_functionality() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    let mut database = SymbolDatabase::new();
    
    let test_code = r#"
pub struct TestStruct {
    field: String,
}

impl TestStruct {
    pub fn new() -> Self {
        Self { field: String::new() }
    }
    
    pub fn method1(&self) -> &String {
        &self.field
    }
    
    pub fn method2(&mut self, value: String) {
        self.field = value;
    }
}

pub fn standalone_function() -> i32 {
    42
}
"#;
    
    let symbols = parser.extract_symbols(test_code, "test.rs")
        .expect("Failed to parse test code");
    
    database.add_symbols(symbols);
    
    // Test basic database operations
    assert_eq!(database.files_indexed(), 1);
    assert!(database.total_symbols() > 0);
    
    // Test finding references
    let test_struct_refs = database.find_all_references("TestStruct");
    assert!(!test_struct_refs.is_empty(), "Should find TestStruct references");
    
    // Test finding by kind
    let structs = database.find_by_kind(SymbolKind::Struct);
    assert!(!structs.is_empty(), "Should find struct symbols");
    
    let methods = database.find_by_kind(SymbolKind::Method);
    println!("Found {} methods in test code", methods.len());
    
    // Test symbols in file
    let file_symbols = database.symbols_in_file("test.rs");
    assert!(!file_symbols.is_empty(), "Should find symbols in test.rs");
    
    println!("✅ SymbolDatabase functionality tests passed");
    println!("   - Total symbols: {}", database.total_symbols());
    println!("   - Files indexed: {}", database.files_indexed());
}

#[test]
fn test_parsing_performance_metrics() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    
    // Test multiple source files for performance metrics
    let test_files = [
        "src/lib.rs",
        "src/error.rs",
        "src/config/mod.rs",
    ];
    
    let mut total_symbols = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    let mut total_lines = 0;
    let mut files_processed = 0;
    let mut failures = 0;
    
    for file_path in &test_files {
        if let Ok(content) = fs::read_to_string(file_path) {
            let line_count = content.lines().count();
            files_processed += 1;
            total_lines += line_count;
            
            let start = Instant::now();
            match parser.extract_symbols(&content, file_path) {
                Ok(symbols) => {
                    let elapsed = start.elapsed();
                    total_time += elapsed;
                    total_symbols += symbols.len();
                    
                    println!("✅ {}: {} symbols, {} lines, {:.2}ms", 
                             file_path, symbols.len(), line_count, elapsed.as_millis());
                }
                Err(e) => {
                    failures += 1;
                    println!("❌ {}: parsing failed - {}", file_path, e);
                }
            }
        } else {
            println!("⚠️  Could not read: {}", file_path);
        }
    }
    
    if files_processed > 0 {
        let symbols_per_sec = total_symbols as f64 / total_time.as_secs_f64();
        let lines_per_sec = total_lines as f64 / total_time.as_secs_f64();
        
        println!("\n=== FINAL PERFORMANCE REPORT ===");
        println!("Files processed: {}", files_processed);
        println!("Files failed: {} ({:.1}%)", failures, 
                 (failures as f64 / files_processed as f64) * 100.0);
        println!("Total symbols: {}", total_symbols);
        println!("Total lines: {}", total_lines);
        println!("Total time: {:.3}ms", total_time.as_millis());
        println!("Speed: {:.1} symbols/second", symbols_per_sec);
        println!("Line processing: {:.1} lines/second", lines_per_sec);
        
        // Performance assertions
        assert!(symbols_per_sec > 500.0, 
                "Parser should process at least 500 symbols/second, got {:.1}", symbols_per_sec);
        assert!(failures as f64 / files_processed as f64 < 0.2, 
                "Less than 20% of files should fail parsing");
        
        println!("✅ All performance targets met!");
    } else {
        println!("⚠️  No files were processed");
    }
}