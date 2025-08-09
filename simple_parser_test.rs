// Simple standalone test for the parser

use std::time::Instant;

fn main() {
    println!("=== SimpleRustParser Standalone Test ===");
    
    // Create parser
    let parser = match embed_search::ast::SimpleRustParser::new() {
        Ok(p) => p,
        Err(e) => {
            println!("❌ Failed to create parser: {}", e);
            return;
        }
    };
    
    // Test with sample Rust code
    let test_code = r#"
use std::collections::HashMap;

pub struct TestStruct {
    pub field1: String,
    field2: i32,
}

impl TestStruct {
    pub fn new(field1: String, field2: i32) -> Self {
        Self { field1, field2 }
    }
    
    pub fn get_field1(&self) -> &String {
        &self.field1
    }
}

pub enum TestEnum {
    Variant1,
    Variant2(String),
    Variant3 { name: String, value: i32 },
}

pub fn standalone_function() -> Result<String, Box<dyn std::error::Error>> {
    Ok("test".to_string())
}

pub const TEST_CONSTANT: &str = "hello";
pub static TEST_STATIC: i32 = 42;

pub trait TestTrait {
    fn required_method(&self) -> String;
}

macro_rules! test_macro {
    ($name:ident) => {
        struct $name;
    };
}

pub mod test_module {
    pub fn module_function() {}
}

pub type TestType = HashMap<String, i32>;
"#;
    
    println!("Testing with {} lines of code...", test_code.lines().count());
    
    let start = Instant::now();
    match parser.extract_symbols(test_code, "test.rs") {
        Ok(symbols) => {
            let elapsed = start.elapsed();
            
            println!("✅ Parsing successful!");
            println!("📊 Results:");
            println!("  - Symbols found: {}", symbols.len());
            println!("  - Parse time: {:.2}ms", elapsed.as_millis());
            println!("  - Speed: {:.1} symbols/sec", symbols.len() as f64 / elapsed.as_secs_f64());
            
            println!("\n🔍 Symbol breakdown:");
            let mut kind_counts = std::collections::HashMap::new();
            for symbol in &symbols {
                *kind_counts.entry(format!("{:?}", symbol.kind)).or_insert(0) += 1;
            }
            
            for (kind, count) in kind_counts {
                println!("  - {}: {}", kind, count);
            }
            
            println!("\n📋 First 10 symbols:");
            for (i, symbol) in symbols.iter().take(10).enumerate() {
                println!("  [{:2}] {:12} '{}' @ line {}", 
                         i + 1, 
                         format!("{:?}", symbol.kind), 
                         symbol.name, 
                         symbol.line_start);
                if let Some(parent) = &symbol.parent {
                    println!("       └── parent: {}", parent);
                }
            }
            
            // Test database functionality
            let mut db = embed_search::ast::SymbolDatabase::new();
            db.add_symbols(symbols);
            
            println!("\n💾 Database test:");
            println!("  - Total symbols: {}", db.total_symbols());
            println!("  - Files indexed: {}", db.files_indexed());
            
            // Test find_all_references
            if let Some(first_symbol) = db.symbols_in_file("test.rs").first() {
                let refs = db.find_all_references(&first_symbol.name);
                println!("  - References for '{}': {}", first_symbol.name, refs.len());
            }
            
            println!("\n🎯 PERFORMANCE REPORT:");
            println!("✅ Parser created successfully");
            println!("✅ Symbols extracted: {} symbols", db.total_symbols());
            println!("✅ No parsing failures");
            println!("✅ Speed: {:.1} symbols/second", db.total_symbols() as f64 / elapsed.as_secs_f64());
            
            if elapsed.as_millis() < 10 {
                println!("✅ Fast parsing: under 10ms");
            }
        }
        Err(e) => {
            println!("❌ Parsing failed: {}", e);
        }
    }
    
    // Test on actual source files if available
    println!("\n=== Testing on Real Source Files ===");
    test_real_files(&parser);
}

fn test_real_files(parser: &embed_search::ast::SimpleRustParser) {
    use std::fs;
    
    let test_files = [
        "src/lib.rs",
        "src/ast/simple_parser.rs", 
        "src/error.rs"
    ];
    
    let mut total_symbols = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    let mut files_tested = 0;
    let mut failures = 0;
    
    for file_path in &test_files {
        if let Ok(content) = fs::read_to_string(file_path) {
            let line_count = content.lines().count();
            files_tested += 1;
            
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
    
    if files_tested > 0 {
        let avg_speed = total_symbols as f64 / total_time.as_secs_f64();
        println!("\n📊 Real Files Summary:");
        println!("  - Files tested: {}", files_tested);
        println!("  - Failures: {} ({:.1}%)", failures, (failures as f64 / files_tested as f64) * 100.0);
        println!("  - Total symbols: {}", total_symbols);
        println!("  - Average speed: {:.1} symbols/second", avg_speed);
        
        if avg_speed > 1000.0 {
            println!("✅ Performance target met: > 1000 symbols/second");
        } else {
            println!("⚠️  Performance below target: {} symbols/second < 1000", avg_speed);
        }
    }
}