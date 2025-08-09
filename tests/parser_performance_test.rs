use embed_search::ast::{SimpleRustParser, SymbolDatabase};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_parser_performance_on_real_rust_code() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    let mut total_symbols = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    let mut total_lines = 0;
    let mut files_processed = 0;
    let mut failed_files = 0;

    println!("Testing SimpleRustParser performance on real Rust code...");
    
    // Test on src directory
    test_directory(&parser, "src", &mut total_symbols, &mut total_time, &mut total_lines, &mut files_processed, &mut failed_files);
    
    if files_processed > 0 {
        let avg_time_per_file = total_time / files_processed as u32;
        let symbols_per_second = if total_time.as_secs_f64() > 0.0 {
            total_symbols as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };
        let lines_per_second = if total_time.as_secs_f64() > 0.0 {
            total_lines as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };
        
        println!("\n=== PERFORMANCE REPORT ===");
        println!("Files processed: {}", files_processed);
        println!("Files failed: {} ({:.1}%)", failed_files, (failed_files as f64 / files_processed as f64) * 100.0);
        println!("Total symbols extracted: {}", total_symbols);
        println!("Total lines processed: {}", total_lines);
        println!("Total time: {:.3}ms", total_time.as_millis());
        println!("Average time per file: {:.3}ms", avg_time_per_file.as_millis());
        println!("Parsing speed: {:.1} symbols/second", symbols_per_second);
        println!("Line processing speed: {:.1} lines/second", lines_per_second);
        
        // Performance expectations
        assert!(symbols_per_second > 1000.0, "Parser should process at least 1000 symbols/second, got {:.1}", symbols_per_second);
        assert!(lines_per_second > 10000.0, "Parser should process at least 10000 lines/second, got {:.1}", lines_per_second);
        assert!(failed_files as f64 / files_processed as f64 < 0.1, "Less than 10% of files should fail parsing");
    } else {
        println!("No Rust files found to test");
    }
}

fn test_directory(
    parser: &SimpleRustParser,
    dir_path: &str,
    total_symbols: &mut usize,
    total_time: &mut std::time::Duration,
    total_lines: &mut usize,
    files_processed: &mut usize,
    failed_files: &mut usize,
) {
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            if path.is_dir() {
                // Recursively process subdirectories
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if !dir_name.starts_with('.') { // Skip hidden directories
                        test_directory(
                            parser,
                            &path.to_string_lossy(),
                            total_symbols,
                            total_time,
                            total_lines,
                            files_processed,
                            failed_files,
                        );
                    }
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                test_rust_file(parser, &path, total_symbols, total_time, total_lines, files_processed, failed_files);
            }
        }
    }
}

fn test_rust_file(
    parser: &SimpleRustParser,
    file_path: &Path,
    total_symbols: &mut usize,
    total_time: &mut std::time::Duration,
    total_lines: &mut usize,
    files_processed: &mut usize,
    failed_files: &mut usize,
) {
    *files_processed += 1;
    
    match fs::read_to_string(file_path) {
        Ok(content) => {
            let line_count = content.lines().count();
            *total_lines += line_count;
            
            let start = Instant::now();
            match parser.extract_symbols(&content, &file_path.to_string_lossy()) {
                Ok(symbols) => {
                    let elapsed = start.elapsed();
                    *total_time += elapsed;
                    *total_symbols += symbols.len();
                    
                    println!("✓ {}: {} symbols, {} lines, {:.2}ms", 
                             file_path.to_string_lossy(), 
                             symbols.len(), 
                             line_count,
                             elapsed.as_millis());
                    
                    // Log some example symbols for verification
                    if symbols.len() > 0 {
                        for (i, symbol) in symbols.iter().take(3).enumerate() {
                            println!("  [{i}] {:?} '{}' at line {}", symbol.kind, symbol.name, symbol.line_start);
                        }
                        if symbols.len() > 3 {
                            println!("  ... and {} more symbols", symbols.len() - 3);
                        }
                    }
                }
                Err(e) => {
                    *failed_files += 1;
                    println!("✗ Failed to parse {}: {}", file_path.to_string_lossy(), e);
                }
            }
        }
        Err(e) => {
            *failed_files += 1;
            println!("✗ Failed to read {}: {}", file_path.to_string_lossy(), e);
        }
    }
}

#[test] 
fn test_symbol_database_with_real_code() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    let mut db = SymbolDatabase::new();
    
    // Test with a real Rust file
    let test_file = "src/lib.rs";
    if let Ok(content) = fs::read_to_string(test_file) {
        match parser.extract_symbols(&content, test_file) {
            Ok(symbols) => {
                let symbol_count = symbols.len();
                db.add_symbols(symbols);
                
                println!("Added {} symbols to database", symbol_count);
                println!("Total symbols in database: {}", db.total_symbols());
                println!("Files indexed: {}", db.files_indexed());
                
                // Test find_all_references functionality
                if let Some(first_symbol) = db.symbols_in_file(test_file).first() {
                    let refs = db.find_all_references(&first_symbol.name);
                    println!("References for '{}': {}", first_symbol.name, refs.len());
                    
                    assert!(!refs.is_empty(), "Should find at least the definition itself");
                }
            }
            Err(e) => {
                panic!("Failed to parse {}: {}", test_file, e);
            }
        }
    } else {
        println!("Warning: Could not read {} for testing", test_file);
    }
}

#[test]
fn test_parser_on_complex_rust_patterns() {
    let parser = SimpleRustParser::new().expect("Failed to create parser");
    
    let complex_code = r#"
// This is a complex Rust file to test edge cases
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GenericStruct<T, U> 
where 
    T: Clone,
    U: std::fmt::Debug,
{
    pub field1: T,
    field2: Option<U>,
}

pub enum ComplexEnum<T> {
    Variant1(T),
    Variant2 { 
        named_field: String, 
        another_field: u32 
    },
    Variant3,
}

impl<T, U> GenericStruct<T, U> 
where 
    T: Clone + Send,
    U: std::fmt::Debug + Sync,
{
    pub fn new(field1: T, field2: Option<U>) -> Self {
        Self { field1, field2 }
    }
    
    pub async fn async_method(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok("test".to_string())
    }
    
    unsafe fn unsafe_method(&self) -> *const T {
        &self.field1 as *const T
    }
}

impl<T> Clone for ComplexEnum<T> 
where 
    T: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Variant1(v) => Self::Variant1(v.clone()),
            Self::Variant2 { named_field, another_field } => {
                Self::Variant2 { 
                    named_field: named_field.clone(), 
                    another_field: *another_field 
                }
            }
            Self::Variant3 => Self::Variant3,
        }
    }
}

pub trait ComplexTrait<T> {
    type AssociatedType;
    
    fn required_method(&self, param: T) -> Self::AssociatedType;
    
    fn default_method(&self) -> String {
        "default".to_string()
    }
}

macro_rules! complex_macro {
    ($name:ident, $type:ty) => {
        pub struct $name {
            value: $type,
        }
    };
}

complex_macro!(GeneratedStruct, i32);

pub const COMPLEX_CONSTANT: &str = "test";
pub static mut GLOBAL_STATE: Option<String> = None;

pub mod inner_module {
    pub fn module_function() -> i32 {
        42
    }
}

pub type ComplexTypeAlias<T> = Result<HashMap<String, T>, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_something() {
        let _instance = GenericStruct::new(42, Some("test"));
    }
}
"#;

    let start = Instant::now();
    match parser.extract_symbols(complex_code, "complex_test.rs") {
        Ok(symbols) => {
            let elapsed = start.elapsed();
            
            println!("Complex code parsing results:");
            println!("Symbols found: {}", symbols.len());
            println!("Parse time: {:.2}ms", elapsed.as_millis());
            
            let mut kind_counts = std::collections::HashMap::new();
            for symbol in &symbols {
                *kind_counts.entry(symbol.kind.clone()).or_insert(0) += 1;
            }
            
            println!("Symbol breakdown:");
            for (kind, count) in kind_counts {
                println!("  {:?}: {}", kind, count);
            }
            
            // Verify we found expected symbols
            assert!(symbols.iter().any(|s| s.name == "GenericStruct"), "Should find GenericStruct");
            assert!(symbols.iter().any(|s| s.name == "ComplexEnum"), "Should find ComplexEnum");
            assert!(symbols.iter().any(|s| s.name == "new"), "Should find new method");
            assert!(symbols.iter().any(|s| s.name == "async_method"), "Should find async_method");
            assert!(symbols.iter().any(|s| s.name == "ComplexTrait"), "Should find ComplexTrait");
            assert!(symbols.iter().any(|s| s.name == "complex_macro"), "Should find complex_macro");
            
            // Test that methods have parent information
            let methods: Vec<_> = symbols.iter().filter(|s| s.kind == embed_search::ast::SymbolKind::Method).collect();
            assert!(!methods.is_empty(), "Should find methods");
            
            for method in methods {
                println!("Method '{}' parent: {:?}", method.name, method.parent);
            }
            
            assert!(symbols.len() >= 10, "Should find at least 10 symbols in complex code");
        }
        Err(e) => {
            panic!("Failed to parse complex code: {}", e);
        }
    }
}