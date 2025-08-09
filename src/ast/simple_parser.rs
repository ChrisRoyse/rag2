use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use regex::Regex;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
    pub parent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Enum,
    Impl,
    Trait,
    Type,
    Constant,
    Static,
    Module,
    Macro,
    Field,
}

pub struct SimpleRustParser {
    // Compiled regex patterns for performance
    function_regex: Regex,
    struct_regex: Regex,
    enum_regex: Regex,
    impl_regex: Regex,
    trait_regex: Regex,
    type_regex: Regex,
    const_regex: Regex,
    static_regex: Regex,
    mod_regex: Regex,
    macro_regex: Regex,
    field_regex: Regex,
}

impl SimpleRustParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            // Function definitions with visibility and async modifiers
            function_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?\s*\(")?,
            
            // Struct definitions with generic parameters
            struct_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?")?,
            
            // Enum definitions
            enum_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?")?,
            
            // Impl blocks (both inherent and trait impls)
            impl_regex: Regex::new(r"(?m)^[\s]*impl\s+(?:(?:<[^>]*>)\s+)?(?:([a-zA-Z_][a-zA-Z0-9_:]*)\s+for\s+)?([a-zA-Z_][a-zA-Z0-9_:]*)")?,
            
            // Trait definitions
            trait_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?")?,
            
            // Type aliases
            type_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?\s*=")?,
            
            // Constants
            const_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:")?,
            
            // Static variables
            static_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?static\s+(?:mut\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:")?,
            
            // Modules
            mod_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)")?,
            
            // Macro definitions
            macro_regex: Regex::new(r"(?m)^[\s]*macro_rules!\s+([a-zA-Z_][a-zA-Z0-9_]*)")?,
            
            // Struct/enum fields
            field_regex: Regex::new(r"(?m)^[\s]*(?:pub(?:\([^)]*\))?\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:")?,
        })
    }

    pub fn extract_symbols(&self, code: &str, file_path: &str) -> Result<Vec<Symbol>> {
        let mut symbols = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        // Track context for nested symbols
        let mut impl_context: Option<String> = None;
        let mut struct_context: Option<String> = None;
        let mut enum_context: Option<String> = None;
        let mut brace_depth = 0;
        let mut impl_depth = 0;
        
        // Process each pattern type
        self.extract_functions(code, file_path, &mut symbols)?;
        self.extract_structs(code, file_path, &mut symbols)?;
        self.extract_enums(code, file_path, &mut symbols)?;
        self.extract_impls(code, file_path, &mut symbols)?;
        self.extract_traits(code, file_path, &mut symbols)?;
        self.extract_types(code, file_path, &mut symbols)?;
        self.extract_constants(code, file_path, &mut symbols)?;
        self.extract_statics(code, file_path, &mut symbols)?;
        self.extract_modules(code, file_path, &mut symbols)?;
        self.extract_macros(code, file_path, &mut symbols)?;
        
        // Extract methods and associated functions from impl blocks
        self.extract_impl_methods(code, file_path, &mut symbols)?;
        
        // Deduplicate symbols based on name, kind, and line
        symbols.sort_by(|a, b| {
            (a.line_start, &a.name, &a.kind).cmp(&(b.line_start, &b.name, &b.kind))
        });
        symbols.dedup_by(|a, b| {
            a.name == b.name && a.kind == b.kind && a.line_start == b.line_start
        });
        
        Ok(symbols)
    }

    fn extract_functions(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.function_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            
            // Extract the full signature (simplified)
            let signature = self.extract_function_signature(code, match_start)?;
            let line_end = self.find_function_end(code, match_start, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Function,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_structs(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.struct_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = cap.get(0).unwrap().as_str().trim_end_matches('{').trim().to_string();
            let line_end = self.find_block_end(code, match_start, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Struct,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_enums(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.enum_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = cap.get(0).unwrap().as_str().trim().to_string();
            let line_end = self.find_block_end(code, match_start, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Enum,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_impls(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.impl_regex.captures_iter(code) {
            // For trait impls: cap[1] is trait, cap[2] is type
            // For inherent impls: cap[1] is None, cap[2] is type
            let name = if cap.get(1).is_some() {
                format!("{} for {}", cap.get(1).unwrap().as_str(), cap.get(2).unwrap().as_str())
            } else {
                cap.get(2).unwrap().as_str().to_string()
            };
            
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = cap.get(0).unwrap().as_str().trim().to_string();
            let line_end = self.find_block_end(code, match_start, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Impl,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_traits(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.trait_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = cap.get(0).unwrap().as_str().trim().to_string();
            let line_end = self.find_block_end(code, match_start, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Trait,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_types(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.type_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = self.extract_line(code, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Type,
                file_path: file_path.to_string(),
                line_start,
                line_end: line_start,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_constants(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.const_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = self.extract_line(code, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Constant,
                file_path: file_path.to_string(),
                line_start,
                line_end: line_start,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_statics(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.static_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = self.extract_line(code, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Static,
                file_path: file_path.to_string(),
                line_start,
                line_end: line_start,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_modules(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.mod_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let signature = self.extract_line(code, line_start);
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Module,
                file_path: file_path.to_string(),
                line_start,
                line_end: line_start,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_macros(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        for cap in self.macro_regex.captures_iter(code) {
            let name = cap[1].to_string();
            let match_start = cap.get(0).unwrap().start();
            let line_start = self.get_line_number(code, match_start);
            let line_end = self.find_macro_end(code, match_start, line_start);
            let signature = cap.get(0).unwrap().as_str().to_string();
            
            symbols.push(Symbol {
                name,
                kind: SymbolKind::Macro,
                file_path: file_path.to_string(),
                line_start,
                line_end,
                signature: Some(signature),
                parent: None,
            });
        }
        Ok(())
    }

    fn extract_impl_methods(&self, code: &str, file_path: &str, symbols: &mut Vec<Symbol>) -> Result<()> {
        // Find impl blocks and extract methods from them
        for impl_match in self.impl_regex.captures_iter(code) {
            let impl_start = impl_match.get(0).unwrap().start();
            let impl_line_start = self.get_line_number(code, impl_start);
            let impl_line_end = self.find_block_end(code, impl_start, impl_line_start);
            
            let impl_name = if impl_match.get(1).is_some() {
                format!("{} for {}", impl_match.get(1).unwrap().as_str(), impl_match.get(2).unwrap().as_str())
            } else {
                impl_match.get(2).unwrap().as_str().to_string()
            };
            
            // Extract the impl block content
            let lines: Vec<&str> = code.lines().collect();
            if impl_line_start <= lines.len() && impl_line_end <= lines.len() {
                let impl_content = lines[impl_line_start-1..impl_line_end.min(lines.len())].join("\n");
                
                // Find functions within this impl block
                for fn_match in self.function_regex.captures_iter(&impl_content) {
                    let method_name = fn_match[1].to_string();
                    let method_start_in_impl = fn_match.get(0).unwrap().start();
                    let method_line_in_impl = impl_content[..method_start_in_impl].matches('\n').count() + 1;
                    let method_line_start = impl_line_start + method_line_in_impl - 1;
                    
                    let signature = self.extract_function_signature(&impl_content, method_start_in_impl)?;
                    let method_line_end = self.find_function_end_in_block(&impl_content, method_start_in_impl, method_line_in_impl) + impl_line_start - 1;
                    
                    symbols.push(Symbol {
                        name: method_name,
                        kind: SymbolKind::Method,
                        file_path: file_path.to_string(),
                        line_start: method_line_start,
                        line_end: method_line_end,
                        signature: Some(signature),
                        parent: Some(impl_name.clone()),
                    });
                }
            }
        }
        Ok(())
    }

    // Helper methods
    fn get_line_number(&self, code: &str, byte_pos: usize) -> usize {
        code[..byte_pos].matches('\n').count() + 1
    }

    fn extract_line(&self, code: &str, line_num: usize) -> String {
        code.lines().nth(line_num - 1).unwrap_or("").trim().to_string()
    }

    fn extract_function_signature(&self, code: &str, start_pos: usize) -> Result<String> {
        // Find the complete function signature (until the opening brace or semicolon)
        let remaining = &code[start_pos..];
        if let Some(brace_pos) = remaining.find('{') {
            Ok(remaining[..brace_pos].trim().to_string())
        } else if let Some(semi_pos) = remaining.find(';') {
            Ok(remaining[..semi_pos].trim().to_string())
        } else {
            // Fallback: take until newline
            Ok(remaining.lines().next().unwrap_or("").trim().to_string())
        }
    }

    fn find_function_end(&self, code: &str, start_pos: usize, start_line: usize) -> usize {
        let remaining = &code[start_pos..];
        if let Some(brace_start) = remaining.find('{') {
            self.find_matching_brace_end(code, start_pos + brace_start, start_line)
        } else {
            // Function declaration without body
            start_line
        }
    }

    fn find_function_end_in_block(&self, code: &str, start_pos: usize, start_line: usize) -> usize {
        let remaining = &code[start_pos..];
        if let Some(brace_start) = remaining.find('{') {
            let abs_brace_pos = start_pos + brace_start;
            let mut brace_count = 1;
            let mut current_pos = abs_brace_pos + 1;
            
            while current_pos < code.len() && brace_count > 0 {
                match code.chars().nth(current_pos) {
                    Some('{') => brace_count += 1,
                    Some('}') => brace_count -= 1,
                    _ => {}
                }
                current_pos += 1;
            }
            
            self.get_line_number(code, current_pos.saturating_sub(1))
        } else {
            start_line
        }
    }

    fn find_block_end(&self, code: &str, start_pos: usize, start_line: usize) -> usize {
        let remaining = &code[start_pos..];
        if let Some(brace_start) = remaining.find('{') {
            self.find_matching_brace_end(code, start_pos + brace_start, start_line)
        } else {
            start_line
        }
    }

    fn find_macro_end(&self, code: &str, start_pos: usize, start_line: usize) -> usize {
        // Macro definitions are more complex, use a simple heuristic
        let remaining = &code[start_pos..];
        if let Some(brace_start) = remaining.find('{') {
            self.find_matching_brace_end(code, start_pos + brace_start, start_line)
        } else {
            start_line
        }
    }

    fn find_matching_brace_end(&self, code: &str, brace_start: usize, _start_line: usize) -> usize {
        let mut brace_count = 1;
        let mut current_pos = brace_start + 1;
        let mut in_string = false;
        let mut in_char = false;
        let mut escape_next = false;
        
        while current_pos < code.len() && brace_count > 0 {
            let ch = code.chars().nth(current_pos).unwrap_or('\0');
            
            if escape_next {
                escape_next = false;
            } else {
                match ch {
                    '\\' if in_string || in_char => escape_next = true,
                    '"' if !in_char => in_string = !in_string,
                    '\'' if !in_string => in_char = !in_char,
                    '{' if !in_string && !in_char => brace_count += 1,
                    '}' if !in_string && !in_char => brace_count -= 1,
                    _ => {}
                }
            }
            current_pos += 1;
        }
        
        self.get_line_number(code, current_pos.saturating_sub(1))
    }

    pub fn detect_language(path: &Path) -> Option<&'static str> {
        match path.extension()?.to_str()? {
            "rs" => Some("rust"),
            _ => None,
        }
    }
}

// Symbol database for fast lookups and reference finding
pub struct SymbolDatabase {
    pub symbols_by_name: FxHashMap<String, Vec<Symbol>>,
    pub symbols_by_file: FxHashMap<String, Vec<Symbol>>,
    pub symbols_by_kind: FxHashMap<SymbolKind, Vec<Symbol>>,
}

impl SymbolDatabase {
    pub fn new() -> Self {
        Self {
            symbols_by_name: FxHashMap::default(),
            symbols_by_file: FxHashMap::default(),
            symbols_by_kind: FxHashMap::default(),
        }
    }
    
    pub fn add_symbols(&mut self, symbols: Vec<Symbol>) {
        for symbol in symbols {
            // Index by name
            self.symbols_by_name
                .entry(symbol.name.clone())
                .or_insert_with(Vec::new)
                .push(symbol.clone());
            
            // Index by file
            self.symbols_by_file
                .entry(symbol.file_path.clone())
                .or_insert_with(Vec::new)
                .push(symbol.clone());
            
            // Index by kind
            self.symbols_by_kind
                .entry(symbol.kind.clone())
                .or_insert_with(Vec::new)
                .push(symbol);
        }
    }
    
    pub fn find_definition(&self, name: &str) -> Option<Symbol> {
        self.symbols_by_name.get(name)?.first().cloned()
    }
    
    pub fn find_all_references(&self, name: &str) -> Vec<Symbol> {
        self.symbols_by_name.get(name).map(|v| v.clone()).unwrap_or_default()
    }
    
    pub fn clear(&mut self) {
        self.symbols_by_name.clear();
        self.symbols_by_file.clear();
        self.symbols_by_kind.clear();
    }
    
    pub fn remove_file_symbols(&mut self, file_path: &str) {
        let symbols_to_remove = match self.symbols_by_file.remove(file_path) {
            Some(symbols) => symbols,
            None => return,
        };
        
        for symbol in &symbols_to_remove {
            if let Some(name_symbols) = self.symbols_by_name.get_mut(&symbol.name) {
                name_symbols.retain(|s| s.file_path != file_path);
                if name_symbols.is_empty() {
                    self.symbols_by_name.remove(&symbol.name);
                }
            }
        }
        
        for symbol in &symbols_to_remove {
            if let Some(kind_symbols) = self.symbols_by_kind.get_mut(&symbol.kind) {
                kind_symbols.retain(|s| s.file_path != file_path);
                if kind_symbols.is_empty() {
                    self.symbols_by_kind.remove(&symbol.kind);
                }
            }
        }
    }
    
    pub fn find_by_kind(&self, kind: SymbolKind) -> Vec<Symbol> {
        self.symbols_by_kind.get(&kind).map(|v| v.clone()).unwrap_or_default()
    }
    
    pub fn symbols_in_file(&self, file_path: &str) -> Vec<&Symbol> {
        self.symbols_by_file.get(file_path).map(|v| v.iter().collect()).unwrap_or_default()
    }
    
    pub fn total_symbols(&self) -> usize {
        self.symbols_by_file.values().map(|v| v.len()).sum()
    }
    
    pub fn files_indexed(&self) -> usize {
        self.symbols_by_file.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_extraction() {
        let code = r#"
        pub fn main() {
            println!("Hello");
        }
        
        async fn fetch_data() -> Result<String> {
            Ok("data".to_string())
        }
        
        unsafe fn raw_memory() {}
        "#;
        
        let parser = SimpleRustParser::new().unwrap();
        let symbols = parser.extract_symbols(code, "test.rs").unwrap();
        
        let functions: Vec<_> = symbols.iter().filter(|s| s.kind == SymbolKind::Function).collect();
        assert_eq!(functions.len(), 3);
        assert!(functions.iter().any(|s| s.name == "main"));
        assert!(functions.iter().any(|s| s.name == "fetch_data"));
        assert!(functions.iter().any(|s| s.name == "raw_memory"));
    }

    #[test]
    fn test_rust_struct_extraction() {
        let code = r#"
        pub struct UserAuth {
            username: String,
            password: String,
        }
        
        struct Point<T> {
            x: T,
            y: T,
        }
        "#;
        
        let parser = SimpleRustParser::new().unwrap();
        let symbols = parser.extract_symbols(code, "test.rs").unwrap();
        
        let structs: Vec<_> = symbols.iter().filter(|s| s.kind == SymbolKind::Struct).collect();
        assert_eq!(structs.len(), 2);
        assert!(structs.iter().any(|s| s.name == "UserAuth"));
        assert!(structs.iter().any(|s| s.name == "Point"));
    }

    #[test]
    fn test_rust_impl_and_methods() {
        let code = r#"
        struct UserAuth {
            username: String,
        }
        
        impl UserAuth {
            pub fn new(username: String) -> Self {
                Self { username }
            }
            
            pub fn validate(&self) -> bool {
                !self.username.is_empty()
            }
        }
        
        impl Clone for UserAuth {
            fn clone(&self) -> Self {
                Self { username: self.username.clone() }
            }
        }
        "#;
        
        let parser = SimpleRustParser::new().unwrap();
        let symbols = parser.extract_symbols(code, "test.rs").unwrap();
        
        let impls: Vec<_> = symbols.iter().filter(|s| s.kind == SymbolKind::Impl).collect();
        let methods: Vec<_> = symbols.iter().filter(|s| s.kind == SymbolKind::Method).collect();
        
        assert_eq!(impls.len(), 2);
        assert_eq!(methods.len(), 3);
        assert!(methods.iter().any(|s| s.name == "new"));
        assert!(methods.iter().any(|s| s.name == "validate"));
        assert!(methods.iter().any(|s| s.name == "clone"));
    }

    #[test]
    fn test_symbol_database_operations() {
        let symbols = vec![
            Symbol {
                name: "test_fn".to_string(),
                kind: SymbolKind::Function,
                file_path: "test.rs".to_string(),
                line_start: 1,
                line_end: 3,
                signature: Some("fn test_fn()".to_string()),
                parent: None,
            },
            Symbol {
                name: "test_fn".to_string(),
                kind: SymbolKind::Function,
                file_path: "other.rs".to_string(),
                line_start: 1,
                line_end: 3,
                signature: Some("fn test_fn()".to_string()),
                parent: None,
            },
        ];

        let mut db = SymbolDatabase::new();
        db.add_symbols(symbols);

        let refs = db.find_all_references("test_fn");
        assert_eq!(refs.len(), 2);

        let def = db.find_definition("test_fn");
        assert!(def.is_some());
        assert_eq!(def.unwrap().name, "test_fn");
    }
}