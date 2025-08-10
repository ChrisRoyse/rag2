use rustc_hash::FxHashMap;
use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use regex::Regex;

// TODO: Replace with proper tree-sitter implementation once dependencies are available
// This is a temporary regex-based implementation to fix compilation errors

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
    pub parent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Interface,
    Enum,
    Variable,
    Constant,
    Type,
    Module,
    Namespace,
    Property,
    Field,
    Constructor,
    Parameter,
    TypeParameter,
    Label,
    Tag,  // For HTML/XML
    Selector,  // For CSS
    Key,  // For JSON/YAML/TOML
}

// TODO: Replace with tree-sitter parsers once dependencies are available
pub struct SymbolIndexer {
    // Placeholder - will contain regex patterns for each language
    regex_patterns: FxHashMap<String, Vec<(Regex, SymbolKind)>>,
}

impl SymbolIndexer {
    pub fn new() -> Result<Self> {
        let mut regex_patterns = FxHashMap::default();
        
        // TODO: Replace with proper tree-sitter parsers once dependencies are available
        // Initialize basic regex patterns for common symbols
        Self::init_regex_patterns(&mut regex_patterns)?;
        
        Ok(Self { regex_patterns })
    }
    
    // TODO: Replace with tree-sitter once available - basic regex symbol detection
    fn init_regex_patterns(patterns: &mut FxHashMap<String, Vec<(Regex, SymbolKind)>>) -> Result<()> {
        // Rust patterns
        let rust_patterns = vec![
            (Regex::new(r"(?m)^pub\s+fn\s+(\w+)")?, SymbolKind::Function),
            (Regex::new(r"(?m)^fn\s+(\w+)")?, SymbolKind::Function),
            (Regex::new(r"(?m)^pub\s+struct\s+(\w+)")?, SymbolKind::Struct),
            (Regex::new(r"(?m)^struct\s+(\w+)")?, SymbolKind::Struct),
            (Regex::new(r"(?m)^pub\s+enum\s+(\w+)")?, SymbolKind::Enum),
            (Regex::new(r"(?m)^enum\s+(\w+)")?, SymbolKind::Enum),
        ];
        patterns.insert("rust".to_string(), rust_patterns);
        
        // Python patterns
        let python_patterns = vec![
            (Regex::new(r"(?m)^def\s+(\w+)")?, SymbolKind::Function),
            (Regex::new(r"(?m)^class\s+(\w+)")?, SymbolKind::Class),
        ];
        patterns.insert("python".to_string(), python_patterns);
        
        // JavaScript/TypeScript patterns
        let js_patterns = vec![
            (Regex::new(r"(?m)^function\s+(\w+)")?, SymbolKind::Function),
            (Regex::new(r"(?m)^class\s+(\w+)")?, SymbolKind::Class),
            (Regex::new(r"(?m)const\s+(\w+)\s*=")?, SymbolKind::Variable),
        ];
        patterns.insert("javascript".to_string(), js_patterns.clone());
        patterns.insert("js".to_string(), js_patterns.clone());
        patterns.insert("typescript".to_string(), js_patterns.clone());
        patterns.insert("ts".to_string(), js_patterns);
        
        Ok(())
    }
    
    // TODO: Replace with tree-sitter implementation once dependencies are available
    pub fn extract_symbols(&mut self, code: &str, language: &str, file_path: &str) -> Result<Vec<Symbol>> {
        let lang = language.to_lowercase();
        
        let patterns = self.regex_patterns.get(&lang)
            .ok_or_else(|| anyhow::anyhow!("Unsupported language: {}", language))?;
        
        let mut symbols = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let lines: Vec<&str> = code.lines().collect();
        
        for (pattern, kind) in patterns {
            for (line_num, line) in lines.iter().enumerate() {
                if let Some(captures) = pattern.captures(line) {
                    if let Some(name_match) = captures.get(1) {
                        let name = name_match.as_str().to_string();
                        
                        // Skip empty or invalid names
                        if name.is_empty() || name.contains('@') {
                            continue;
                        }
                        
                        let symbol = Symbol {
                            name: name.clone(),
                            kind: kind.clone(),
                            file_path: file_path.to_string(),
                            line_start: line_num + 1,
                            line_end: line_num + 1,
                            signature: None,
                            parent: None,
                        };
                        
                        // Deduplicate symbols
                        let key = format!("{}:{}:{}", symbol.file_path, symbol.name, symbol.line_start);
                        if seen.insert(key) {
                            symbols.push(symbol);
                        }
                    }
                }
            }
        }
        
        Ok(symbols)
    }
    
    pub fn detect_language(path: &Path) -> Option<&'static str> {
        match path.extension()?.to_str()? {
            "rs" => Some("rust"),
            "py" => Some("python"),
            "js" | "mjs" | "cjs" => Some("javascript"),
            "ts" => Some("typescript"),
            "tsx" => Some("tsx"),
            "go" => Some("go"),
            "java" => Some("java"),
            "c" => Some("c"),
            "h" => Some("h"),
            "cpp" | "cc" | "cxx" | "c++" => Some("cpp"),
            "hpp" | "hh" | "hxx" | "h++" => Some("hpp"),
            "html" | "htm" => Some("html"),
            "css" => Some("css"),
            "scss" | "sass" => Some("scss"),
            "json" => Some("json"),
            "sh" => Some("sh"),
            "bash" => Some("bash"),
            _ => None,
        }
    }
}

// Quick symbol database for fast lookups
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
        match self.symbols_by_name.get(name) {
            Some(symbols) => symbols.iter().cloned().collect(),
            None => {
                // Symbol name not found in index - legitimate empty result
                Vec::new()
            }
        }
    }
    
    pub fn clear(&mut self) {
        self.symbols_by_name.clear();
        self.symbols_by_file.clear();
        self.symbols_by_kind.clear();
    }
    
    /// Remove all symbols from a specific file
    pub fn remove_file_symbols(&mut self, file_path: &str) {
        // Get symbols that need to be removed from this file
        let symbols_to_remove = match self.symbols_by_file.remove(file_path) {
            Some(symbols) => symbols,
            None => return, // File not indexed, nothing to remove
        };
        
        // Remove symbols from name-based index
        for symbol in &symbols_to_remove {
            if let Some(name_symbols) = self.symbols_by_name.get_mut(&symbol.name) {
                name_symbols.retain(|s| s.file_path != file_path);
                // Remove the name entry if no symbols left
                if name_symbols.is_empty() {
                    self.symbols_by_name.remove(&symbol.name);
                }
            }
        }
        
        // Remove symbols from kind-based index
        for symbol in &symbols_to_remove {
            if let Some(kind_symbols) = self.symbols_by_kind.get_mut(&symbol.kind) {
                kind_symbols.retain(|s| s.file_path != file_path);
                // Remove the kind entry if no symbols left
                if kind_symbols.is_empty() {
                    self.symbols_by_kind.remove(&symbol.kind);
                }
            }
        }
    }
    
    pub fn find_by_kind(&self, kind: SymbolKind) -> Vec<Symbol> {
        match self.symbols_by_kind.get(&kind) {
            Some(symbols) => symbols.iter().cloned().collect(),
            None => {
                // Symbol kind not found in index - legitimate empty result
                Vec::new()
            }
        }
    }
    
    pub fn symbols_in_file(&self, file_path: &str) -> Vec<&Symbol> {
        match self.symbols_by_file.get(file_path) {
            Some(symbols) => symbols.iter().collect(),
            None => {
                // File not found in index - legitimate empty result
                Vec::new()
            }
        }
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
    fn test_rust_symbol_extraction() {
        let code = r#"
        pub struct UserAuth {
            username: String,
        }
        
        impl UserAuth {
            pub fn validate(&self) -> bool {
                true
            }
        }
        
        fn main() {
            println!("Hello");
        }
        "#;
        
        let mut indexer = SymbolIndexer::new().unwrap();
        let symbols = indexer.extract_symbols(code, "rust", "test.rs").unwrap();
        
        assert!(symbols.iter().any(|s| s.name == "UserAuth" && s.kind == SymbolKind::Struct));
        assert!(symbols.iter().any(|s| s.name == "main" && s.kind == SymbolKind::Function));
    }
    
    #[test]
    fn test_python_symbol_extraction() {
        let code = r#"
class AuthManager:
    def __init__(self):
        pass
    
    def validate_user(self, username, password):
        return True

def main():
    manager = AuthManager()
    print("Starting")
        "#;
        
        let mut indexer = SymbolIndexer::new().unwrap();
        let symbols = indexer.extract_symbols(code, "python", "test.py").unwrap();
        
        assert!(symbols.iter().any(|s| s.name == "AuthManager" && s.kind == SymbolKind::Class));
        assert!(symbols.iter().any(|s| s.name == "main" && s.kind == SymbolKind::Function));
    }
    
    #[test]
    fn test_java_symbol_extraction() {
        let code = r#"
        public class UserService {
            private String name;
            
            public UserService(String name) {
                this.name = name;
            }
            
            public void processUser() {
                System.out.println("Processing");
            }
        }
        "#;
        
        let mut indexer = SymbolIndexer::new().unwrap();
        let symbols = indexer.extract_symbols(code, "java", "UserService.java").unwrap();
        
        assert!(symbols.iter().any(|s| s.name == "UserService" && s.kind == SymbolKind::Class));
        assert!(symbols.iter().any(|s| s.name == "processUser" && s.kind == SymbolKind::Method));
    }
    
    #[test]
    fn test_c_symbol_extraction() {
        let code = r#"
        typedef struct {
            int x;
            int y;
        } Point;
        
        int add(int a, int b) {
            return a + b;
        }
        
        int main() {
            return 0;
        }
        "#;
        
        let mut indexer = SymbolIndexer::new().unwrap();
        let symbols = indexer.extract_symbols(code, "c", "test.c").unwrap();
        
        assert!(symbols.iter().any(|s| s.name == "add" && s.kind == SymbolKind::Function));
        assert!(symbols.iter().any(|s| s.name == "main" && s.kind == SymbolKind::Function));
    }
}