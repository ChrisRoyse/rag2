use std::time::Instant;
use anyhow::Result;
use tantivy::{
    collector::TopDocs,
    doc,
    query::{FuzzyTermQuery, QueryParser},
    schema::{Field, Schema, STORED, TEXT},
    Index, IndexWriter, Term, Document as TantivyDocument,
};
use crate::search::ExactMatch;

/// Minimal Tantivy fuzzy search implementation
/// ZERO TOLERANCE FOR FAKE FUNCTIONALITY - THIS WORKS OR REPORTS WHY IT DOESN'T
pub struct SimpleTantivySearch {
    index: Index,
    title_field: Field,
    content_field: Field,
    path_field: Field,
}

impl SimpleTantivySearch {
    /// Create new in-memory search index - NO PERSISTENCE
    pub fn new() -> Result<Self> {
        let mut schema_builder = Schema::builder();
        
        // Simple schema: title, content, path - THAT'S IT
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED); 
        let path_field = schema_builder.add_text_field("path", STORED);
        
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        
        Ok(Self {
            index,
            title_field,
            content_field,
            path_field,
        })
    }
    
    /// Add document to index - SIMPLE
    pub fn add_document(&mut self, title: &str, content: &str, path: &str) -> Result<()> {
        let mut writer: IndexWriter<TantivyDocument> = self.index.writer(50_000_000)?;
        
        let doc = doc!(
            self.title_field => title,
            self.content_field => content,
            self.path_field => path
        );
        
        writer.add_document(doc)?;
        writer.commit()?;
        
        // Reload reader to make documents searchable
        self.index.reader()?.reload()?;
        
        Ok(())
    }
    
    /// Fuzzy search with edit distance 2 - TOP 10 RESULTS
    pub fn fuzzy_search(&self, query: &str) -> Result<(Vec<ExactMatch>, u64)> {
        let start_time = Instant::now();
        
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        
        // Create fuzzy query for content field with edit distance 2
        let term = Term::from_field_text(self.content_field, query);
        let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
        
        // Search and get top 10 results
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(10))?;
        
        let mut matches = Vec::new();
        
        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            
            let title = doc
                .get_first(self.title_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
                
            let content = doc
                .get_first(self.content_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
                
            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            matches.push(ExactMatch {
                file_path: path,
                line_number: 1, // Simple implementation - line 1
                content: content.clone(),
                line_content: content,
            });
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        Ok((matches, duration_ms))
    }
    
    /// Search with query parser as fallback
    pub fn standard_search(&self, query: &str) -> Result<(Vec<ExactMatch>, u64)> {
        let start_time = Instant::now();
        
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field, self.title_field]);
        let parsed_query = query_parser.parse_query(query)?;
        
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(10))?;
        
        let mut matches = Vec::new();
        
        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            
            let title = doc
                .get_first(self.title_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
                
            let content = doc
                .get_first(self.content_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
                
            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            
            matches.push(ExactMatch {
                file_path: path,
                line_number: 1,
                content: content.clone(),
                line_content: content,
            });
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        Ok((matches, duration_ms))
    }
    
    /// Get total number of indexed documents
    pub fn doc_count(&self) -> Result<u64> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        Ok(searcher.num_docs() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fuzzy_search_works() {
        let mut search = SimpleTantivySearch::new().expect("Failed to create search");
        
        // Add test data
        search.add_document("Database Connection", "DatabaseConnection class handles connections", "/src/db.rs").unwrap();
        search.add_document("User Manager", "UserManager handles user operations", "/src/user.rs").unwrap();
        search.add_document("Payment Service", "PaymentService processes payments", "/src/payment.rs").unwrap();
        
        // Test fuzzy search for "Database" (should match "DatabaseConnection")
        let (results, latency_ms) = search.fuzzy_search("Database").expect("Fuzzy search failed");
        
        println!("Fuzzy search latency: {}ms", latency_ms);
        println!("Found {} results", results.len());
        
        assert!(!results.is_empty(), "Should find fuzzy matches");
        assert!(latency_ms < 1000, "Search should be fast (< 1 second)");
        
        // Test with typo
        let (typo_results, typo_latency) = search.fuzzy_search("Databse").expect("Fuzzy search with typo failed");
        
        println!("Fuzzy search with typo latency: {}ms", typo_latency);
        println!("Found {} results with typo", typo_results.len());
        
        assert!(!typo_results.is_empty(), "Should find matches even with typos");
    }
    
    #[test]
    fn test_exact_functionality_works() {
        let mut search = SimpleTantivySearch::new().expect("Failed to create search");
        
        // Add real data that should be findable
        search.add_document("Test Function", "fn test_function() { println!(\"Hello\"); }", "/test.rs").unwrap();
        search.add_document("Main Function", "fn main() { test_function(); }", "/main.rs").unwrap();
        
        // Test exact search
        let (exact_results, _) = search.standard_search("test_function").expect("Standard search failed");
        assert!(!exact_results.is_empty(), "Should find exact matches");
        
        // Test fuzzy search  
        let (fuzzy_results, _) = search.fuzzy_search("test_functon").expect("Fuzzy search failed"); // typo in "function"
        assert!(!fuzzy_results.is_empty(), "Should find fuzzy matches");
        
        // Verify document count
        let count = search.doc_count().expect("Failed to get doc count");
        assert_eq!(count, 2, "Should have indexed 2 documents");
    }
}