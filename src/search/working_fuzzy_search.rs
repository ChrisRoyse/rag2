use std::time::Instant;
use std::collections::HashMap;
use anyhow::Result;
use crate::search::ExactMatch;

/// MINIMAL WORKING FUZZY SEARCH - ZERO TOLERANCE FOR FAKE FUNCTIONALITY
/// Uses Levenshtein distance for fuzzy matching - ACTUALLY WORKS
pub struct WorkingFuzzySearch {
    documents: Vec<Document>,
}

#[derive(Debug, Clone)]
struct Document {
    title: String,
    content: String,
    path: String,
}

impl WorkingFuzzySearch {
    /// Create new in-memory search - SIMPLE AND WORKING
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }
    
    /// Add document to search index
    pub fn add_document(&mut self, title: &str, content: &str, path: &str) {
        self.documents.push(Document {
            title: title.to_string(),
            content: content.to_string(),
            path: path.to_string(),
        });
    }
    
    /// Compute Levenshtein distance between two strings
    fn levenshtein_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let a_len = a_chars.len();
        let b_len = b_chars.len();
        
        if a_len == 0 { return b_len; }
        if b_len == 0 { return a_len; }
        
        let mut dp = vec![vec![0; b_len + 1]; a_len + 1];
        
        // Initialize first row and column
        for i in 0..=a_len { dp[i][0] = i; }
        for j in 0..=b_len { dp[0][j] = j; }
        
        // Fill the matrix
        for i in 1..=a_len {
            for j in 1..=b_len {
                let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
                dp[i][j] = std::cmp::min(
                    std::cmp::min(
                        dp[i-1][j] + 1,     // deletion
                        dp[i][j-1] + 1      // insertion
                    ),
                    dp[i-1][j-1] + cost     // substitution
                );
            }
        }
        
        dp[a_len][b_len]
    }
    
    /// Check if query matches text within edit distance of 2
    fn is_fuzzy_match(&self, query: &str, text: &str, max_distance: usize) -> bool {
        let query_lower = query.to_lowercase();
        let text_lower = text.to_lowercase();
        
        // Exact match
        if text_lower.contains(&query_lower) {
            return true;
        }
        
        // Split text into words and check each word
        for word in text_lower.split_whitespace() {
            if Self::levenshtein_distance(&query_lower, word) <= max_distance {
                return true;
            }
            
            // Check if query is a substring of word (for compound words)
            if word.contains(&query_lower) {
                return true;
            }
            
            // Check word prefixes and suffixes
            if query_lower.len() >= 3 {
                for start in 0..word.len().saturating_sub(query_lower.len() - 1) {
                    let end = std::cmp::min(start + query_lower.len(), word.len());
                    let substring = &word[start..end];
                    if Self::levenshtein_distance(&query_lower, substring) <= max_distance {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Fuzzy search with edit distance 2 - TOP 10 RESULTS
    pub fn fuzzy_search(&self, query: &str) -> Result<(Vec<ExactMatch>, u64)> {
        let start_time = Instant::now();
        
        if query.is_empty() {
            return Ok((Vec::new(), 0));
        }
        
        let mut matches = Vec::new();
        let mut scores = Vec::new();
        
        for doc in &self.documents {
            let mut score = 0;
            let mut found = false;
            
            // Check title with higher weight
            if self.is_fuzzy_match(query, &doc.title, 2) {
                score += 10;
                found = true;
            }
            
            // Check content
            if self.is_fuzzy_match(query, &doc.content, 2) {
                score += 5;
                found = true;
            }
            
            // Exact matches get higher scores
            let query_lower = query.to_lowercase();
            if doc.title.to_lowercase().contains(&query_lower) {
                score += 20;
            }
            if doc.content.to_lowercase().contains(&query_lower) {
                score += 15;
            }
            
            if found {
                matches.push(ExactMatch {
                    file_path: doc.path.clone(),
                    line_number: 1,
                    content: doc.content.clone(),
                    line_content: doc.content.clone(),
                });
                scores.push(score);
            }
        }
        
        // Sort by score (highest first) and take top 10
        let mut indexed_matches: Vec<(usize, i32)> = scores.into_iter().enumerate().collect();
        indexed_matches.sort_by(|a, b| b.1.cmp(&a.1));
        
        let sorted_matches = indexed_matches
            .into_iter()
            .take(10)
            .map(|(idx, _)| matches[idx].clone())
            .collect();
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        Ok((sorted_matches, duration_ms))
    }
    
    /// Standard exact search
    pub fn exact_search(&self, query: &str) -> Result<(Vec<ExactMatch>, u64)> {
        let start_time = Instant::now();
        
        if query.is_empty() {
            return Ok((Vec::new(), 0));
        }
        
        let mut matches = Vec::new();
        let query_lower = query.to_lowercase();
        
        for doc in &self.documents {
            if doc.title.to_lowercase().contains(&query_lower) || 
               doc.content.to_lowercase().contains(&query_lower) {
                matches.push(ExactMatch {
                    file_path: doc.path.clone(),
                    line_number: 1,
                    content: doc.content.clone(),
                    line_content: doc.content.clone(),
                });
            }
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        Ok((matches.into_iter().take(10).collect(), duration_ms))
    }
    
    /// Get total number of indexed documents
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("", ""), 0);
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("hello", "hello"), 0);
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("database", "databse"), 1);
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("function", "functon"), 1);
        assert_eq!(WorkingFuzzySearch::levenshtein_distance("hello", "world"), 4);
    }
    
    #[test] 
    fn test_fuzzy_search_actually_works() {
        let mut search = WorkingFuzzySearch::new();
        
        // Add realistic test data
        search.add_document("Database Connection", "DatabaseConnection class handles database connections to PostgreSQL", "/src/db/connection.rs");
        search.add_document("User Manager", "UserManager struct manages user authentication and sessions", "/src/auth/user.rs");  
        search.add_document("Payment Service", "PaymentService processes credit card payments via Stripe API", "/src/payment/service.rs");
        search.add_document("Config Parser", "ConfigParser loads application configuration from TOML files", "/src/config/parser.rs");
        
        // Test 1: Exact match
        let (exact_results, exact_latency) = search.fuzzy_search("Database").expect("Fuzzy search failed");
        println!("Exact search for 'Database': Found {} results in {}ms", exact_results.len(), exact_latency);
        assert!(!exact_results.is_empty(), "Should find exact matches for 'Database'");
        assert!(exact_latency < 100, "Search should be very fast (< 100ms)");
        
        // Test 2: Typo with edit distance 1
        let (typo_results, typo_latency) = search.fuzzy_search("Databse").expect("Fuzzy search with typo failed");
        println!("Fuzzy search for 'Databse' (typo): Found {} results in {}ms", typo_results.len(), typo_latency);
        assert!(!typo_results.is_empty(), "Should find matches even with 1-char typo");
        
        // Test 3: Typo with edit distance 2  
        let (typo2_results, typo2_latency) = search.fuzzy_search("Dataase").expect("Fuzzy search with 2-char typo failed");
        println!("Fuzzy search for 'Dataase' (2-char typo): Found {} results in {}ms", typo2_results.len(), typo2_latency);
        assert!(!typo2_results.is_empty(), "Should find matches with 2-char typo");
        
        // Test 4: Partial word match
        let (partial_results, partial_latency) = search.fuzzy_search("Config").expect("Partial search failed");
        println!("Search for 'Config': Found {} results in {}ms", partial_results.len(), partial_latency);  
        assert!(!partial_results.is_empty(), "Should find partial word matches");
        
        // Test 5: Case insensitive
        let (case_results, case_latency) = search.fuzzy_search("payment").expect("Case insensitive search failed");
        println!("Search for 'payment' (lowercase): Found {} results in {}ms", case_results.len(), case_latency);
        assert!(!case_results.is_empty(), "Should be case insensitive");
        
        // Verify document count
        assert_eq!(search.doc_count(), 4, "Should have indexed 4 documents");
        
        println!("✅ ALL FUZZY SEARCH TESTS PASSED - FUNCTIONALITY WORKS!");
    }
    
    #[test]
    fn test_performance_with_large_dataset() {
        let mut search = WorkingFuzzySearch::new();
        
        // Add 1000 documents
        for i in 0..1000 {
            search.add_document(
                &format!("Document Title {}", i),
                &format!("This is document {} with some content about various topics like database, user management, configuration, and payment processing", i),
                &format!("/docs/doc_{}.txt", i)
            );
        }
        
        // Search should still be fast
        let (results, latency) = search.fuzzy_search("database").expect("Performance test failed");
        println!("Performance test: Found {} results in {}ms with 1000 docs", results.len(), latency);
        
        assert!(latency < 1000, "Search should be fast even with 1000 documents (< 1000ms)");
        assert!(!results.is_empty(), "Should find matches in large dataset");
        
        println!("✅ PERFORMANCE TEST PASSED - Search latency: {}ms", latency);
    }
}