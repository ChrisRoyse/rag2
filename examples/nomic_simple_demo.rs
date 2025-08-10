/// Simple demonstration of Nomic + Vector Storage integration
/// 
/// This example shows how to use the combined embedder and storage system
/// for semantic search over code snippets.
/// 
/// Requirements:
/// - Model file: /home/cabdru/rag/model/nomic-embed-code.Q4_K_M.gguf (4.38GB)
/// - Features: `cargo run --example nomic_simple_demo --features "vectordb,ml"`

use anyhow::Result;
use embed_search::storage::nomic_integration::NomicVectorStore;
use embed_search::chunking::Chunk;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Nomic + Simple Storage Integration Demo");
    
    // Initialize the integrated store
    let store = match NomicVectorStore::new().await {
        Ok(store) => {
            println!("✅ Successfully initialized Nomic vector store");
            println!("   Embedding dimensions: {}", store.dimensions());
            store
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize store: {}", e);
            eprintln!("   Make sure the model file exists at:");
            eprintln!("   /home/cabdru/rag/model/nomic-embed-code.Q4_K_M.gguf");
            return Err(e.into());
        }
    };
    
    // Sample code snippets to embed and store
    let code_samples = vec![
        ("hello.rs", "fn main() { println!(\"Hello, world!\"); }"),
        ("math.rs", "fn add(a: i32, b: i32) -> i32 { a + b }"),
        ("sort.rs", "fn bubble_sort(arr: &mut [i32]) { /* sorting logic */ }"),
        ("hash.rs", "use std::collections::HashMap; fn new_map() -> HashMap<String, i32> { HashMap::new() }"),
        ("async.rs", "async fn fetch_data() -> Result<String, Error> { todo!() }"),
    ];
    
    println!("\n📝 Storing {} code samples...", code_samples.len());
    
    // Store each sample
    for (i, (filename, code)) in code_samples.iter().enumerate() {
        let chunk = Chunk {
            content: code.to_string(),
            start_line: 1,
            end_line: 1,
        };
        
        match store.embed_and_store(filename, i, &chunk).await {
            Ok(_) => println!("   ✅ Stored {}: {}", filename, &code[..30.min(code.len())]),
            Err(e) => {
                eprintln!("   ❌ Failed to store {}: {}", filename, e);
                continue;
            }
        }
    }
    
    // Test searches
    let queries = vec![
        "hello world program",
        "mathematical addition",
        "sorting algorithm",
        "asynchronous function",
        "hash map creation",
    ];
    
    println!("\n🔍 Testing semantic search...");
    
    for query in queries {
        println!("\n   Query: \"{}\"", query);
        
        match store.search(query, 3).await {
            Ok(results) => {
                if results.is_empty() {
                    println!("     No results found");
                } else {
                    for (i, result) in results.iter().enumerate() {
                        println!("     {}. {}: {}", 
                                i + 1, 
                                result.file_path, 
                                result.content);
                    }
                }
            }
            Err(e) => {
                eprintln!("     ❌ Search failed: {}", e);
            }
        }
    }
    
    // Show storage statistics
    match store.storage().count().await {
        Ok(count) => println!("\n📊 Total stored embeddings: {}", count),
        Err(e) => eprintln!("Failed to get count: {}", e),
    }
    
    println!("\n🎉 Demo completed successfully!");
    Ok(())
}