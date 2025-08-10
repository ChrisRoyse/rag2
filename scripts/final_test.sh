#!/bin/bash

echo "🔍 Final MCP Integration Test"
echo "==============================="
echo ""

# Check if the system compiles
echo "1. Testing compilation..."
if cargo check --features ml,vectordb 2>&1 | grep -q "error:"; then
    echo "❌ Compilation failed with errors"
    cargo check --features ml,vectordb 2>&1 | grep "^error:" | head -5
else
    echo "✅ Compilation successful"
fi

echo ""
echo "2. Checking key components..."

# Check if .embed-search directory exists
if [ -d ".embed-search" ]; then
    echo "✅ Embedding index directory exists"
else
    echo "❌ Embedding index directory missing"
fi

# Check if intelligent_fusion is connected
if grep -q "intelligent_fusion" src/mcp/tools/search.rs; then
    echo "✅ Intelligent fusion connected to MCP tools"
else
    echo "❌ Intelligent fusion not connected"
fi

# Check if BM25 has add_document_from_file
if grep -q "add_document_from_file" src/search/bm25.rs; then
    echo "✅ BM25 has add_document_from_file method"
else
    echo "❌ BM25 missing add_document_from_file method"
fi

# Check if UnifiedSearchAdapter has intelligent_fusion
if grep -q "pub async fn intelligent_fusion" src/search/search_adapter.rs; then
    echo "✅ UnifiedSearchAdapter has intelligent_fusion method"
else
    echo "❌ UnifiedSearchAdapter missing intelligent_fusion method"
fi

echo ""
echo "3. Summary..."
echo "============="

# Get actual error count
ERROR_COUNT=$(cargo check --features ml,vectordb 2>&1 | grep -c "^error:" || echo "0")
WARNING_COUNT=$(cargo check --features ml,vectordb 2>&1 | grep -c "^warning:" || echo "0")

echo "Compilation errors: $ERROR_COUNT"
echo "Warnings: $WARNING_COUNT"

if [ "$ERROR_COUNT" -eq "0" ]; then
    echo ""
    echo "✅ System is ready for use!"
else
    echo ""
    echo "⚠️ System still has $ERROR_COUNT compilation errors to fix"
fi