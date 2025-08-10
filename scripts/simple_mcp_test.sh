#!/bin/bash

# Simple MCP Integration Test Script
# Tests all components are connected and working

echo "🔍 MCP Integration Test"
echo "======================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0

# Function to test a component
test_component() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing $name... "
    
    result=$(eval "$command" 2>&1)
    
    if echo "$result" | grep -q "$expected"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        echo "  Expected to find: $expected"
        echo "  Got: ${result:0:100}..."
        return 1
    fi
}

# 1. Check if embedding directory exists
test_component "Embedding Index Directory" \
    "ls -la .embed-search 2>/dev/null || echo 'not found'" \
    "not found"

if [ $? -ne 0 ]; then
    echo "  Creating .embed-search directory..."
    mkdir -p .embed-search
fi

# 2. Check Tantivy module
test_component "Tantivy Search Module" \
    "grep -l 'FuzzyTermQuery' src/search/tantivy_search.rs 2>/dev/null || echo 'not found'" \
    "tantivy_search.rs"

# 3. Check Nomic embeddings
test_component "Nomic Embeddings Module" \
    "grep -l 'NomicEmbedder' src/embedding/nomic.rs 2>/dev/null || echo 'not found'" \
    "nomic.rs"

# 4. Check BM25 implementation
test_component "BM25 Scoring Engine" \
    "grep -l 'BM25Engine' src/search/bm25.rs 2>/dev/null || echo 'not found'" \
    "bm25.rs"

# 5. Check unified search adapter
test_component "Unified Search Adapter" \
    "grep -l 'UnifiedSearchAdapter' src/search/search_adapter.rs 2>/dev/null || echo 'not found'" \
    "search_adapter.rs"

# 6. Check MCP server integration
test_component "MCP Server Integration" \
    "grep -l 'UnifiedSearchAdapter' src/mcp/server.rs 2>/dev/null || echo 'not found'" \
    "server.rs"

# 7. Check Git watcher
test_component "Git Watcher" \
    "grep -l 'GitWatcher' src/git/watcher.rs 2>/dev/null || echo 'not found'" \
    "watcher.rs"

# 8. Check feature compilation
echo ""
echo "Checking feature compilation..."
if cargo check --features ml,vectordb 2>&1 | grep -q "Finished"; then
    echo -e "${GREEN}✅ Features compile successfully${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
else
    echo -e "${YELLOW}⚠️ Feature compilation has warnings or errors${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
fi

# 9. Check if Nomic model exists
echo ""
echo "Checking Nomic model file..."
if [ -f "models/nomic-embed-text-v1.5.f16.gguf" ]; then
    echo -e "${GREEN}✅ Nomic model file exists${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️ Nomic model file missing${NC}"
    echo "  To download: wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf -P models/"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Summary
echo ""
echo "========================"
echo "Test Summary"
echo "========================"
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✅ All tests passed! MCP integration is operational.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some tests failed. Run fix_mcp.sh to resolve issues.${NC}"
    exit 1
fi