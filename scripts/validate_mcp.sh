#!/bin/bash
#
# MCP Server Validation Script
# Tests that the MCP server wrapper is working correctly
#

# set -e  # Disabled to allow test failures

echo "=== MCP Server Validation Script ==="
echo "Testing embed-rag MCP server wrapper functionality"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"
    
    echo -n "Testing $test_name... "
    
    if result=$(eval "$test_command" 2>/dev/null); then
        if echo "$result" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✓${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗${NC} (Pattern not found)"
            echo "  Expected pattern: $expected_pattern"
            echo "  Got: ${result:0:100}..."
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗${NC} (Command failed)"
        ((TESTS_FAILED++))
    fi
}

# Test 1: Check if wrappers exist
echo "1. Checking wrapper scripts..."
if [[ -f "/home/cabdru/rag/scripts/mcp_wrapper.sh" ]]; then
    echo -e "  Bash wrapper: ${GREEN}✓${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  Bash wrapper: ${RED}✗${NC}"
    ((TESTS_FAILED++))
fi

if [[ -f "/home/cabdru/rag/scripts/mcp_wrapper.js" ]]; then
    echo -e "  Node.js wrapper: ${GREEN}✓${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  Node.js wrapper: ${RED}✗${NC}"
    ((TESTS_FAILED++))
fi

# Test 2: Check if binary exists
echo ""
echo "2. Checking MCP server binary..."
if [[ -f "/home/cabdru/rag/target/debug/mcp_server" ]]; then
    echo -e "  Binary exists: ${GREEN}✓${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  Binary exists: ${RED}✗${NC}"
    echo "  Run 'cargo build' to build the server"
    ((TESTS_FAILED++))
fi

# Test 3: Test initialize method
echo ""
echo "3. Testing JSON-RPC initialize method..."
run_test "bash wrapper initialize" \
    "echo '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{},\"id\":1}' | bash /home/cabdru/rag/scripts/mcp_wrapper.sh" \
    '"jsonrpc":"2.0".*"result".*"server_info"'

run_test "node wrapper initialize" \
    "echo '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{},\"id\":1}' | timeout 1 node /home/cabdru/rag/scripts/mcp_wrapper.js 2>/dev/null || true" \
    '"jsonrpc":"2.0".*"result".*"server_info"'

# Test 4: Test unknown method (should return error)
echo ""
echo "4. Testing error handling..."
run_test "unknown method error" \
    "echo '{\"jsonrpc\":\"2.0\",\"method\":\"unknown_method\",\"params\":{},\"id\":2}' | bash /home/cabdru/rag/scripts/mcp_wrapper.sh" \
    '"error"'

# Test 5: Check log file
echo ""
echo "5. Checking log file..."
if [[ -f "/home/cabdru/rag/logs/mcp_server.log" ]]; then
    echo -e "  Log file exists: ${GREEN}✓${NC}"
    ((TESTS_PASSED++))
    
    # Check if logs are being written
    if grep -q "MCP server ready" /home/cabdru/rag/logs/mcp_server.log 2>/dev/null; then
        echo -e "  Logs being written: ${GREEN}✓${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  Logs being written: ${YELLOW}⚠${NC}"
    fi
else
    echo -e "  Log file exists: ${RED}✗${NC}"
    ((TESTS_FAILED++))
fi

# Test 6: Check .mcp.json configuration
echo ""
echo "6. Checking .mcp.json configuration..."
if [[ -f "/home/cabdru/rag/.mcp.json" ]]; then
    echo -e "  Configuration exists: ${GREEN}✓${NC}"
    ((TESTS_PASSED++))
    
    # Check if embed-search server is configured
    if grep -q '"embed-search"' /home/cabdru/rag/.mcp.json; then
        echo -e "  embed-search configured: ${GREEN}✓${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  embed-search configured: ${RED}✗${NC}"
        ((TESTS_FAILED++))
    fi
    
    # Check if using wrapper
    if grep -q 'mcp_wrapper' /home/cabdru/rag/.mcp.json; then
        echo -e "  Using wrapper script: ${GREEN}✓${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "  Using wrapper script: ${RED}✗${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo -e "  Configuration exists: ${RED}✗${NC}"
    ((TESTS_FAILED++))
fi

# Summary
echo ""
echo "=== Validation Summary ==="
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "\n${GREEN}✓ All tests passed! MCP server is ready for use.${NC}"
    echo ""
    echo "To use with Claude Code:"
    echo "1. Restart Claude Code"
    echo "2. The embed-search MCP server should appear in the available tools"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed. Please fix the issues above.${NC}"
    exit 1
fi