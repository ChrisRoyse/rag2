#!/bin/bash

echo "Testing MCP wrapper scripts..."

# Test bash wrapper exists and is executable
echo "1. Testing bash wrapper..."
if [[ -x "/home/cabdru/rag/scripts/mcp_wrapper.sh" ]]; then
    echo "   ✓ Bash wrapper is executable"
else
    echo "   ✗ Bash wrapper is not executable"
    exit 1
fi

# Test Node.js wrapper exists and is executable  
echo "2. Testing Node.js wrapper..."
if [[ -x "/home/cabdru/rag/scripts/mcp_wrapper.js" ]]; then
    echo "   ✓ Node.js wrapper is executable"
else
    echo "   ✗ Node.js wrapper is not executable"
    exit 1
fi

# Test that logs directory exists
echo "3. Testing logs directory..."
if [[ -d "/home/cabdru/rag/logs" ]]; then
    echo "   ✓ Logs directory exists"
else
    echo "   ✗ Logs directory missing"
    exit 1
fi

# Test that Rust binary exists
echo "4. Testing Rust binary..."
if [[ -x "/home/cabdru/rag/target/debug/mcp_server" ]]; then
    echo "   ✓ Rust MCP server binary exists and is executable"
else
    echo "   ✗ Rust MCP server binary missing or not executable"
    exit 1
fi

# Test MCP configuration
echo "5. Testing MCP configuration..."
if grep -q "mcp_wrapper.js" "/home/cabdru/rag/.mcp.json"; then
    echo "   ✓ .mcp.json updated to use Node.js wrapper"
else
    echo "   ✗ .mcp.json not properly configured"
    exit 1
fi

echo ""
echo "✅ All wrapper tests passed!"
echo ""
echo "Usage:"
echo "  Bash wrapper:   /home/cabdru/rag/scripts/mcp_wrapper.sh [project_path]"
echo "  Node.js wrapper: node /home/cabdru/rag/scripts/mcp_wrapper.js [project_path]"
echo ""
echo "Configuration:"
echo "  MCP config uses Node.js wrapper by default"
echo "  Stderr logs go to: /home/cabdru/rag/logs/mcp_server.log"
echo ""
echo "To test manually:"
echo "  echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}' | node /home/cabdru/rag/scripts/mcp_wrapper.js"