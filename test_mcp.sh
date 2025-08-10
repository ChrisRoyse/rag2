#!/bin/bash

echo "Testing MCP Server"
echo "=================="

# Build first
echo "Building MCP server..."
cargo build --bin mcp_server

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "✅ Build successful"

# Test initialization
echo ""
echo "Testing JSON-RPC initialization..."
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}},"id":1}' | timeout 2 ./target/debug/mcp_server /home/cabdru/rag 2>&1 | head -5

echo ""
echo "MCP server test complete"