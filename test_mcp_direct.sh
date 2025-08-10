#!/bin/bash

echo "Testing MCP Server Direct Execution"
echo "===================================="

# Create a test input file
cat > test_input.json << 'EOF'
{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}},"id":1}
EOF

echo "Starting MCP server..."
echo ""

# Run the server with test input
timeout 1 ./target/debug/mcp_server /home/cabdru/rag < test_input.json 2> mcp_error.log

echo ""
echo "Checking for errors..."
if [ -f mcp_error.log ]; then
    echo "Error log content:"
    cat mcp_error.log
fi

# Clean up
rm -f test_input.json mcp_error.log