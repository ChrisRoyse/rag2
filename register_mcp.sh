#!/bin/bash

# Script to register the embed-search MCP server with Claude Code

echo "🚀 Registering embed-search MCP server with Claude Code"
echo "======================================================="

# Build the MCP server first
echo "Building MCP server binary..."
cargo build --bin mcp_server

if [ $? -ne 0 ]; then
    echo "❌ Failed to build MCP server"
    exit 1
fi

echo "✅ MCP server built successfully"

# Get the absolute path to the binary
MCP_BINARY="$(pwd)/target/debug/mcp_server"
PROJECT_PATH="$(pwd)"

echo ""
echo "📝 MCP Server Details:"
echo "  Binary: $MCP_BINARY"
echo "  Project: $PROJECT_PATH"

# Register with Claude Code
echo ""
echo "Registering with Claude Code..."

# Option 1: Using claude CLI if available
if command -v claude &> /dev/null; then
    echo "Using claude CLI..."
    claude mcp add embed-search "$MCP_BINARY" "$PROJECT_PATH" --scope local
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully registered with Claude Code via CLI"
    else
        echo "⚠️ Registration via CLI failed, please use manual method below"
    fi
else
    echo "Claude CLI not found, using manual registration..."
fi

# Option 2: Create .mcp.json configuration
echo ""
echo "Creating .mcp.json configuration..."

cat > .mcp.json << EOF
{
  "mcpServers": {
    "embed-search": {
      "command": "$MCP_BINARY",
      "args": ["$PROJECT_PATH"],
      "type": "stdio"
    }
  }
}
EOF

echo "✅ Created .mcp.json configuration"

# Option 3: Manual registration instructions
echo ""
echo "📋 Manual Registration Instructions:"
echo "====================================="
echo ""
echo "If automatic registration didn't work, you can manually register by:"
echo ""
echo "1. Open Claude Code"
echo "2. Run the following command in Claude Code terminal:"
echo ""
echo "   claude mcp add embed-search $MCP_BINARY $PROJECT_PATH --scope local"
echo ""
echo "Or add to your Claude Code settings:"
echo ""
echo "   {
     \"mcpServers\": {
       \"embed-search\": {
         \"command\": \"$MCP_BINARY\",
         \"args\": [\"$PROJECT_PATH\"],
         \"type\": \"stdio\"
       }
     }
   }"
echo ""

# Test the MCP server
echo "Testing MCP server..."
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"1.0.0","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}},"id":1}' | "$MCP_BINARY" "$PROJECT_PATH" 2>/dev/null | head -1

if [ $? -eq 0 ]; then
    echo "✅ MCP server responds correctly to initialization"
else
    echo "⚠️ MCP server test failed - check logs for details"
fi

echo ""
echo "✅ Registration complete! Restart Claude Code to see the MCP server."