#!/bin/bash
#
# MCP Server Wrapper for embed-rag Rust MCP server
# 
# This script wraps the Rust MCP server binary to:
# 1. Redirect stderr to prevent JSON-RPC corruption
# 2. Ensure clean stdout for MCP communication
# 3. Handle error logging properly
#
# Usage: ./mcp_wrapper.sh [project_path]
#

# Default project path
PROJECT_PATH="${1:-/home/cabdru/rag}"

# Path to the Rust binary
RUST_BINARY="/home/cabdru/rag/target/debug/mcp_server"

# Log file for debugging (stderr output)
LOG_FILE="/home/cabdru/rag/logs/mcp_server.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Check if binary exists
if [[ ! -x "$RUST_BINARY" ]]; then
    echo "Error: MCP server binary not found or not executable: $RUST_BINARY" >&2
    echo "Run 'cargo build' to build the server" >&2
    exit 1
fi

# Check if project path exists
if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "Error: Project path does not exist: $PROJECT_PATH" >&2
    exit 1
fi

# Execute the Rust binary with stderr redirected to log file
# This ensures only JSON-RPC messages go to stdout
exec "$RUST_BINARY" "$PROJECT_PATH" 2>>"$LOG_FILE"