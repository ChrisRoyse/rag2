# MCP Server Setup Documentation

## Overview
This document explains how the embed-rag MCP (Model Context Protocol) server is set up to work with Claude Code on Windows WSL.

## Problem Solved
The original issue was that the Rust MCP server's debug logs were being written to stderr, which mixed with the JSON-RPC messages on stdout, causing parsing errors in Claude Code. The solution was to create wrapper scripts that properly separate stderr (logs) from stdout (JSON-RPC).

## Architecture

```
Claude Code (Windows)
        ↓
   .mcp.json config
        ↓
Node.js wrapper (mcp_wrapper.js)
        ↓
Rust MCP Server (mcp_server binary)
        ↓
   JSON-RPC on stdout
   Debug logs → logs/mcp_server.log
```

## Components

### 1. Rust MCP Server Binary
**Location**: `/home/cabdru/rag/target/debug/mcp_server`
- Built from the Rust source code
- Implements JSON-RPC protocol for MCP
- Provides search and indexing capabilities

### 2. Node.js Wrapper
**Location**: `/home/cabdru/rag/scripts/mcp_wrapper.js`
- Spawns the Rust binary as a child process
- Pipes stdin/stdout cleanly for JSON-RPC
- Redirects stderr to log file
- Handles process lifecycle and signals

### 3. Bash Wrapper (Alternative)
**Location**: `/home/cabdru/rag/scripts/mcp_wrapper.sh`
- Simple bash script alternative
- Uses exec for process replacement
- Redirects stderr to log file

### 4. Configuration
**Location**: `/home/cabdru/rag/.mcp.json`
```json
{
  "mcpServers": {
    "embed-search": {
      "command": "node",
      "args": ["/home/cabdru/rag/scripts/mcp_wrapper.js", "/home/cabdru/rag"],
      "type": "stdio"
    }
  }
}
```

### 5. Log Files
**Location**: `/home/cabdru/rag/logs/mcp_server.log`
- All debug and info logs from the server
- Timestamps and log levels preserved
- Rotated automatically

## Setup Instructions

### Prerequisites
1. Node.js installed in WSL
2. Rust toolchain installed
3. Claude Code installed on Windows

### Installation Steps

1. **Build the Rust binary**:
   ```bash
   cd /home/cabdru/rag
   cargo build
   ```

2. **Ensure wrapper scripts are executable**:
   ```bash
   chmod +x scripts/mcp_wrapper.js scripts/mcp_wrapper.sh
   ```

3. **Fix line endings if needed** (for scripts created on Windows):
   ```bash
   sed -i 's/\r$//' scripts/mcp_wrapper.sh
   ```

4. **Run validation**:
   ```bash
   bash scripts/validate_mcp.sh
   ```

5. **Restart Claude Code**:
   - Close Claude Code completely
   - Reopen Claude Code
   - The embed-search server should appear in available tools

## Validation

The validation script (`scripts/validate_mcp.sh`) checks:
- Wrapper scripts exist and are executable
- Rust binary is built
- JSON-RPC initialize method works
- Error handling returns proper error responses
- Logs are being written correctly
- Configuration is valid

Run validation with:
```bash
bash /home/cabdru/rag/scripts/validate_mcp.sh
```

Expected output:
```
=== Validation Summary ===
Tests passed: 11
Tests failed: 0
✓ All tests passed! MCP server is ready for use.
```

## Troubleshooting

### Server not appearing in Claude Code
1. Check if the binary exists: `ls -la target/debug/mcp_server`
2. Run validation script to check all components
3. Check Claude Code logs for errors
4. Ensure .mcp.json is in the project root

### Server crashes immediately
1. Check the log file: `tail -f logs/mcp_server.log`
2. Test directly: `echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | bash scripts/mcp_wrapper.sh`
3. Rebuild if needed: `cargo build`

### Permission errors
1. Make scripts executable: `chmod +x scripts/*.sh scripts/*.js`
2. Check file ownership: `ls -la scripts/`

### Line ending issues (Windows/WSL)
1. Fix with: `sed -i 's/\r$//' scripts/*.sh`
2. Or use dos2unix: `dos2unix scripts/*.sh`

## How It Works

1. **Claude Code** reads `.mcp.json` and starts the Node.js wrapper
2. **Node.js wrapper** spawns the Rust binary with proper stdio handling
3. **Rust binary** reads JSON-RPC from stdin, writes responses to stdout
4. **All logs** are redirected to `logs/mcp_server.log` to keep stdout clean
5. **Clean JSON-RPC** communication enables proper MCP protocol function

## Testing

### Manual test with initialize:
```bash
echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | \
  bash scripts/mcp_wrapper.sh /home/cabdru/rag
```

### Expected response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "server_info": {
      "name": "embed-search-mcp",
      "version": "0.1.0",
      "features": ["symbol_search", "statistical_search"]
    }
  },
  "error": null,
  "id": 1
}
```

## Key Insights

1. **MCP requires clean stdout**: Any non-JSON output corrupts the protocol
2. **Wrapper pattern**: Using a wrapper script is the standard solution
3. **Node.js preferred**: Claude Code works better with Node.js wrappers
4. **Logging separation**: All debug info must go to files, not stdout
5. **Process management**: Proper signal handling ensures clean shutdown

## Maintenance

- **Update binary**: Run `cargo build` after code changes
- **Check logs**: Monitor `logs/mcp_server.log` for issues
- **Validate regularly**: Run validation script after changes
- **Clean logs**: Periodically clean old log files

## Architecture Decision Record

### Decision: Use Node.js wrapper instead of direct binary
**Rationale**: 
- Better process management in Windows/WSL environment
- Cleaner stdio handling
- Better signal management
- More robust error handling

### Decision: Separate log file for debugging
**Rationale**:
- MCP protocol requires clean stdout
- Debugging still possible via log file
- No corruption of JSON-RPC messages
- Persistent debugging information

### Decision: Keep both Node.js and bash wrappers
**Rationale**:
- Node.js for production use with Claude Code
- Bash for quick testing and debugging
- Flexibility for different environments
- Fallback option if Node.js has issues