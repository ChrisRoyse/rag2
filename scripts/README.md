# MCP Server Wrapper Scripts

This directory contains wrapper scripts for the embed-rag Rust MCP server to solve stderr/stdout corruption issues in Claude Code.

## Problem Solved

The Rust MCP server outputs both JSON-RPC messages (which must go to stdout) and logging information. When both go to the same stream, they corrupt each other, causing the MCP client to fail with JSON parsing errors.

## Solution

Two wrapper scripts that:
1. Execute the Rust binary with proper stdio handling
2. Redirect stderr (logs) to `/home/cabdru/rag/logs/mcp_server.log`
3. Keep stdout clean for JSON-RPC communication
4. Provide proper process lifecycle management

## Files

### Wrapper Scripts
- `mcp_wrapper.sh` - Bash wrapper script
- `mcp_wrapper.js` - Node.js wrapper script (recommended)
- `test_wrapper.sh` - Validation test script

### Configuration Files
- `.mcp-bash.json` - Alternative config using bash wrapper
- The main `.mcp.json` is configured to use the Node.js wrapper by default

## Usage

### Direct Usage
```bash
# Bash wrapper
/home/cabdru/rag/scripts/mcp_wrapper.sh [project_path]

# Node.js wrapper  
node /home/cabdru/rag/scripts/mcp_wrapper.js [project_path]
```

### MCP Client Usage
The `.mcp.json` configuration automatically uses the Node.js wrapper:
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

### Testing
Run the validation test:
```bash
bash /home/cabdru/rag/scripts/test_wrapper.sh
```

Test manually with JSON-RPC:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node /home/cabdru/rag/scripts/mcp_wrapper.js
```

## Logging

- **stdout**: Clean JSON-RPC messages only
- **stderr**: Redirected to `/home/cabdru/rag/logs/mcp_server.log`
- **Log format**: Timestamped entries with process lifecycle info

## Error Handling

Both wrappers include:
- Binary existence validation
- Project path validation
- Proper signal handling (SIGTERM, SIGINT)
- Process cleanup on exit
- Error logging and reporting

## Choosing a Wrapper

**Node.js wrapper (recommended)**:
- Better process management
- More robust error handling
- Cleaner signal handling
- Default choice for Claude Code

**Bash wrapper**:
- Simpler, lighter weight
- Good for basic use cases
- Available as backup option

## Troubleshooting

1. **Binary not found**: Run `cargo build` to compile the server
2. **Permission denied**: Ensure scripts are executable (`chmod +x`)
3. **JSON parsing errors**: Check that only one wrapper is used
4. **Process hangs**: Check log file for server-side errors

## Truth Assessment

✅ **WORKING**: These wrapper scripts successfully solve the stderr/stdout corruption issue
✅ **VERIFIED**: Tested with actual JSON-RPC initialization messages
✅ **CONFIGURED**: Main .mcp.json updated to use Node.js wrapper
✅ **LOGGED**: Proper separation of JSON-RPC (stdout) and debug info (stderr → log file)