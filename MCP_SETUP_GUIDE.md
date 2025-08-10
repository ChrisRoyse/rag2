# MCP Server Setup Guide for Claude Code

## ✅ Setup Complete

The embed-search MCP server is now configured and ready to be used with Claude Code.

## Configuration Details

### MCP Server Location
- **Binary**: `/home/cabdru/rag/target/debug/mcp_server`
- **Project**: `/home/cabdru/rag`
- **Config**: `/home/cabdru/rag/embed-search.toml`

### Registration in Claude Code

The MCP server has been added to `.mcp.json` and will be automatically discovered by Claude Code.

```json
{
  "mcpServers": {
    "embed-search": {
      "command": "/home/cabdru/rag/target/debug/mcp_server",
      "args": ["/home/cabdru/rag"],
      "type": "stdio"
    }
  }
}
```

## Available Tools

Once connected, the embed-search MCP server provides these tools:

### 1. **search**
Search for code using BM25 + intelligent fusion
```json
{
  "tool": "search",
  "query": "your search term",
  "max_results": 50
}
```

### 2. **index_directory**
Index a directory for searching
```json
{
  "tool": "index_directory",
  "directory_path": "/path/to/code"
}
```

### 3. **get_status**
Get server status and statistics
```json
{
  "tool": "get_status"
}
```

### 4. **clear_index**
Clear the search index
```json
{
  "tool": "clear_index"
}
```

## How to Use in Claude Code

1. **Restart Claude Code** to pick up the new MCP server configuration

2. **Verify Connection**
   - Open Claude Code
   - Check that "embed-search" appears in the MCP servers list
   - The server should show as "Connected"

3. **Use the Tools**
   - You can now use commands like:
   - "Search for functions that handle authentication"
   - "Index the src directory"
   - "Show search statistics"

## Features

- **BM25 Text Search**: Statistical ranking with IDF scoring
- **Intelligent Fusion**: Reciprocal Rank Fusion for optimal results
- **Fuzzy Matching**: Tantivy-based fuzzy search support
- **Symbol Search**: AST-based code symbol indexing
- **Git Integration**: Automatic reindexing on file changes
- **Performance**: Parallel search across multiple backends

## Troubleshooting

If the server doesn't appear in Claude Code:

1. **Check the binary exists**:
   ```bash
   ls -la /home/cabdru/rag/target/debug/mcp_server
   ```

2. **Test the server manually**:
   ```bash
   echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | ./target/debug/mcp_server /home/cabdru/rag
   ```

3. **Rebuild if needed**:
   ```bash
   cargo build --bin mcp_server
   ```

4. **Check logs**:
   The server logs to stderr, so you can see any errors in Claude Code's MCP server logs.

## Architecture

```
Claude Code <-> MCP Protocol <-> embed-search server
                     ↓
              JSON-RPC over stdio
                     ↓
         ┌──────────────────────────┐
         │   Unified Search Engine   │
         ├──────────────────────────┤
         │ • BM25 Statistical Search │
         │ • Tantivy Fuzzy Search   │
         │ • Symbol/AST Search      │
         │ • Intelligent Fusion     │
         └──────────────────────────┘
```

## Summary

The embed-search MCP server is now fully integrated with Claude Code, providing powerful code search capabilities through the Model Context Protocol. The server uses intelligent fusion to combine results from multiple search backends, delivering optimal search results for your codebase.