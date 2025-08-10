#!/usr/bin/env node
/**
 * Node.js MCP Server Wrapper for embed-rag Rust MCP server
 * 
 * This wrapper:
 * 1. Spawns the Rust MCP server as a child process
 * 2. Pipes stdin/stdout cleanly for JSON-RPC
 * 3. Redirects stderr to a log file to prevent corruption
 * 4. Provides proper error handling and cleanup
 * 
 * Usage: node mcp_wrapper.js [project_path]
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const PROJECT_PATH = process.argv[2] || '/home/cabdru/rag';
const RUST_BINARY = '/home/cabdru/rag/target/debug/mcp_server';
const LOG_FILE = '/home/cabdru/rag/logs/mcp_server.log';

// Ensure log directory exists
const logDir = path.dirname(LOG_FILE);
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
}

// Validate binary exists
if (!fs.existsSync(RUST_BINARY)) {
    console.error(`Error: MCP server binary not found: ${RUST_BINARY}`);
    console.error('Run "cargo build" to build the server');
    process.exit(1);
}

// Validate project path exists
if (!fs.existsSync(PROJECT_PATH)) {
    console.error(`Error: Project path does not exist: ${PROJECT_PATH}`);
    process.exit(1);
}

// Create log stream
const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' });

// Log startup
logStream.write(`\n[${new Date().toISOString()}] Starting MCP server wrapper\n`);
logStream.write(`Binary: ${RUST_BINARY}\n`);
logStream.write(`Project: ${PROJECT_PATH}\n`);

// Spawn the Rust binary
const child = spawn(RUST_BINARY, [PROJECT_PATH], {
    stdio: ['pipe', 'pipe', 'pipe'] // stdin, stdout, stderr
});

// Handle process errors
child.on('error', (err) => {
    logStream.write(`[${new Date().toISOString()}] Process error: ${err.message}\n`);
    process.exit(1);
});

// Pipe stdin to child
process.stdin.pipe(child.stdin);

// Pipe child stdout to process stdout (JSON-RPC messages)
child.stdout.pipe(process.stdout);

// Redirect stderr to log file
child.stderr.pipe(logStream, { end: false });

// Handle child process exit
child.on('exit', (code, signal) => {
    logStream.write(`[${new Date().toISOString()}] Process exited with code ${code}, signal ${signal}\n`);
    logStream.end();
    process.exit(code || 0);
});

// Handle process signals
process.on('SIGTERM', () => {
    logStream.write(`[${new Date().toISOString()}] Received SIGTERM, shutting down\n`);
    child.kill('SIGTERM');
});

process.on('SIGINT', () => {
    logStream.write(`[${new Date().toISOString()}] Received SIGINT, shutting down\n`);
    child.kill('SIGINT');
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    logStream.write(`[${new Date().toISOString()}] Uncaught exception: ${err.message}\n${err.stack}\n`);
    child.kill('SIGTERM');
    process.exit(1);
});