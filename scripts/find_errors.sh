#!/bin/bash
echo "Finding compilation errors..."
cargo check 2>&1 | grep -A2 "^error\[E" | head -20