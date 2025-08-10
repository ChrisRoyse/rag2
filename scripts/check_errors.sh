#!/bin/bash
# Script to check compilation errors
cargo check --features ml,vectordb 2>&1 | grep -E "^error" | head -10