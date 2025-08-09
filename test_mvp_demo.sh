#!/bin/bash

echo "🚀 Testing Minimal MVP Demo"
echo "=============================="

# Test basic compilation and test run first
echo "Running unit tests..."
cargo test --lib minimal_mvp --quiet

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    
    echo ""
    echo "Running automated demo test..."
    
    # Create a test input file
    echo -e "test\nquit" > /tmp/mvp_test_input.txt
    
    # Run demo with input from file
    timeout 30s cargo run --bin minimal_mvp_demo < /tmp/mvp_test_input.txt
    
    # Clean up
    rm -f /tmp/mvp_test_input.txt
    
    echo ""
    echo "✅ MVP Demo Test Complete!"
else
    echo "❌ Tests failed!"
    exit 1
fi