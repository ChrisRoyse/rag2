#!/bin/bash

# Quick validation script to check test structure and dependencies

set -e

echo "🔍 Validating Embedding Migration Test Structure"
echo "==============================================="
echo ""

# Check test files exist
echo "📁 Checking test files exist..."

test_files=(
    "tests/embedding_migration/unit_tests.rs"
    "tests/embedding_migration/integration_tests.rs" 
    "tests/benchmarks/embedding_performance_bench.rs"
    "tests/validation/rag_system_validation.rs"
    "tests/validation/regression_tests.rs"
    "tests/load_testing/embedding_stress_tests.rs"
    "tests/validation/llama_cpp_integration_tests.rs"
    "tests/validation/cache_validation_tests.rs"
    "tests/validation/test_runner.rs"
)

missing_files=0
for file in "${test_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
        missing_files=$((missing_files + 1))
    fi
done

echo ""

# Check model file
echo "🤖 Checking model file..."
if [[ -f "./model/nomic-embed-code.Q4_K_M.gguf" ]]; then
    model_size=$(du -h "./model/nomic-embed-code.Q4_K_M.gguf" | cut -f1)
    echo "✅ Model file found (Size: $model_size)"
else
    echo "❌ Model file missing: ./model/nomic-embed-code.Q4_K_M.gguf"
    missing_files=$((missing_files + 1))
fi

echo ""

# Check Cargo.toml test configurations
echo "📦 Checking Cargo.toml test configurations..."

if grep -q "embedding_migration" Cargo.toml; then
    echo "❌ Individual test configs not found in Cargo.toml"
    echo "   Add test configurations to Cargo.toml for proper test discovery"
else
    echo "ℹ️  Test files will be run using --test parameter"
fi

echo ""

# Check feature flags
echo "🏗️  Checking feature flags..."

required_features=("ml" "vectordb" "full-system")
for feature in "${required_features[@]}"; do
    if grep -q "\"$feature\"" Cargo.toml; then
        echo "✅ Feature '$feature' found"
    else
        echo "❌ Feature '$feature' missing"
        missing_files=$((missing_files + 1))
    fi
done

echo ""

# Check dependencies
echo "📚 Checking test dependencies..."

test_deps=("tokio" "tempfile" "anyhow")
for dep in "${test_deps[@]}"; do
    if grep -q "$dep" Cargo.toml; then
        echo "✅ Dependency '$dep' found"
    else
        echo "⚠️  Dependency '$dep' may be missing"
    fi
done

echo ""

# Syntax check (if rust is available)
echo "🔍 Performing basic syntax checks..."

if command -v cargo &> /dev/null; then
    echo "Running cargo check..."
    if cargo check --features ml,vectordb --quiet 2>/dev/null; then
        echo "✅ Basic syntax check passed"
    else
        echo "⚠️  Syntax issues detected - run 'cargo check' for details"
    fi
else
    echo "⚠️  Cargo not available, skipping syntax check"
fi

echo ""

# Test structure validation
echo "📋 Validating test structure..."

structure_issues=0

for test_file in "${test_files[@]}"; do
    if [[ -f "$test_file" ]]; then
        # Check for basic test structure
        if grep -q "#\[tokio::test\]" "$test_file"; then
            echo "✅ $test_file: Contains async tests"
        elif grep -q "#\[test\]" "$test_file"; then
            echo "✅ $test_file: Contains tests"
        else
            echo "⚠️  $test_file: No test functions detected"
            structure_issues=$((structure_issues + 1))
        fi
        
        # Check for proper error handling
        if grep -q "expect\|unwrap" "$test_file"; then
            if grep -q "panic!" "$test_file"; then
                echo "✅ $test_file: Has proper test assertions"
            else
                echo "⚠️  $test_file: Uses expect/unwrap without panic assertions"
            fi
        fi
    fi
done

echo ""

# Generate summary
echo "📊 VALIDATION SUMMARY"
echo "===================="

if [[ $missing_files -eq 0 ]] && [[ $structure_issues -eq 0 ]]; then
    echo "✅ All validations passed!"
    echo "🚀 Test structure is ready for execution"
    echo ""
    echo "Next steps:"
    echo "1. Run individual test suites: cargo test --test <test_name> --features <features>"
    echo "2. Run full suite: ./scripts/run_embedding_tests.sh"
    echo "3. Check test documentation: tests/EMBEDDING_MIGRATION_TEST_SUITE.md"
    exit 0
else
    echo "❌ Issues detected:"
    if [[ $missing_files -gt 0 ]]; then
        echo "   - $missing_files missing files or dependencies"
    fi
    if [[ $structure_issues -gt 0 ]]; then
        echo "   - $structure_issues test structure issues"
    fi
    echo ""
    echo "Please fix these issues before running the test suite."
    exit 1
fi