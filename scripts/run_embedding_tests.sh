#!/bin/bash

# Comprehensive Embedding Migration Test Suite Runner
# This script runs all embedding migration tests with proper reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/embedding_migration_tests_$TIMESTAMP.log"

# Test suites configuration
declare -A TEST_SUITES=(
    ["unit_tests"]="ml:300:CRITICAL"
    ["integration_tests"]="ml,vectordb:600:CRITICAL"
    ["embedding_performance_bench"]="ml,vectordb:900:IMPORTANT"
    ["rag_system_validation"]="full-system:1200:CRITICAL"
    ["regression_tests"]="ml:600:CRITICAL"
    ["embedding_stress_tests"]="ml,vectordb:1800:IMPORTANT"
    ["llama_cpp_integration_tests"]="ml:900:CRITICAL"
    ["cache_validation_tests"]="ml:600:IMPORTANT"
)

# Global counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
CRITICAL_FAILURES=0

# Utility functions
log_message() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] $level: $message" | tee -a "$MAIN_LOG"
}

log_info() {
    log_message "${BLUE}INFO${NC}" "$@"
}

log_success() {
    log_message "${GREEN}SUCCESS${NC}" "$@"
}

log_warning() {
    log_message "${YELLOW}WARNING${NC}" "$@"
}

log_error() {
    log_message "${RED}ERROR${NC}" "$@"
}

log_critical() {
    log_message "${RED}CRITICAL${NC}" "$@"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check model file
    if [[ ! -f "./model/nomic-embed-code.Q4_K_M.gguf" ]]; then
        log_critical "Required model file not found: ./model/nomic-embed-code.Q4_K_M.gguf"
        log_critical "Please download the nomic-embed-code model before running tests"
        exit 1
    fi
    log_success "Model file found"
    
    # Check Rust toolchain
    if ! command -v cargo &> /dev/null; then
        log_critical "Cargo not found. Please install Rust toolchain"
        exit 1
    fi
    
    local rust_version=$(rustc --version)
    log_info "Rust version: $rust_version"
    
    # Check available memory
    if command -v free &> /dev/null; then
        local available_mem=$(free -h | awk '/^Mem:/ {print $7}')
        log_info "Available memory: $available_mem"
    fi
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    log_success "Prerequisites check completed"
}

# Run a single test suite
run_test_suite() {
    local test_name=$1
    local features=$2
    local timeout=$3
    local criticality=$4
    
    log_info "Running test suite: $test_name"
    log_info "  Features: $features"
    log_info "  Timeout: ${timeout}s"
    log_info "  Criticality: $criticality"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local test_log="$LOG_DIR/${test_name}_$TIMESTAMP.log"
    local start_time=$(date +%s)
    
    # Build the cargo command
    local cmd="cargo test --test $test_name"
    if [[ -n "$features" ]]; then
        cmd="$cmd --features $features"
    fi
    cmd="$cmd -- --nocapture"
    
    # Run the test with timeout
    log_info "Executing: $cmd"
    
    if timeout "${timeout}s" bash -c "$cmd" &> "$test_log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "✅ $test_name PASSED (${duration}s)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Extract key metrics from log
        if grep -q "test result: ok" "$test_log"; then
            local test_count=$(grep -o "[0-9]\\+ passed" "$test_log" | head -1 | grep -o "[0-9]\\+")
            if [[ -n "$test_count" ]]; then
                log_info "  Individual tests passed: $test_count"
            fi
        fi
        
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [[ $duration -ge $timeout ]]; then
            log_error "⏰ $test_name TIMEOUT after ${timeout}s"
        else
            log_error "❌ $test_name FAILED (${duration}s)"
        fi
        
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        if [[ "$criticality" == "CRITICAL" ]]; then
            CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
            log_critical "Critical test suite failed: $test_name"
        fi
        
        # Show last few lines of error output
        log_error "Last 10 lines of output:"\n        tail -n 10 "$test_log" | while read line; do
            log_error "  $line"
        done
    fi
    
    echo "" | tee -a "$MAIN_LOG"
}

# Generate final report
generate_report() {
    local total_time=$1
    
    echo "" | tee -a "$MAIN_LOG"
    echo "=====================================" | tee -a "$MAIN_LOG"
    echo "🏁 EMBEDDING MIGRATION TEST RESULTS" | tee -a "$MAIN_LOG" 
    echo "=====================================" | tee -a "$MAIN_LOG"
    
    log_info "Execution Summary:"
    log_info "  Total test suites: $TOTAL_TESTS"
    log_info "  ✅ Passed: $PASSED_TESTS"
    log_info "  ❌ Failed: $FAILED_TESTS"
    log_info "  🚨 Critical failures: $CRITICAL_FAILURES"
    log_info "  Total execution time: ${total_time}s"
    
    local success_rate=0
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi
    
    log_info "  Success rate: $success_rate%"
    
    echo "" | tee -a "$MAIN_LOG"
    
    # System validation decision
    if [[ $CRITICAL_FAILURES -eq 0 ]] && [[ $success_rate -ge 95 ]]; then
        echo "${GREEN}✅✅✅ SYSTEM VALIDATION: PASSED ✅✅✅${NC}" | tee -a "$MAIN_LOG"
        echo "The nomic-embed-code integration is validated and ready for production." | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        
        # Success recommendations
        log_success "🎉 All critical tests passed!"
        log_success "🚀 System is ready for production deployment"
        log_success "📊 Performance metrics within acceptable ranges"
        log_success "🔒 No critical security or functionality issues detected"
        
    elif [[ $CRITICAL_FAILURES -eq 0 ]] && [[ $success_rate -ge 80 ]]; then
        echo "${YELLOW}⚠️⚠️⚠️ SYSTEM VALIDATION: CONDITIONAL ⚠️⚠️⚠️${NC}" | tee -a "$MAIN_LOG"
        echo "System may work but has issues that should be addressed." | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        
        # Conditional recommendations
        log_warning "⚠️ Some non-critical tests failed"
        log_warning "📋 Review failed tests and consider fixes"
        log_warning "🧪 Consider additional testing in staging environment"
        log_warning "📈 Monitor system closely after deployment"
        
    else
        echo "${RED}❌❌❌ SYSTEM VALIDATION: FAILED ❌❌❌${NC}" | tee -a "$MAIN_LOG"
        echo "System is NOT ready for production. Critical issues must be fixed." | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        
        # Failure recommendations
        log_critical "🚨 CRITICAL FAILURES DETECTED"
        log_critical "🛑 DO NOT DEPLOY TO PRODUCTION"
        log_critical "🔧 Fix all critical issues before proceeding"
        log_critical "🧪 Re-run full test suite after fixes"
        
        if [[ $CRITICAL_FAILURES -gt 0 ]]; then
            log_critical "📋 Critical test suites that failed:"
            for suite_name in "${!TEST_SUITES[@]}"; do
                IFS=':' read -ra SUITE_INFO <<< "${TEST_SUITES[$suite_name]}"
                if [[ "${SUITE_INFO[2]}" == "CRITICAL" ]]; then
                    log_critical "  - $suite_name"
                fi
            done
        fi
    fi
    
    echo "" | tee -a "$MAIN_LOG"
    log_info "📁 Detailed logs available in: $LOG_DIR/"
    log_info "📊 Main log file: $MAIN_LOG"
    
    # Return appropriate exit code
    if [[ $CRITICAL_FAILURES -eq 0 ]] && [[ $success_rate -ge 95 ]]; then
        return 0
    else
        return 1
    fi
}

# Main execution
main() {
    echo "🚀 Starting Comprehensive Embedding Migration Test Suite" | tee "$MAIN_LOG"
    echo "=======================================================" | tee -a "$MAIN_LOG"
    echo "Timestamp: $(date)" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    
    local overall_start=$(date +%s)
    
    # Check prerequisites
    check_prerequisites
    echo "" | tee -a "$MAIN_LOG"
    
    # Run all test suites
    for suite_name in "${!TEST_SUITES[@]}"; do
        IFS=':' read -ra SUITE_INFO <<< "${TEST_SUITES[$suite_name]}"
        local features="${SUITE_INFO[0]}"
        local timeout="${SUITE_INFO[1]}"
        local criticality="${SUITE_INFO[2]}"
        
        run_test_suite "$suite_name" "$features" "$timeout" "$criticality"
    done
    
    local overall_end=$(date +%s)
    local total_time=$((overall_end - overall_start))
    
    # Generate final report
    generate_report "$total_time"
}

# Handle interrupts
cleanup() {
    log_warning "Test execution interrupted"
    generate_report "INTERRUPTED"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"

# Exit with appropriate code
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    log_success "🎉 Test suite completed successfully!"
else
    log_critical "💥 Test suite completed with failures!"
fi

exit $exit_code