#!/bin/bash

# Script to run the full integration test for AC/DC pipeline
# Usage: ./tests/run_integration_test.sh [--skip-build] [--keep-output]

set -e  # Exit on error

echo "=========================================="
echo "AC/DC Full Integration Test Runner"
echo "=========================================="
echo ""

# Parse arguments
SKIP_BUILD=false
KEEP_OUTPUT=false
EXISTING_RUN_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --keep-output)
            KEEP_OUTPUT=true
            shift
            ;;
        --existing-run-path)
            EXISTING_RUN_PATH="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-build    Skip environment setup/dependency check"
            echo "  --keep-output   Keep test output for inspection (don't clean up)"
            echo "  --existing-run-path Path to existing run to use for test"
            echo "  --help          Show this help message"
            echo ""
            echo "This script runs the full integration test with proper setup."
            echo "Expected runtime: 30-60 minutes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check dependencies (unless --skip-build)
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "Checking dependencies..."

    # Check if pytest is installed
    if ! python -c "import pytest" 2>/dev/null; then
        echo "ERROR: pytest not found. Installing test dependencies..."
        pip install -r requirements-dev.txt
    else
        echo "✓ pytest is installed"
    fi

    # Check if main dependencies are installed
    if ! python -c "import torch" 2>/dev/null; then
        echo "ERROR: torch not found. Installing main dependencies..."
        pip install -r requirements.txt
    else
        echo "✓ torch is installed"
    fi
fi

# Check if configuration file exists
echo ""
echo "Checking test configuration..."
if [ ! -f "configs/test_integration.yaml" ]; then
    echo "ERROR: Test configuration not found: configs/test_integration.yaml"
    exit 1
fi
echo "✓ Test configuration exists"

# Check Docker (optional but recommended)
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo "✓ Docker is available and running"
    else
        echo "⚠ Docker is installed but may not be running"
        echo "  Some tests may fail without Docker"
    fi
else
    echo "⚠ Docker not found"
    echo "  AC/DC sandbox tests will fail without Docker"
fi

# Check GPU
echo ""
echo "Checking GPU availability..."
if python -c "import torch; print('✓ GPU available' if torch.cuda.is_available() else '⚠ No GPU found')" 2>/dev/null; then
    :  # Output already printed by python command
else
    echo "⚠ Could not check GPU status"
fi

# Run the test
echo ""
echo "=========================================="
echo "Starting Full Integration Test"
echo "=========================================="
echo ""
echo "This will take approximately 30-60 minutes..."
echo "Test output will be displayed in real-time."
echo ""

# Parse --existing-run-path argument (already exists!)
# Export as environment variable for pytest
if [ -n "$EXISTING_RUN_PATH" ]; then
    export PYTEST_EXISTING_RUN_PATH="$EXISTING_RUN_PATH"
    echo "Using existing run path: $EXISTING_RUN_PATH"
    output_dir="$EXISTING_RUN_PATH"
else
    # Create timestamp for output directory
    timestamp=$(date +%Y%m%d_%H%M%S)
    output_dir="outputs/test_integration_${timestamp}"
fi

echo "Test output directory: $output_dir"
echo ""

# Run pytest with full output
if python -m pytest tests/integration/test_full_pipeline.py \
    -v \
    -s \
    --log-cli-level=INFO \
    --tb=short \
    --junit-xml=test_results.xml; then

    echo ""
    echo "=========================================="
    echo "✓ Integration Test PASSED"
    echo "=========================================="
    echo ""

    # Show summary
    if [ -d "$output_dir" ]; then
        echo "Test artifacts created in: $output_dir"
        echo ""
        echo "Directory structure:"
        ls -lh "$output_dir" 2>/dev/null || echo "  (output dir not found)"
        echo ""

        if [ "$KEEP_OUTPUT" = false ]; then
            echo "Cleaning up test output..."
            # Uncomment to enable automatic cleanup
            # rm -rf "$output_dir"
            echo "  (cleanup disabled, manually remove if needed)"
        else
            echo "Test output preserved at: $output_dir"
        fi
    fi

    echo ""
    echo "Test results saved to: test_results.xml"
    exit 0

else
    echo ""
    echo "=========================================="
    echo "✗ Integration Test FAILED"
    echo "=========================================="
    echo ""
    echo "Check the output above for error details."
    echo "Test results saved to: test_results.xml"
    echo ""

    if [ -d "$output_dir" ]; then
        echo "Test artifacts may be available in: $output_dir"
        echo "Inspect these files for debugging:"
        echo "  - Archives: $output_dir/archives/"
        echo "  - Models: $output_dir/models/"
        echo "  - Tasks: $output_dir/generated_tasks/"
    fi

    exit 1
fi
