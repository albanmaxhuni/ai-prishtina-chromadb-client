#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to handle errors
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

print_section "Starting test preparation"

# Clean up test artifacts
print_section "Cleaning up test artifacts"
rm -rf tests/__pycache__
rm -rf tests/.pytest_cache
rm -rf tests/.coverage*
rm -rf tests/test_chroma_db
rm -rf tests/.chroma
rm -rf tests/data/*
rm -rf tests/logs/*

# Create necessary directories
print_section "Creating test directories"
mkdir -p tests/data
mkdir -p tests/logs

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    print_section "Activating virtual environment"
    source .venv/bin/activate
else
    print_section "Creating virtual environment"
    python -m venv .venv
    source .venv/bin/activate
fi

# Install/upgrade pip
print_section "Upgrading pip"
pip install --upgrade pip

# Install dependencies
print_section "Installing dependencies"
pip install -e .
pip install pytest pytest-cov pytest-xdist

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TESTING=True
export PYTEST_ADDOPTS="--tb=short"

# Run tests with detailed reporting
print_section "Running tests"
python -m pytest tests/ -v \
    --cov=ai_prishtina_vectordb \
    --cov-report=term-missing \
    --cov-report=html \
    --junitxml=test-results.xml \
    --durations=10 \
    --maxfail=1 \
    --showlocals

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}All tests completed successfully!${NC}"
    echo -e "\n${BLUE}Coverage report generated in htmlcov/index.html${NC}"
    echo -e "${BLUE}Test results saved in test-results.xml${NC}"
else
    echo -e "\n${RED}Some tests failed. Please check the output above.${NC}"
    echo -e "${YELLOW}Detailed test results are available in:${NC}"
    echo -e "- HTML coverage report: htmlcov/index.html"
    echo -e "- JUnit XML report: test-results.xml"
fi

# Deactivate virtual environment
deactivate 