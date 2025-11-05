#!/bin/bash
# MNIST Benchmark Comparison: Charl vs PyTorch
# Runs both benchmarks and generates comparison report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}     MNIST Benchmark: Charl vs PyTorch         ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if Charl benchmark is built
if [ ! -f "$PROJECT_ROOT/target/release/charl_mnist_bench" ]; then
    echo -e "${YELLOW}Building Charl benchmark...${NC}"
    cd "$PROJECT_ROOT"
    cargo build --release --bin charl_mnist_bench
fi

# Check if Python venv exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${RED}Error: Python virtual environment not found${NC}"
    echo "Please create it with: python3 -m venv venv"
    echo "Then install PyTorch: venv/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cpu"
    exit 1
fi

echo -e "${GREEN}Running Charl benchmark...${NC}"
echo ""
cd "$PROJECT_ROOT"
CHARL_OUTPUT=$("$PROJECT_ROOT/target/release/charl_mnist_bench" | tee "$RESULTS_DIR/charl_results_${TIMESTAMP}.txt")
CHARL_TIME=$(echo "$CHARL_OUTPUT" | grep "Total training time:" | awk -F': ' '{print $2}')
CHARL_SAMPLES=$(echo "$CHARL_OUTPUT" | grep "Samples per second:" | awk -F': ' '{print $2}')

echo ""
echo -e "${GREEN}Running PyTorch benchmark...${NC}"
echo ""
PYTORCH_OUTPUT=$("$PROJECT_ROOT/venv/bin/python3" "$SCRIPT_DIR/pytorch_mnist.py" | tee "$RESULTS_DIR/pytorch_results_${TIMESTAMP}.txt")
PYTORCH_TIME=$(echo "$PYTORCH_OUTPUT" | grep "Total training time:" | awk -F': ' '{print $2}')
PYTORCH_SAMPLES=$(echo "$PYTORCH_OUTPUT" | grep "Samples per second:" | awk -F': ' '{print $2}')

# Calculate speedup
SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CHARL_SAMPLES / $PYTORCH_SAMPLES}")

# Generate comparison report
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}              COMPARISON RESULTS                ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo "Configuration:"
echo "  Model: MNIST Classifier (784→128→64→10)"
echo "  Parameters: 109,386"
echo "  Dataset: 1,000 synthetic samples"
echo "  Batch size: 32"
echo "  Epochs: 5"
echo "  Learning rate: 0.001"
echo ""
echo "Performance:"
echo -e "  ${GREEN}Charl:${NC}"
echo "    Total time: $CHARL_TIME"
echo "    Throughput: $CHARL_SAMPLES samples/sec"
echo ""
echo -e "  ${YELLOW}PyTorch:${NC}"
echo "    Total time: $PYTORCH_TIME"
echo "    Throughput: $PYTORCH_SAMPLES samples/sec"
echo ""
echo -e "  ${GREEN}⚡ SPEEDUP: ${SPEEDUP}x faster! ⚡${NC}"
echo ""
echo -e "${BLUE}=================================================${NC}"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/charl_results_${TIMESTAMP}.txt"
echo "  - $RESULTS_DIR/pytorch_results_${TIMESTAMP}.txt"
echo ""

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/comparison_summary_${TIMESTAMP}.txt"
cat > "$SUMMARY_FILE" << EOF
MNIST Benchmark Comparison: Charl vs PyTorch
Date: $(date)
=================================================

Configuration:
  Model: MNIST Classifier (784→128→64→10)
  Parameters: 109,386
  Dataset: 1,000 synthetic samples
  Batch size: 32
  Epochs: 5
  Learning rate: 0.001

Results:
  Charl:
    Total time: $CHARL_TIME
    Throughput: $CHARL_SAMPLES samples/sec

  PyTorch:
    Total time: $PYTORCH_TIME
    Throughput: $PYTORCH_SAMPLES samples/sec

  SPEEDUP: ${SPEEDUP}x faster!

Environment:
  Platform: Linux $(uname -r)
  Python: $(python3 --version)
  PyTorch: $("$PROJECT_ROOT/venv/bin/python3" -c "import torch; print(torch.__version__)")
  Rust: $(~/.cargo/bin/rustc --version 2>/dev/null || echo "rustc not in PATH")
EOF

echo "Summary saved to: $SUMMARY_FILE"
echo ""
