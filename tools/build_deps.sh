#!/bin/bash

set -e  # Exit on error

# Get the script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================="
echo "Building drake-franka-setup dependencies"
echo "========================================="

# Build drake_franka_driver
echo ""
echo "Building drake_franka_driver..."
cd "$REPO_ROOT/deps/drake_franka_driver"
bazel build //...
echo "✓ drake_franka_driver build complete"

# Build franka_analytical_ik
echo ""
echo "Building franka_analytical_ik..."
cd "$REPO_ROOT/deps/franka_analytical_ik"
bazel build //...
echo "✓ franka_analytical_ik build complete"

# Optional: Build the wheel for franka_analytical_ik
echo ""
echo "Building franka_analytical_ik wheel..."
bazel build //:franka_ik_wheel
echo "✓ franka_analytical_ik wheel complete"

echo ""
echo "========================================="
echo "All dependencies built successfully!"
echo "========================================="
echo ""
echo "drake_franka_driver binaries:"
echo "  - $REPO_ROOT/deps/drake_franka_driver/bazel-bin/franka-driver/franka_driver_v4"
echo "  - $REPO_ROOT/deps/drake_franka_driver/bazel-bin/franka-driver/franka_driver_v5"
echo ""
echo "franka_analytical_ik wheel:"
echo "  - $REPO_ROOT/deps/franka_analytical_ik/bazel-bin/franka_ik_wheel.whl"
echo ""
