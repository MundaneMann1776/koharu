#!/usr/bin/env bash
# Build vision.cpp and install the vision-cli binary.
# Requires: cmake, a C++20 compiler (Xcode Command Line Tools on macOS)
#
# Usage:
#   ./build.sh               # CPU-only
#   ./build.sh --vulkan      # CPU + Vulkan (requires Vulkan SDK)
#
# After building, copy the binary:
#   cp build/bin/vision-cli /usr/local/bin/
set -euo pipefail

REPO_URL="https://github.com/Acly/vision.cpp.git"
BUILD_DIR="$(dirname "$0")/vision-cpp"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning vision.cpp..."
    git clone --recursive "$REPO_URL" "$BUILD_DIR"
else
    echo "Updating vision.cpp..."
    git -C "$BUILD_DIR" pull --recurse-submodules
fi

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
if [[ "${1:-}" == "--vulkan" ]]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DVISP_VULKAN=ON"
    echo "Building with Vulkan support..."
else
    echo "Building CPU-only (no Metal on macOS via vision.cpp)..."
fi

cmake "$BUILD_DIR" -B "$BUILD_DIR/build" $CMAKE_FLAGS
cmake --build "$BUILD_DIR/build" --config Release --parallel "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"

BINARY="$BUILD_DIR/build/bin/vision-cli"
echo ""
echo "Build complete: $BINARY"
echo ""
echo "Install with:"
echo "  sudo cp '$BINARY' /usr/local/bin/"
