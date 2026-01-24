#!/bin/bash
# Shader Compilation Script
#
# Compiles GLSL compute shaders to SPIR-V using glslangValidator or shaderc.
#
# Prerequisites:
#   - glslangValidator (from Vulkan SDK or glslang package)
#   - OR shaderc (glslc compiler)
#
# Usage:
#   ./compile.sh          # Compile all shaders
#   ./compile.sh clean    # Remove compiled SPIR-V files
#   ./compile.sh check    # Check if compiler is available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect compiler
detect_compiler() {
    if command -v glslangValidator &> /dev/null; then
        COMPILER="glslangValidator"
        return 0
    elif command -v glslc &> /dev/null; then
        COMPILER="glslc"
        return 0
    else
        return 1
    fi
}

print_install_instructions() {
    echo "Error: No GLSL compiler found."
    echo "Please install one of:"
    echo "  - glslang (glslangValidator)"
    echo "  - shaderc (glslc)"
    echo ""
    echo "Ubuntu/Debian: sudo apt install glslang-tools"
    echo "Fedora: sudo dnf install glslang"
    echo "Arch: sudo pacman -S glslang"
    echo "macOS: brew install glslang"
    echo "Windows: Install Vulkan SDK from https://vulkan.lunarg.com/"
}

# Compile a single shader file
compile_shader() {
    local shader="$1"
    local output="$2"

    echo "Compiling $shader -> $output"
    if [ "$COMPILER" = "glslangValidator" ]; then
        glslangValidator -V --target-env vulkan1.0 -o "$output" "$shader"
    else
        glslc -fshader-stage=compute --target-env=vulkan1.0 -O -o "$output" "$shader"
    fi
}

# Check command
if [ "$1" = "check" ]; then
    if detect_compiler; then
        echo "Compiler available: $COMPILER"
        exit 0
    else
        print_install_instructions
        exit 1
    fi
fi

# Clean command
if [ "$1" = "clean" ]; then
    echo "Cleaning compiled shaders..."
    rm -f *.spv
    echo "Done."
    exit 0
fi

# Require compiler for compilation
if ! detect_compiler; then
    print_install_instructions
    exit 1
fi

echo "Using compiler: $COMPILER"
echo ""

# Track compilation results
COMPILED=0
FAILED=0

# Compile .comp files (legacy format)
for shader in *.comp 2>/dev/null; do
    if [ -f "$shader" ]; then
        output="${shader%.comp}.spv"
        if compile_shader "$shader" "$output"; then
            ((COMPILED++))
        else
            echo "  FAILED: $shader"
            ((FAILED++))
        fi
    fi
done

# Compile .comp.glsl files (new format)
for shader in *.comp.glsl 2>/dev/null; do
    if [ -f "$shader" ]; then
        # Remove both .comp.glsl to get base name
        output="${shader%.comp.glsl}.spv"
        if compile_shader "$shader" "$output"; then
            ((COMPILED++))
        else
            echo "  FAILED: $shader"
            ((FAILED++))
        fi
    fi
done

echo ""
echo "Compilation complete: $COMPILED succeeded, $FAILED failed"
echo ""
echo "Generated SPIR-V files:"
ls -la *.spv 2>/dev/null || echo "  (none)"
