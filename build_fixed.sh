#!/bin/bash

# Fixed build script for InfiniCore
# This script sets up the environment and builds with proper linker configuration

echo "Setting up InfiniCore build environment..."

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the infinicore-env environment
conda activate infinicore-env

# Set CUDA_HOME to the conda environment
export CUDA_HOME=$CONDA_PREFIX

# Clean up conflicting environment variables
unset CC
unset CXX
unset NVCC_PREPEND_FLAGS
unset NVCC_APPEND_FLAGS
unset CUDA_ROOT

# Use system tools
export PATH="/usr/bin:$PATH"

# Create a wrapper for ld that converts -m64 to -m elf_x86_64
mkdir -p /tmp/ld_wrapper
cat > /tmp/ld_wrapper/ld << 'EOF'
#!/bin/bash
# Convert -m64 to -m elf_x86_64 for system linker compatibility
args=()
skip_next=false
for arg in "$@"; do
    if [ "$skip_next" = true ]; then
        skip_next=false
        continue
    fi
    if [ "$arg" = "-m64" ]; then
        args+=("-m" "elf_x86_64")
    elif [ "$arg" = "-fopenmp" ]; then
        # Skip -fopenmp flag for linker, but add libgomp
        args+=("-lgomp")
        continue
    elif [ "$arg" = "-m" ]; then
        # Skip -m flag and its argument if it's elf_x86_64 (to avoid duplication)
        skip_next=true
        continue
    else
        args+=("$arg")
    fi
done
# Add standard C++ library and other required libraries
args+=("-lstdc++" "-lm" "-lc" "-lgcc_s")
exec /usr/bin/ld "${args[@]}"
EOF
chmod +x /tmp/ld_wrapper/ld
export PATH="/tmp/ld_wrapper:$PATH"

echo "Environment setup complete!"
echo "CUDA_HOME: $CUDA_HOME"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Configure and build
echo "Configuring xmake..."
xmake f -c

echo "Building InfiniCore..."
xmake build

echo "Build completed!"
