#!/bin/bash

# InfiniCore Environment Setup Script
# This script sets up the proper environment for building InfiniCore with CUDA 12.9

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

# Use system linker to avoid conda cross-compilation issues
export LD=/usr/bin/ld
export PATH="/home/zenghua/miniconda3/envs/infinicore-env/bin:/usr/bin:$PATH"

# Suppress libtinfo version warning (cosmetic issue)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Create a wrapper for the conda linker that redirects to system linker and fixes flags
mkdir -p /tmp/ld_wrapper
cat > /tmp/ld_wrapper/x86_64-conda-linux-gnu-ld << 'EOF'
#!/bin/bash
# Convert -m64 to -m elf_x86_64 and remove -fopenmp for system linker compatibility
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
chmod +x /tmp/ld_wrapper/x86_64-conda-linux-gnu-ld
export PATH="/tmp/ld_wrapper:$PATH"

# Verify the setup
echo "Environment setup complete!"
echo "CUDA_HOME: $CUDA_HOME"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "NVCC version:"
nvcc --version

echo ""
echo "To build InfiniCore, run:"
echo "xmake f -c --nv-gpu=true --cuda=\$CUDA_HOME -vD"
echo "xmake build"
