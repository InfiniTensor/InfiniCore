#!/bin/bash

# Setup cuDNN environment variables for InfiniCore build
# This script sets up all necessary environment variables for cuDNN linking and includes

# Get the conda environment path
CONDA_ENV_PATH="/home/zenghua/miniconda3/envs/infinicore-env"
CUDNN_PATH="$CONDA_ENV_PATH/lib/python3.11/site-packages/nvidia/cudnn"

# Set cuDNN environment variables
export CUDNN_ROOT="$CUDNN_PATH"
export CUDNN_INCLUDE_DIR="$CUDNN_PATH/include"
export CUDNN_LIB_DIR="$CUDNN_PATH/lib"
export CUDNN_HOME="$CUDNN_PATH"
export CUDNN_PATH="$CUDNN_PATH"

# Add cuDNN library path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"

# Add pkg-config path for cuDNN
export PKG_CONFIG_PATH="$CUDNN_PATH/lib/pkgconfig:$PKG_CONFIG_PATH"

# Create symbolic links for cuDNN libraries if they don't exist
if [ -d "$CUDNN_LIB_DIR" ]; then
    cd "$CUDNN_LIB_DIR"
    [ ! -L libcudnn.so ] && ln -sf libcudnn.so.9 libcudnn.so
    [ ! -L libcudnn_ops.so ] && ln -sf libcudnn_ops.so.9 libcudnn_ops.so
    [ ! -L libcudnn_cnn.so ] && ln -sf libcudnn_cnn.so.9 libcudnn_cnn.so
    [ ! -L libcudnn_adv.so ] && ln -sf libcudnn_adv.so.9 libcudnn_adv.so
    cd - > /dev/null
fi

# Also set CUDA environment variables for completeness
export CUDA_HOME="$CONDA_ENV_PATH"
export CUDA_ROOT="$CONDA_ENV_PATH"
export CUDA_PATH="$CONDA_ENV_PATH"

echo "cuDNN environment variables set:"
echo "  CUDNN_ROOT=$CUDNN_ROOT"
echo "  CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR"
echo "  CUDNN_LIB_DIR=$CUDNN_LIB_DIR"
echo "  CUDNN_HOME=$CUDNN_HOME"
echo "  CUDNN_PATH=$CUDNN_PATH"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  CUDA_HOME=$CUDA_HOME"
echo ""
echo "You can now run: xmake build"
