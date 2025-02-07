#!/bin/bash

# Function to check if conda is initialized
check_conda_init() {
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed or not in PATH"
        return 1
    fi
    return 0
}

# Function to initialize conda
init_conda() {
    # Get the path to conda.sh
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    
    if [ $? -ne 0 ]; then
        echo "Failed to source conda.sh"
        exit 1
    fi
    
    echo "Conda initialized successfully"
}

# Function to check CUDA version
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
        echo "CUDA version detected: $CUDA_VERSION"
        return 0
    else
        echo "NVIDIA driver not found, installing CPU-only version"
        return 1
    fi
}

# Function to setup Ollama
setup_ollama() {
    if ! command -v ollama &> /dev/null; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if [ $? -ne 0 ]; then
            echo "Failed to install Ollama"
            exit 1
        fi
        echo "Ollama installed successfully"
    else
        echo "Ollama is already installed"
    fi
    
    echo "Pulling default model (llama3.2)..."
    ollama pull llama3.2
    if [ $? -ne 0 ]; then
        echo "Failed to pull llama3.2 model"
        exit 1
    fi
    echo "Model pulled successfully"
}

# Main script
echo "Setting up Ollama..."
setup_ollama

echo "Checking conda initialization..."
if ! check_conda_init; then
    echo "Please install conda first"
    exit 1
fi

echo "Initializing conda..."
init_conda

# Check if dp-clinical environment exists
if conda env list | grep -q "dp-clinical"; then
    echo "Removing existing dp-clinical environment..."
    conda deactivate
    conda env remove -n dp-clinical -y
fi

echo "Creating new dp-clinical environment..."
conda create -n dp-clinical python=3.9 -y
conda activate dp-clinical

# Install PyTorch with appropriate CUDA version
echo "Checking CUDA availability..."
if check_cuda; then
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "Installing CPU-only PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

echo "Installing other requirements..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py 