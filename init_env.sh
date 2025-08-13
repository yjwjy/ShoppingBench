#!/bin/bash

# Script to set up Python environment, install dependencies, and start search engine

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing..."
    pip install uv
else
    echo "uv is already installed"
fi

# Check if Python virtual environment already exists and has correct dependencies
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, checking dependencies..."
    source .venv/bin/activate
    # Check if requirements are already installed
    if uv pip install --dry-run -r requirements.txt &>/dev/null; then
        echo "Dependencies are already installed and up to date"
    else
        echo "Installing/Updating project dependencies..."
        uv pip install -r requirements.txt
    fi
else
    # Create Python 3.10 virtual environment (as per README)
    echo "Creating Python 3.10 virtual environment..."
    uv venv --python 3.10
    # Activate virtual environment
    echo "Activating virtual environment..."
    source .venv/bin/activate
    # Install project dependencies
    echo "Installing project dependencies..."
    uv pip install -r requirements.txt
fi

# Create resources directory if it doesn't exist
echo "Creating resources directory..."
mkdir -p resources

# Check if documents.jsonl exists in resources
if [ ! -f "resources/documents.jsonl" ]; then
    echo "Warning: documents.jsonl not found in resources directory"
    echo "Please place the documents.jsonl file in the resources directory"
else
    echo "Found documents.jsonl in resources directory"
fi

# Build search index if build_index.sh exists and index is not already built
if [ -f "build_index.sh" ]; then
    if [ ! -d "indexes" ] || [ -z "$(ls -A indexes)" ]; then
        echo "Building search index..."
        ./build_index.sh
    else
        echo "Search index already exists, skipping build."
    fi
else
    echo "Warning: build_index.sh not found"
fi

# Start search engine service in background and monitor output
echo "Starting search engine service..."
python src/search_engine/server.py > server.log 2>&1 &
SERVER_PID=$!

# Monitor the server output until "Load index done" is detected
echo "Waiting for server to load index..."
INDEX_LOADED=false
while kill -0 $SERVER_PID 2>/dev/null; do
    # Check if "Load index done" appears in the output
    if grep -q "Load indexes done." server.log; then
        echo "Index loaded successfully!"
        echo "Server is running in background with PID: $SERVER_PID"
        echo "Search engine service started successfully!"
        INDEX_LOADED=true
        break
    fi
    sleep 1
done

# Check if index was loaded successfully
if [ "$INDEX_LOADED" = true ]; then
    echo "Python environment setup complete and search engine started in background!"
    echo "Server log is being written to server.log"
    echo "Script completed successfully."
else
    echo "Server may have encountered an issue or failed to load index. Check server.log for details."
fi