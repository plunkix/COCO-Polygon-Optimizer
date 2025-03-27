#!/bin/bash
# Setup script for COCO Polygon Optimizer with Streamlit Frontend

echo "Setting up COCO Polygon Optimizer with Streamlit Frontend..."

# Create required directories
mkdir -p input output images screenshots

# Check if the main Python script exists
if [ ! -f "coco_polygon_optimizer.py" ]; then
    echo "Copying coco_polygon_optimizer.py to the current directory..."
    if [ -f "./documents/coco_polygon_optimizer.py" ]; then
        cp ./documents/coco_polygon_optimizer.py .
    else
        echo "Error: coco-polygon-optimizer.py not found!"
        echo "Please make sure to copy the script to the current directory."
        exit 1
    fi
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
numpy>=1.20.0
opencv-python>=4.5.0
matplotlib>=3.3.0
shapely>=1.7.0
tqdm>=4.50.0
EOF
fi

# Create Streamlit app if it doesn't exist
if [ ! -f "streamlit_app.py" ]; then
    echo "Creating Streamlit app..."
    if [ -f "./documents/streamlit_app.py" ]; then
        cp ./documents/streamlit_app.py .
    else
        echo "Downloading Streamlit app template..."
        curl -s https://raw.githubusercontent.com/yourusername/coco-polygon-optimizer/main/streamlit_app.py > streamlit_app.py 2>/dev/null || {
            echo "Warning: Could not download Streamlit app. Creating a template..."
            cat > streamlit_app.py << EOF
import streamlit as st
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the optimizer
try:
    from coco_polygon_optimizer import optimize_coco_annotations, generate_comparisons_opencv
    st.success("Successfully imported the optimizer module")
except ImportError as e:
    st.error(f"Could not import the optimizer module: {e}")
    st.info("Please make sure coco-polygon-optimizer.py is in the same directory")

st.title("COCO Polygon Optimizer")
st.write("This is a placeholder Streamlit app for the COCO Polygon Optimizer.")
st.write("Please replace this file with the actual streamlit_app.py file.")
EOF
        }
    fi
fi

# Create Dockerfile.frontend if it doesn't exist
if [ ! -f "Dockerfile.frontend" ]; then
    echo "Creating Dockerfile.frontend..."
    cat > Dockerfile.frontend << EOF
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Add Streamlit to requirements
RUN echo "streamlit>=1.12.0" >> requirements.txt
RUN echo "pillow>=9.0.0" >> requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY coco-polygon-optimizer.py .
COPY LICENSE .
COPY README.md .

# Make the script executable
RUN chmod +x coco-polygon-optimizer.py

# Copy the Streamlit app
COPY streamlit_app.py .

# Create directories for input, output and images
RUN mkdir -p input output images

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
EOF
fi

# Create docker-compose.yml if it doesn't exist or update it
if [ ! -f "docker-compose.yml" ]; then
    echo "Creating docker-compose.yml..."
    cat > docker-compose.yml << EOF
version: '3'

services:
  # Core optimizer service
  optimizer:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./input:/data/input
      - ./output:/data/output
      - ./images:/data/images
    command: --help
    # This service just builds the image to be used by the frontend
    profiles:
      - cli-only

  # Streamlit frontend
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./images:/app/images
      - ./streamlit_app.py:/app/streamlit_app.py
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
EOF
else
    echo "docker-compose.yml already exists. Adding frontend service if not present..."
    if ! grep -q "frontend:" docker-compose.yml; then
        cat >> docker-compose.yml << EOF

  # Streamlit frontend
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./images:/app/images
      - ./streamlit_app.py:/app/streamlit_app.py
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
EOF
    fi
fi

# Create a README if it doesn't exist
if [ ! -f "README.frontend.md" ]; then
    echo "Creating README.frontend.md..."
    cat > README.frontend.md << EOF
# COCO Polygon Optimizer with Streamlit Frontend

This project adds a user-friendly web interface to the COCO Polygon Optimizer tool, making it easier to simplify and optimize polygon annotations in COCO format datasets.

## Quick Start

1. Run the setup script:
   \`\`\`bash
   chmod +x setup-frontend.sh
   ./setup-frontend.sh
   \`\`\`

2. Launch the Streamlit frontend:
   \`\`\`bash
   docker-compose up frontend
   \`\`\`

3. Open your browser and navigate to:
   \`\`\`
   http://localhost:8501
   \`\`\`

See the full documentation for more details.
EOF
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed successfully!"
    echo ""
    echo "To start the Streamlit frontend, run:"
    echo "docker-compose up frontend"
    echo ""
    echo "Then open your browser and navigate to:"
    echo "http://localhost:8501"
    echo ""
    echo "Please place your COCO annotation JSON file in the 'input' directory."
    echo "For visualization, place your images in the 'images' directory."
    echo "Results will be saved to the 'output' directory."
else
    echo "Error building Docker images. Please check the logs above for details."
fi