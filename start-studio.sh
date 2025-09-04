#!/bin/bash

# Exit on error
set -e

# Check if yarn is available
if ! command -v yarn &> /dev/null; then
    echo "Error: yarn is not installed. Please install yarn first."
    echo "Run: npm install -g yarn"
    exit 1
fi

# Change to coral-studio directory
cd coral-studio

# Install dependencies and start development server
echo "Installing dependencies..."
yarn install

echo "Starting development server..."
yarn dev
