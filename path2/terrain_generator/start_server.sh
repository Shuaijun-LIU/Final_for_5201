#!/bin/bash
# Quick start script for terrain visualizer

cd "$(dirname "$0")"

echo "Starting terrain visualizer server..."
echo ""
echo "Server will be available at:"
echo "  http://localhost:8000/visualize.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m http.server 8000

