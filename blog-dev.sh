#!/bin/bash
# Hugo-only development server for rapid blog iteration
# Usage: ./blog-dev.sh

echo "Starting Hugo development server..."
cd blog
hugo server --buildDrafts --watch --baseURL http://localhost:1313/blog/

echo "Hugo blog: http://localhost:1313/blog/"
