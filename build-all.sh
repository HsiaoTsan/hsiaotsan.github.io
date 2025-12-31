#!/bin/bash
# Combined build script for Jekyll homepage + Hugo blog
# Usage: ./build-all.sh [serve]

set -e

echo "Building Hugo blog..."
cd blog && hugo --cleanDestinationDir && cd ..

echo "Building Jekyll site..."
if [ "$1" = "serve" ]; then
    echo "Starting Jekyll server..."
    bundle exec jekyll serve --watch
else
    bundle exec jekyll build
fi

echo "Build complete!"
echo "Jekyll homepage: http://localhost:4000"
echo "Hugo blog: http://localhost:4000/blog/"
