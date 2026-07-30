#!/bin/bash

# Source directory
SRC_DIR="/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio"

# HandBrake preset
PRESET="Creator 2160p60 4K"

# Max number of parallel jobs (adjust to your CPU cores)
MAX_JOBS=10
CURRENT_JOBS=0

# Function to process a single file
process_file() {
    FILE="$1"
    DIR=$(dirname "$FILE")
    BASE=$(basename "$FILE" .mp4)
    OUT="$DIR/${BASE}-YouTube.mp4"

    echo -e "Processing: $(basename "$FILE") \t->\t $(basename "$OUT")"
    HandBrakeCLI -v -i "$FILE" -o "$OUT" --preset "$PRESET"
}

export -f process_file
export PRESET

# Find and process files in parallel
find "$SRC_DIR" -name "*.mp4" -print0 | while IFS= read -r -d '' FILE; do
    process_file "$FILE" &   # Run in background
    ((CURRENT_JOBS++))

    # Limit number of parallel jobs
    if (( CURRENT_JOBS >= MAX_JOBS )); then
        wait -n              # Wait for at least one to finish
        ((CURRENT_JOBS--))
    fi
done

# Wait for all remaining background jobs
wait
