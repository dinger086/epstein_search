#!/usr/bin/env bash
# Wait for file transfer to complete, then start indexing.
# Watches the data/ folder and waits until no files have changed
# for 2 minutes (meaning the transfer is done), then runs indexing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
STABLE_SECONDS=120  # 2 minutes of no changes = transfer done
CHECK_INTERVAL=10   # Check every 10 seconds

echo "========================================"
echo " Wait-and-Index Script"
echo "========================================"
echo ""
echo "Watching: $DATA_DIR"
echo "Will start indexing after $STABLE_SECONDS seconds of no file changes."
echo "Go to bed - this will handle it!"
echo ""

# Wait for data directory to exist
while [ ! -d "$DATA_DIR" ]; do
    echo "$(date '+%H:%M:%S') - Waiting for data directory to appear..."
    sleep "$CHECK_INTERVAL"
done

get_snapshot() {
    # Get file count and total size
    local count size
    count=$(find "$DATA_DIR" -type f 2>/dev/null | wc -l)
    size=$(du -sb "$DATA_DIR" 2>/dev/null | cut -f1)
    echo "${count}:${size:-0}"
}

last_snapshot=$(get_snapshot)
stable_since=$(date +%s)

last_count="${last_snapshot%%:*}"
last_size="${last_snapshot##*:}"
echo "$(date '+%H:%M:%S') - Initial state: $last_count files, $((last_size / 1048576)) MB"

while true; do
    sleep "$CHECK_INTERVAL"
    current=$(get_snapshot)
    current_count="${current%%:*}"
    current_size="${current##*:}"

    if [ "$current" != "$last_snapshot" ]; then
        # Files are still changing - reset the timer
        stable_since=$(date +%s)
        last_snapshot="$current"
        echo "$(date '+%H:%M:%S') - Transfer in progress: $current_count files, $((current_size / 1048576)) MB"
    else
        now=$(date +%s)
        elapsed=$((now - stable_since))
        remaining=$((STABLE_SECONDS - elapsed))
        if [ "$remaining" -le 0 ]; then
            echo ""
            echo "$(date '+%H:%M:%S') - No changes for $STABLE_SECONDS seconds. Transfer appears complete!"
            echo "Final state: $current_count files, $((current_size / 1048576)) MB"
            break
        else
            echo "$(date '+%H:%M:%S') - Stable for ${elapsed}s / ${STABLE_SECONDS}s needed..."
        fi
    fi
done

# Transfer is done - start indexing
echo ""
echo "========================================"
echo " Starting indexing..."
echo "========================================"
echo ""

cd "$SCRIPT_DIR"
uv run python -m main index --mode hybrid

echo ""
echo "========================================"
echo " Done! Indexing complete."
echo "========================================"
