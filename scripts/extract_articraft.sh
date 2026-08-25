#!/bin/bash
# Extract Articraft-10K tar.gz assets into articraft_extracted/
# Usage: bash scripts/extract_articraft.sh [--limit N]

SRC="/mmfs1/gscratch/krishna/bxia2/Articraft-10K"
DST="articraft_extracted"

LIMIT=""
if [ "$1" = "--limit" ] && [ -n "$2" ]; then
    LIMIT=$2
fi

mkdir -p "$DST"

count=0
skipped=0
for f in "$SRC"/*.tar.gz; do
    base=$(basename "$f" .tar.gz)
    if [ -d "$DST/$base" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    tar -xzf "$f" -C "$DST/"
    count=$((count + 1))
    if [ -n "$LIMIT" ] && [ $count -ge $LIMIT ]; then
        break
    fi
done

total=$(ls -d "$DST"/*/ 2>/dev/null | wc -l)
echo "Extracted $count new assets ($skipped already existed). Total: $total"
