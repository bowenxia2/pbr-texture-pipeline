#!/usr/bin/env bash
# Run the pipeline on all Articraft assets in batches of 8.
# Uses progressive --limit with --resume so each batch processes only new assets
# while the stage-major loop keeps model loading efficient within each batch.
#
# Usage:
#   bash scripts/run_articraft_batches.sh [START_BATCH]
#
# START_BATCH (1-indexed, default 1) lets you resume from a specific batch
# if a previous run was interrupted.

set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_SIZE=8
JOBS_ROOT=jobs_v2
ASSET_GLOB='articraft_extracted/*/model.urdf'
STAGES='render,vlm,imageedit,texture'
START_BATCH="${1:-1}"

TOTAL=$(ls -d $ASSET_GLOB 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 0 ]; then
    echo "No Articraft assets found matching $ASSET_GLOB"
    exit 1
fi

NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "=== Articraft batch pipeline ==="
echo "Total assets: $TOTAL"
echo "Batch size: $BATCH_SIZE"
echo "Total batches: $NUM_BATCHES"
echo "Starting from batch: $START_BATCH"
echo ""

for (( batch=1; batch<=NUM_BATCHES; batch++ )); do
    if [ "$batch" -lt "$START_BATCH" ]; then
        continue
    fi

    limit=$(( batch * BATCH_SIZE ))
    if [ "$limit" -gt "$TOTAL" ]; then
        limit=$TOTAL
    fi
    prev_limit=$(( (batch - 1) * BATCH_SIZE ))
    count=$(( limit - prev_limit ))

    echo "=============================================="
    echo "  Batch $batch / $NUM_BATCHES"
    echo "  Assets $((prev_limit + 1)) - $limit of $TOTAL ($count new)"
    echo "  --limit $limit --resume"
    echo "=============================================="

    conda run -n trellis2 python -m pbr_texture_pipeline.batch \
        --assets "$ASSET_GLOB" \
        --jobs-root "$JOBS_ROOT" \
        --stages "$STAGES" \
        --limit "$limit" \
        --resume \
        2>&1 | tee -a "$JOBS_ROOT/batch_log.txt"

    echo ""
    echo "Batch $batch / $NUM_BATCHES complete."
    echo ""
done

echo "=== All $NUM_BATCHES batches complete ==="
