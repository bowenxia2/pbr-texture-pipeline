#!/bin/bash
# Process Articraft assets one at a time through the full pipeline.
# Each asset completes all stages before moving to the next.
# Usage: bash scripts/run_one_by_one.sh [--limit N]

JOBS_ROOT="jobs_v2"
LIMIT=""
if [ "$1" = "--limit" ] && [ -n "$2" ]; then
    LIMIT=$2
fi

count=0
done_count=0
fail_count=0

for d in "$JOBS_ROOT"/rec_*/; do
    [ -d "$d" ] || continue
    job_id=$(basename "$d")

    judge_status=$(python3 -c "import json; j=json.load(open('$d/job.json')); print(j.get('stages',{}).get('judge',{}).get('status','?'))" 2>/dev/null)
    if [ "$judge_status" = "done" ]; then
        continue
    fi

    render_status=$(python3 -c "import json; j=json.load(open('$d/job.json')); print(j.get('stages',{}).get('render',{}).get('status','?'))" 2>/dev/null)
    if [ "$render_status" != "done" ]; then
        continue
    fi

    # Find the original URDF for this job
    urdf_path=$(python3 -c "import json; j=json.load(open('$d/job.json')); print(j.get('asset',{}).get('urdf',''))" 2>/dev/null)
    if [ -z "$urdf_path" ]; then
        echo "[one-by-one] SKIP $job_id: no urdf in job.json"
        continue
    fi

    count=$((count + 1))
    echo ""
    echo "==== [$count] $job_id ===="

    conda run -n trellis2 --no-capture-output python -m pbr_texture_pipeline.batch \
        --assets "$urdf_path" \
        --jobs-root "$JOBS_ROOT" \
        --stages vlm,diffuse,plan,texture,eval,judge \
        --gpu-mode single \
        --resume 2>&1

    rc=$?
    if [ $rc -eq 0 ]; then
        judge_now=$(python3 -c "import json; j=json.load(open('$d/job.json')); print(j.get('stages',{}).get('judge',{}).get('status','?'))" 2>/dev/null)
        if [ "$judge_now" = "done" ]; then
            done_count=$((done_count + 1))
            echo "[one-by-one] DONE $job_id ($done_count complete, $fail_count failed)"
        else
            fail_count=$((fail_count + 1))
            echo "[one-by-one] PARTIAL $job_id ($done_count complete, $fail_count failed)"
        fi
    else
        fail_count=$((fail_count + 1))
        echo "[one-by-one] FAIL $job_id exit=$rc ($done_count complete, $fail_count failed)"
    fi

    if [ -n "$LIMIT" ] && [ $count -ge $LIMIT ]; then
        echo "[one-by-one] limit reached ($LIMIT)"
        break
    fi
done

echo ""
echo "[one-by-one] SUMMARY: processed=$count done=$done_count failed=$fail_count"
