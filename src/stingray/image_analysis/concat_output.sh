#!/bin/bash
###############################################################################
# Parallel YOLO Label File Aggregator (Optimized + Reproducible)
# ---------------------------------------------------------------
# This script:
#   1. Scans multiple YOLO label directories for *.txt annotation files
#   2. Saves a timestamped file list snapshot for reproducibility
#   3. Uses parallel workers to read & merge label records into a large CSV
#   4. Cleans up temporary resources automatically

###############################################################################

set -o pipefail

########################################
# USER CONFIGURATION
########################################
INPUT_DIRS=(
    "/proj/omics/sosik/yolozone/yolo-run-1/gpu0/labels"
    "/proj/omics/sosik/yolozone/yolo-run-1/gpu1/labels"
    "/proj/omics/sosik/yolozone/yolo-run-1/gpu2/labels"
)

OUTPUT_DIR="/user/huy.pham/"
OUTPUT_CSV="$OUTPUT_DIR/en706_yolo_concatenated_results.csv"

########################################
# ENVIRONMENT SETUP
########################################

# Temporary workspace — deleted automatically on exit (success or abort)
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"; exit' INT TERM EXIT

# Number of workers: use (cores - 1) to avoid saturation, minimum 1
NUM_CORES=$(nproc)
JOBS=$((NUM_CORES > 1 ? NUM_CORES - 1 : 1))

echo "[INFO] Detected $NUM_CORES CPU cores → using $JOBS parallel workers"
echo "[INFO] Temporary workspace: $TEMP_DIR"

########################################
# STEP 1 — Write CSV header
########################################
echo "filename class_id x_center y_center width height confidence" > "$OUTPUT_CSV"


########################################
# STEP 2 — Build master file list
########################################
FILELIST="$TEMP_DIR/all_files.txt"
> "$FILELIST"

for DIR in "${INPUT_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "[INFO] Scanning directory: $DIR"
        # Append all label files found in this directory
        find "$DIR" -type f -name "*.txt"
    else
        echo "[WARN] Directory not found — skipping: $DIR"
    fi
done | sort > "$FILELIST"

TOTAL_FILES=$(wc -l < "$FILELIST")
echo "[INFO] Total label files discovered: $TOTAL_FILES"

# Abort if no files found
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "[ERROR] No .txt label files found — terminating."
    exit 1
fi


########################################
# STEP 3 — Archive file list (reproducibility)
########################################
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_FILELIST="$OUTPUT_DIR/filelist_${TIMESTAMP}.txt"

cp "$FILELIST" "$ARCHIVE_FILELIST"
echo "[INFO] Saved reproducible file list → $ARCHIVE_FILELIST"


########################################
# STEP 4 — Parallel workload processing
########################################

# Balanced static batching
BATCH_SIZE=$(( (TOTAL_FILES + JOBS - 1) / JOBS ))

echo "[INFO] Batch size: $BATCH_SIZE lines / worker"

for ((i = 0; i < JOBS; i++)); do
    START=$((i * BATCH_SIZE + 1))
    END=$((START + BATCH_SIZE - 1))
    TMP_OUT="$TEMP_DIR/job_$i.txt"

    # Background worker
    (
        # Read assigned portion of the file list
        sed -n "${START},${END}p" "$FILELIST" | while IFS= read -r file; do
            fname="${file##*/}"   # Efficient string extraction (faster than basename)

            # Print each annotation row prefixed with filename
            while IFS= read -r line; do
                [[ -n "$line" ]] && printf "%s %s\n" "$fname" "$line"
            done < "$file"

        done
    ) > "$TMP_OUT" &
done

wait  # wait for all parallel jobs

########################################
# STEP 5 — Append results to final CSV
########################################
cat "$TEMP_DIR"/job_*.txt >> "$OUTPUT_CSV"

echo "[SUCCESS] Output CSV created:"
echo "          $OUTPUT_CSV"
echo "[INFO] Filelist archived:"
echo "          $ARCHIVE_FILELIST"
echo "[DONE]"