#!/usr/bin/env bash
set -euo pipefail

# Change to script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="imag3d_gpu_times.txt"
> "$OUTPUT_FILE"

# Loop N from 128 to 512 step 8
for N in $(seq 128 8 512); do
  # Prefer 'range-N/N{N}/imag3d-rms.txt' if it exists, otherwise 'N{N}/imag3d-rms.txt'
  RMS_FILE=""
  if [[ -f "range-N/N${N}/imag3d-rms.txt" ]]; then
    RMS_FILE="range-N/N${N}/imag3d-rms.txt"
  elif [[ -f "N${N}/imag3d-rms.txt" ]]; then
    RMS_FILE="N${N}/imag3d-rms.txt"
  fi

  if [[ -n "$RMS_FILE" && -f "$RMS_FILE" ]]; then
    # Try to extract from lines that likely contain the GPU time info
    # Match variants: 'seconds', 'second', 's', or 'Total time ... GPU'
    SECONDS_VAL=$(grep -iE 'seconds|second|total[[:space:]]+time.*gpu|gpu.*time' "$RMS_FILE" | tail -n1 | grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' | tail -n1 || true)
    # Fallback: last numeric in the whole file
    if [[ -z "${SECONDS_VAL:-}" ]]; then
      SECONDS_VAL=$(grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$RMS_FILE" | tail -n1 || true)
    fi
    if [[ -n "${SECONDS_VAL:-}" ]]; then
      echo "$N $SECONDS_VAL" >> "$OUTPUT_FILE"
    else
      # If format not found, write NA to keep alignment
      echo "$N NA" >> "$OUTPUT_FILE"
    fi
  else
    # Missing file, record NA
    echo "$N NA" >> "$OUTPUT_FILE"
  fi
done

echo "Wrote $(wc -l < "$OUTPUT_FILE") lines to $OUTPUT_FILE"

