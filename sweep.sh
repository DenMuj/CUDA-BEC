#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="real3d_out_times.txt"
> "$OUTPUT_FILE"

for N in $(seq 128 16 416); do
  # Prefer 'range-N/N{N}/imag3d-out.txt' if it exists, otherwise 'N{N}/imag3d-out.txt'
  OUT_FILE=""
  if [[ -f "real3d-N/N${N}/real3d-rms.txt" ]]; then
    OUT_FILE="real3d-N/N${N}/real3d-rms.txt"
  elif [[ -f "N${N}/real3d-rms.txt" ]]; then
    OUT_FILE="N${N}/real3d-rms.txt"
  fi

  CALC="NA"

  if [[ -n "$OUT_FILE" && -f "$OUT_FILE" ]]; then
    # Extract numeric for calculation line: "Calculation (iterations) wall-clock time : 2.877 seconds"
    CALC_LINE=$(grep -i "Total time on GPU" "$OUT_FILE" || true)
if [[ -n "$CALC_LINE" ]]; then
  CALC=$(echo "$CALC_LINE" \
         | grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' \
         | head -n1 || true)
  [[ -z "${CALC:-}" ]] && CALC="NA"
fi
  fi

  echo "$N $CALC" >> "$OUTPUT_FILE"
done

echo "Wrote $(($(wc -l < "$OUTPUT_FILE") - 1)) rows to $OUTPUT_FILE" 1>&2