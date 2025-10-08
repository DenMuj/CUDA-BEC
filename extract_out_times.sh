#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="imag3d_out_times.txt"
> "$OUTPUT_FILE"

# Header (optional). Comment out if you want raw numbers only.
echo "N init_seconds calc_seconds total_seconds" >> "$OUTPUT_FILE"

for N in $(seq 128 8 512); do
  # Prefer 'range-N/N{N}/imag3d-out.txt' if it exists, otherwise 'N{N}/imag3d-out.txt'
  OUT_FILE=""
  if [[ -f "range-N/N${N}/imag3d-out.txt" ]]; then
    OUT_FILE="range-N/N${N}/imag3d-out.txt"
  elif [[ -f "N${N}/imag3d-out.txt" ]]; then
    OUT_FILE="N${N}/imag3d-out.txt"
  fi

  INIT="NA"
  CALC="NA"

  if [[ -n "$OUT_FILE" && -f "$OUT_FILE" ]]; then
    # Extract numeric preceding the word 'seconds' for init line
    INIT_LINE=$(grep -i "^Initialization/allocation wall-clock time" "$OUT_FILE" || true)
    if [[ -n "$INIT_LINE" ]]; then
      INIT=$(echo "$INIT_LINE" | grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' | tail -n1 || true)
      [[ -z "${INIT:-}" ]] && INIT="NA"
    fi

    # Extract numeric for calculation line
    CALC_LINE=$(grep -i "^Calculation \(iterations\) wall-clock time" "$OUT_FILE" || true)
    if [[ -n "$CALC_LINE" ]]; then
      CALC=$(echo "$CALC_LINE" | grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' | tail -n1 || true)
      [[ -z "${CALC:-}" ]] && CALC="NA"
    fi
  fi

  # Compute total when both present and numeric
  TOTAL="NA"
  if [[ "$INIT" != "NA" && "$CALC" != "NA" ]]; then
    TOTAL=$(python3 - <<EOF
init_val = float("$INIT")
calc_val = float("$CALC")
print(init_val + calc_val)
EOF
)
  fi

  echo "$N $INIT $CALC $TOTAL" >> "$OUTPUT_FILE"
done

echo "Wrote $(($(wc -l < "$OUTPUT_FILE") - 1)) rows to $OUTPUT_FILE" 1>&2

