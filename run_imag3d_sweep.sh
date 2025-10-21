#!/usr/bin/env bash
set -euo pipefail

# Change to the script's directory (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_FILE="$SCRIPT_DIR/input/imag3d-input"
BIN="$SCRIPT_DIR/imag3d-cuda"

if [[ ! -x "$BIN" ]]; then
  echo "Error: '$BIN' not found or not executable." >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: input file '$INPUT_FILE' not found." >&2
  exit 1
fi

# Backup original input file and ensure it gets restored on exit
BACKUP_FILE="$INPUT_FILE.bak.$(date +%s)"
cp -- "$INPUT_FILE" "$BACKUP_FILE"
restore_input() {
  if [[ -f "$BACKUP_FILE" ]]; then
    mv -f -- "$BACKUP_FILE" "$INPUT_FILE"
  fi
}
trap restore_input EXIT

# Iterate N=184..384 step 8
for N in $(seq 128 8 432); do
  echo "=== Running for N=$N (NX=NY=NZ) ==="

  # Update NX, NY, NZ lines in-place while preserving leading spaces and format
  # Matches lines like: '   NX = 184' and replaces the value only
  sed -E -i "s/^(\t*|[[:space:]]*)NX = .*/\1NX = ${N}/" "$INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NY = .*/\1NY = ${N}/" "$INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NZ = .*/\1NZ = ${N}/" "$INPUT_FILE"

  # Run the simulation
  "$BIN" -i "input/imag3d-input"

  # Prepare destination directory and copy outputs
  DEST_DIR="$SCRIPT_DIR/N${N}"
  mkdir -p -- "$DEST_DIR"

  # Copy the output text files (overwrite if exist)
  if [[ -f "$SCRIPT_DIR/imag3d-rms.txt" ]]; then
    cp -f -- "$SCRIPT_DIR/imag3d-rms.txt" "$DEST_DIR/"
  else
    echo "Warning: imag3d-rms.txt not found after run for N=$N" >&2
  fi

  if [[ -f "$SCRIPT_DIR/imag3d-mu.txt" ]]; then
    cp -f -- "$SCRIPT_DIR/imag3d-mu.txt" "$DEST_DIR/"
  else
    echo "Warning: imag3d-mu.txt not found after run for N=$N" >&2
  fi
done

echo "All runs completed. Restoring original input file."

