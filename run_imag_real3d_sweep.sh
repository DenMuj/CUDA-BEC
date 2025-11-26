#!/usr/bin/env bash
set -euo pipefail

# Change to the script's directory (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOGFILE="$SCRIPT_DIR/run_imag_real3d.log"
exec >"$LOGFILE" 2>&1

IMAG_INPUT_FILE="$SCRIPT_DIR/input/imag3d-input"
REAL_INPUT_FILE="$SCRIPT_DIR/input/real3d-input"
IMAG_BIN="$SCRIPT_DIR/imag3d-cuda"
REAL_BIN="$SCRIPT_DIR/real3d-cuda"

# Check binaries exist
if [[ ! -x "$IMAG_BIN" ]]; then
  echo "Error: '$IMAG_BIN' not found or not executable." >&2
  exit 1
fi

if [[ ! -x "$REAL_BIN" ]]; then
  echo "Error: '$REAL_BIN' not found or not executable." >&2
  exit 1
fi

# Check input files exist
if [[ ! -f "$IMAG_INPUT_FILE" ]]; then
  echo "Error: input file '$IMAG_INPUT_FILE' not found." >&2
  exit 1
fi

if [[ ! -f "$REAL_INPUT_FILE" ]]; then
  echo "Error: input file '$REAL_INPUT_FILE' not found." >&2
  exit 1
fi

# Backup original input files and ensure they get restored on exit
IMAG_BACKUP_FILE="$IMAG_INPUT_FILE.bak.$(date +%s)"
REAL_BACKUP_FILE="$REAL_INPUT_FILE.bak.$(date +%s)"
cp -- "$IMAG_INPUT_FILE" "$IMAG_BACKUP_FILE"
cp -- "$REAL_INPUT_FILE" "$REAL_BACKUP_FILE"

restore_input() {
  if [[ -f "$IMAG_BACKUP_FILE" ]]; then
    mv -f -- "$IMAG_BACKUP_FILE" "$IMAG_INPUT_FILE"
  fi
  if [[ -f "$REAL_BACKUP_FILE" ]]; then
    mv -f -- "$REAL_BACKUP_FILE" "$REAL_INPUT_FILE"
  fi
}
trap restore_input EXIT

# Iterate N=128..416 step 16
for N in $(seq 128 16 416); do
  echo "=== Running for N=$N (NX=NY=NZ) ==="

  # Update NX, NY, NZ lines in imag3d-input
  sed -E -i "s/^(\t*|[[:space:]]*)NX = .*/\1NX = ${N}/" "$IMAG_INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NY = .*/\1NY = ${N}/" "$IMAG_INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NZ = .*/\1NZ = ${N}/" "$IMAG_INPUT_FILE"

  # Update NX, NY, NZ lines in real3d-input
  sed -E -i "s/^(\t*|[[:space:]]*)NX = .*/\1NX = ${N}/" "$REAL_INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NY = .*/\1NY = ${N}/" "$REAL_INPUT_FILE"
  sed -E -i "s/^(\t*|[[:space:]]*)NZ = .*/\1NZ = ${N}/" "$REAL_INPUT_FILE"

  # Run imaginary time simulation first (ground state)
  echo "  Running imag3d-cuda..."
  "$IMAG_BIN" -i "input/imag3d-input"

  # Run real time simulation (uses the finalpsi from imag3d)
  echo "  Running real3d-cuda..."
  "$REAL_BIN" -i "input/real3d-input"

  # Prepare destination directory and copy outputs
  DEST_DIR="$SCRIPT_DIR/N${N}"
  mkdir -p -- "$DEST_DIR"

  # Copy imag3d output files
  if [[ -f "$SCRIPT_DIR/imag3d-rms.txt" ]]; then
    cp -f -- "$SCRIPT_DIR/imag3d-rms.txt" "$DEST_DIR/"
  else
    echo "Warning: imag3d-rms.txt not found after run for N=$N" >&2
  fi

  #if [[ -f "$SCRIPT_DIR/imag3d-finalpsi.bin" ]]; then
  #  cp -f -- "$SCRIPT_DIR/imag3d-finalpsi.bin" "$DEST_DIR/"
  #fi

  # Copy real3d output files
  if [[ -f "$SCRIPT_DIR/real3d-rms.txt" ]]; then
    cp -f -- "$SCRIPT_DIR/real3d-rms.txt" "$DEST_DIR/"
  else
    echo "Warning: real3d-rms.txt not found after run for N=$N" >&2
  fi

done

echo "All runs completed. Restoring original input files."

