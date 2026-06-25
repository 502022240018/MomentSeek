#!/usr/bin/env bash
set -euo pipefail

device="${NPU_DEVICE_ID:-7}"
echo "Checking NPU ${device}; no inference process should remain after a completed index stage."
npu-smi info | sed -n "/| ${device} /,+3p"

