#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_wandb_agent.sh SWEEP_ID [--count N] [--gpu GPU]"
}

sweep_id=""
count=1
gpu=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--count)
      count="$2"
      shift 2
      ;;
    -g|--gpu)
      gpu="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$sweep_id" ]]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      sweep_id="$1"
      shift
      ;;
  esac
done

if [[ -z "$sweep_id" ]]; then
  usage >&2
  exit 1
fi

if [[ "$gpu" -ge 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

uv run wandb agent --count "$count" "$sweep_id"
