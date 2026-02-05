#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Usage:
#   bash run_paddlex_vllm.sh
#
# Optional env overrides:
#   GPU=0
#   PORT=8956
#   NAME=paddlex-vllm
#   IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest
#   WORKDIR=$HOME/paddlex-home
#
# Examples:
#   GPU=0 PORT=8957 bash run_paddlex_vllm.sh
#   WORKDIR=/data/paddlex-home GPU=1 bash run_paddlex_vllm.sh
# ==============================

IMAGE="${IMAGE:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:latest}"
NAME="${NAME:-paddlex-vllm}"

# Host work directory (mounted to container HOME)
WORKDIR="${WORKDIR:-$HOME/paddlex-home}"

# GPU selection: default "all" (more general). set GPU=0 or GPU=0,1 etc.
GPU="${GPU:-all}"

# Host port -> container port (container listens on 8080 in your script)
PORT="${PORT:-8956}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"

# Resources
SHM_SIZE="${SHM_SIZE:-16g}"
PIDS_LIMIT="${PIDS_LIMIT:-4096}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

mkdir -p "${WORKDIR}"/{.cache,models,downloads,tmp}

UID_NUM="$(id -u)"
GID_NUM="$(id -g)"
USER_NAME="$(id -un 2>/dev/null || echo user)"

# Put caches inside the mounted volume to avoid permission / user mismatch issues
XDG_CACHE="/home/app/.cache"
HF_CACHE="${XDG_CACHE}/huggingface"
TORCHINDUCTOR_CACHE="${XDG_CACHE}/torchinductor"

# GPU argument
if [[ "${GPU}" == "all" ]]; then
  GPU_ARGS=(--gpus all)
else
  GPU_ARGS=(--gpus "device=${GPU}")
fi

# If user doesn't have sudo, they can run directly with docker (assuming permissions)
DOCKER_BIN="${DOCKER_BIN:-docker}"
if command -v sudo >/dev/null 2>&1; then
  # If docker requires sudo on this machine, let it work out of the box
  if ! ${DOCKER_BIN} ps >/dev/null 2>&1; then
    DOCKER_BIN="sudo ${DOCKER_BIN}"
  fi
fi

echo "==> Launching container"
echo "    IMAGE=${IMAGE}"
echo "    NAME=${NAME}"
echo "    WORKDIR=${WORKDIR} -> /home/app"
echo "    GPU=${GPU}"
echo "    PORT=${PORT}:${CONTAINER_PORT}"

# Run
${DOCKER_BIN} run --rm --init \
  "${GPU_ARGS[@]}" \
  --name "${NAME}" \
  --user "${UID_NUM}:${GID_NUM}" \
  -e HOME=/home/app \
  -e USER="${USER_NAME}" \
  -e XDG_CACHE_HOME="${XDG_CACHE}" \
  -e HF_HOME="${HF_CACHE}" \
  -e TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE}" \
  -w /home/app \
  --ipc=host --shm-size="${SHM_SIZE}" \
  --pids-limit="${PIDS_LIMIT}" \
  --stop-signal=SIGTERM --stop-timeout="${STOP_TIMEOUT}" \
  -v "${WORKDIR}":/home/app \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -p "${PORT}:${CONTAINER_PORT}" \
  "${IMAGE}"
