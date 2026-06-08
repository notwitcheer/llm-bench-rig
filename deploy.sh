#!/bin/bash
REMOTE_USER="${BENCH_USER:-youruser}"
REMOTE_HOST="${BENCH_HOST:-your-server-ip}"

rsync -avz --exclude='results/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='venv' --exclude='.venv' \
  ./ "${REMOTE_USER}@${REMOTE_HOST}:~/benchmark-rig/"
echo "Deployed to ${REMOTE_HOST}:~/benchmark-rig/"
