#!/usr/bin/env bash
# Install independent upstream environments without mutating pinned sources.
set -euo pipefail
task_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$task_root"
export PATH="$task_root/.local/postgres/bin:$PATH"
export LD_LIBRARY_PATH="$task_root/.local/postgres/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p configs/upstreams results/adapter_smoke
python3.12 -m venv .venv-quite
python3.12 -m venv .venv-rbot
if [[ -f configs/upstreams/quite.lock ]]; then
  .venv-quite/bin/python -m pip install -r configs/upstreams/quite.lock
else
  .venv-quite/bin/python -m pip install -r vendor/QUITE/requirements.txt
fi
if [[ -f configs/upstreams/rbot.lock ]]; then
  .venv-rbot/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r configs/upstreams/rbot.lock
else
  # Preinstall the CPU build, otherwise sentence-transformers selects a CUDA
  # torch distribution and downloads unrelated GPU runtimes.
  .venv-rbot/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  # The pinned upstream imports sqlglot.optimizer.simplify.NONDETERMINISTIC,
  # which newer sqlglot releases removed.
  .venv-rbot/bin/python -m pip install -r vendor/LLM4Rewrite/requirements.txt 'sqlglot==27.29.0' 'llama-index==0.14.25' 'llama-index-embeddings-huggingface==0.8.0' 'llama-index-llms-openai-like<0.8'
fi
.venv-quite/bin/python -m pip freeze > configs/upstreams/quite.lock
.venv-rbot/bin/python -m pip freeze > configs/upstreams/rbot.lock
.venv-rbot/bin/python scripts/prepare_rbot_runtime.py --build-index
