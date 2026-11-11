#!/usr/bin/env bash
set -euo pipefail
MCTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MCTS_ROOT"
if [ ! -d vendor/pg_hint_plan/.git ]; then
  git clone --depth 1 --branch REL14_1_4_3 https://github.com/ossc-db/pg_hint_plan.git vendor/pg_hint_plan
fi
if [ "$(git -C vendor/pg_hint_plan rev-parse HEAD)" != 3b58a184a41602aa5e163562320d8979c35e6586 ]; then
  printf '%s\n' 'pg_hint_plan commit differs from upstream.lock.json; refusing an unpinned build.' >&2
  exit 1
fi
make -C vendor/pg_hint_plan PG_CONFIG="$MCTS_ROOT/.local/postgres/bin/pg_config" > .local/downloads/hint-build.log 2>&1
make -C vendor/pg_hint_plan PG_CONFIG="$MCTS_ROOT/.local/postgres/bin/pg_config" install > .local/downloads/hint-install.log 2>&1
.local/postgres/bin/psql -h "$MCTS_ROOT/.runtime/socket" -p 55432 -d postgres -v ON_ERROR_STOP=1 -c "ALTER SYSTEM SET session_preload_libraries='pg_hint_plan'" -c 'SELECT pg_reload_conf()'
.local/postgres/bin/psql -h "$MCTS_ROOT/.runtime/socket" -p 55432 -d postgres -v ON_ERROR_STOP=1 -c 'SHOW pg_hint_plan.enable_hint'
