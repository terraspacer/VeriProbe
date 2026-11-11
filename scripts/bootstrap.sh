#!/usr/bin/env bash
set -euo pipefail
MCTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MCTS_ROOT"
mkdir -p vendor .local/downloads .local/src .local/pgbuild .runtime/socket
if [ ! -d vendor/LLM4Rewrite/.git ]; then
  git clone https://github.com/curtis-sun/LLM4Rewrite.git vendor/LLM4Rewrite
  git -C vendor/LLM4Rewrite checkout c9c90e5d7867888c3aaba86e4fc9e6d48f53b375
fi
if [ ! -d vendor/QUITE/.git ]; then
  git clone https://github.com/Yuyang-Song/QUITE.git vendor/QUITE
  git -C vendor/QUITE checkout d0dd6ee567adde144cd8a7024a657c9329231092
fi
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.lock
if [ ! -x .local/postgres/bin/postgres ]; then
  curl -fL --retry 3 https://ftp.postgresql.org/pub/source/v14.13/postgresql-14.13.tar.bz2 -o .local/downloads/postgresql-14.13.tar.bz2
  .venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
expected = json.loads(Path('upstream.lock.json').read_text())['postgres']['sha256']
actual = hashlib.sha256(Path('.local/downloads/postgresql-14.13.tar.bz2').read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit('PostgreSQL source checksum does not match upstream.lock.json')
PY
  tar -xjf .local/downloads/postgresql-14.13.tar.bz2 -C .local/src
  (
    cd .local/pgbuild
    ../src/postgresql-14.13/configure --prefix="$MCTS_ROOT/.local/postgres" --without-readline --without-zlib > ../downloads/postgres-configure.log 2>&1
    make -j4 > ../downloads/postgres-build.log 2>&1
    make install > ../downloads/postgres-install.log 2>&1
  )
fi
if [ ! -e .runtime/pgdata/PG_VERSION ]; then
  .local/postgres/bin/initdb -D .runtime/pgdata --auth-local=trust --auth-host=scram-sha-256 --encoding=UTF8 --locale=C > .runtime/initdb.log
  cat >> .runtime/pgdata/postgresql.conf <<EOF
listen_addresses = ''
port = 55432
unix_socket_directories = '$MCTS_ROOT/.runtime/socket'
shared_buffers = '256MB'
work_mem = '8MB'
maintenance_work_mem = '128MB'
max_connections = 20
max_parallel_workers_per_gather = 0
jit = off
track_io_timing = on
log_min_duration_statement = 60000
EOF
fi
if ! .local/postgres/bin/pg_ctl -D .runtime/pgdata status >/dev/null 2>&1; then
  bash scripts/postgres.sh start
fi
.local/postgres/bin/psql -h "$MCTS_ROOT/.runtime/socket" -p 55432 -d postgres -v ON_ERROR_STOP=1 <<'SQL'
DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='mcts_read') THEN CREATE ROLE mcts_read LOGIN; END IF; END $$;
GRANT pg_read_all_data TO mcts_read;
-- The benchmark quietness gate must see maintenance backends owned by other roles.
GRANT pg_read_all_stats TO mcts_read;
ALTER ROLE mcts_read SET default_transaction_read_only=on;
ALTER ROLE mcts_read SET statement_timeout='300s';
SELECT 'CREATE DATABASE mcts_fixtures' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='mcts_fixtures')\gexec
SQL
.venv/bin/python scripts/import_benchmarks.py
