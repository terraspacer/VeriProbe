#!/usr/bin/env bash
set -euo pipefail
MCTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MCTS_ROOT"
case "${1:-status}" in
  start)
    if .local/postgres/bin/pg_ctl -D .runtime/pgdata status >/dev/null 2>&1; then
      .local/postgres/bin/pg_ctl -D .runtime/pgdata status
    elif command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
      if [ "$(systemctl --user show mcts-rewrite-postgres.service -p LoadState --value)" = loaded ]; then
        systemctl --user start mcts-rewrite-postgres.service
      else
        systemd-run --user --unit=mcts-rewrite-postgres --description='MCTS-Rewrite local PostgreSQL benchmark server' \
          --property="WorkingDirectory=$MCTS_ROOT" \
          --property="StandardOutput=append:$MCTS_ROOT/.runtime/postgres-service.log" \
          --property="StandardError=append:$MCTS_ROOT/.runtime/postgres-service.log" \
          --property=KillSignal=SIGINT --property=TimeoutStopSec=1800 \
          --property="MemoryHigh=${MCTS_PG_MEMORY_HIGH:-4G}" \
          --property="MemoryMax=${MCTS_PG_MEMORY_MAX:-6G}" \
          --property="MemorySwapMax=${MCTS_PG_SWAP_MAX:-256M}" \
          --property=ManagedOOMPreference=avoid \
          "$MCTS_ROOT/.local/postgres/bin/postgres" -D "$MCTS_ROOT/.runtime/pgdata"
      fi
      for mcts_attempt in {1..180}; do
        if .local/postgres/bin/pg_isready -q -h "$MCTS_ROOT/.runtime/socket" -p 55432 -d postgres; then break; fi
        sleep 1
      done
      .local/postgres/bin/pg_isready -h "$MCTS_ROOT/.runtime/socket" -p 55432 -d postgres
    else
      .local/postgres/bin/pg_ctl -D .runtime/pgdata -l .runtime/postgres.log start
    fi ;;
  stop)
    if systemctl --user is-active --quiet mcts-rewrite-postgres.service; then
      systemctl --user stop mcts-rewrite-postgres.service
    else
      .local/postgres/bin/pg_ctl -D .runtime/pgdata -m fast stop
    fi ;;
  status) .local/postgres/bin/pg_ctl -D .runtime/pgdata status ;;
  psql) shift; exec .local/postgres/bin/psql -h "$MCTS_ROOT/.runtime/socket" -p 55432 "$@" ;;
  *) printf 'Usage: %s {start|stop|status|psql [psql args]}\n' "$0" >&2; exit 2 ;;
esac
