#!/usr/bin/env python3
"""Run a selected functional/tuning batch with active session workers.

Repeat --dataset and --query-id; select --budgets / --ablations as needed.
--config, --output, --session-id and --purpose are required. Native generation
runs once per exact dataset/SQL/schema/system before selected hybrid tasks.
--skip-native reuses a --seed-catalog, or preserves unavailable baselines without
a seed. --skip-hybrid builds only the native catalog. The supervisor supplies
fresh GPT-6.1-sol xhigh workers; this command establishes no final performance
claims. Shared launcher helpers own all ephemeral secret and bridge handling.
"""
from run_session_probe import main


if __name__ == "__main__":
    raise SystemExit(main(batch=True))
