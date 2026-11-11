# Benchmark data and exact query inventories

`manifest.jsonl` contains each input executed by the pinned upstream benchmark
drivers, including independent memberships when the same SQL occurs more than
once. Its `sha256` hashes the exact executable SQL string. `source_file_sha256`
hashes the original source bytes, and `source_commit` identifies the source Git
revision. Schemas point to the unchanged source files, with separate hashes.

| Source | TPC-H | DSB | Calcite | Total |
| --- | ---: | ---: | ---: | ---: |
| R-Bot | 44 | 76 | 44 | 164 |
| QUITE | 63 | 156 | 58 | 277 |
| Total memberships | 107 | 232 | 102 | 441 |

R-Bot releases 74 DSB SQL files. Two files (`query039_0.sql` and
`query039_1.sql`) each contain two statements. Its own `my_rewriter/test.py`
removes line comments and splits on semicolons, producing 76 executable inputs.
The importer follows that exact procedure and preserves the original full text,
file hash, statement index, and upstream executed query name. Calcite inputs use
`input_sql` and the first sorted rewrite test name, matching the same driver.
QUITE input text and list memberships are preserved without intersection-based
filtering. Parse failures are recorded, never dropped. QUITE's separate
StackOverflow family is outside this repository's requested three-family scope;
its 43 released entries and source path are explicitly recorded in the import
report.

Rebuild the inventory with `.venv/bin/python scripts/import_benchmarks.py`.
`import_report.json` records counts, diagnostics, duplicate memberships, commits,
and the complete manifest hash. Vendor repositories are pinned to:

* R-Bot: `https://github.com/curtis-sun/LLM4Rewrite`, commit
  `c9c90e5d7867888c3aaba86e4fc9e6d48f53b375`.
* QUITE: `https://github.com/Yuyang-Song/QUITE`, commit
  `d0dd6ee567adde144cd8a7024a657c9329231092`.

## Data generation and loading

Both source READMEs point to the following data constructors. Neither release
provides a database snapshot, a full physical configuration, or all details of
the authors' actual generated instances. Local instances therefore reproduce
the released target scales and the linked construction procedures; they are not
claimed to be byte-identical author snapshots.

| Family | Linked constructor | Pinned commit |
| --- | --- | --- |
| TPC-H | [gregrahn/tpch-kit](https://github.com/gregrahn/tpch-kit) | `852ad0a5ee31ebefeed884cea4188781dd9613a3` |
| DSB | [microsoft/dsb](https://github.com/microsoft/dsb) | `ec9a156ceee923db1114cbe388f6183b53d49787` |
| Calcite | [eidos06/SlabCity](https://github.com/eidos06/SlabCity) | `d57a516d7249869c0c61e44f88795e1ce952e31d` |

`scripts/setup_datasets.py` builds the linked C generators locally. TPC-H uses
the generator's deterministic default random streams. DSB uses `-rngseed 0` and
generates all tables together in one invocation, preserving the correlations
required by its README. Schema DDL comes from R-Bot without added physical
indexes. The DSB generator's TPC-DS qualification/publication messages refer to
its inherited scale rules; these instances are DSB at the R-Bot scales and are
not TPC-DS compliant benchmark results, consistent with DSB's own disclaimer.
The original `scaling.c` scale interpolation is preserved as well: `LogScale`
adds the SF1 base even between later scale anchors. Consequently some SF50
dimensions can be smaller than SF10 (for example, `call_center`: 8 versus 24
rows), while linear fact tables grow with the requested scale. SF50 means the
literal original `dsdgen -scale 50` invocation; reports record the actual table
counts rather than assuming every table is five times its SF10 size.

Published DSB data uses the unchanged pinned source and its default Linux
compiler flags. A separate `-O2` qualification exposed a buffer-length error
in `joint_distribution.c`; the isolated one-line repair and compiler build
were rejected because SF1's first `item` row changed two fields. All other
23 benchmark tables matched, but partial equality does not qualify a generator.
`dsb_repaired_build_validation.json` records every table hash, the exact patch,
compiler and executable hashes, and the rejection. `dbgen_version` timestamps
and command paths are reported separately from the 24 benchmark data tables.

Calcite uses the released SlabCity source with its seed `2333`, NumPy `1.26.3`,
and Faker `22.2.0`. Install these versions before generation. The source accepts
250K, 1M, and 4M size switches and holds all rows in Python lists. Our wrapper
supplies an explicit size, streams rows to CSV, uses lazy integer ranges, and
applies the nullable20 subset predicate while streaming. At 10K, both uniform
and Zipf outputs match all six original-generator CSVs byte for byte; see
`calcite_stream_validation.json`. The source Zipf transformation (exponent1.25)
is preserved, including its different treatment of employee names versus bonus
names. At the 40M scale, 400K-domain Zipf sums overflow NumPy int64. The released
helper retries a negative sum using `random.seed()` with system time, despite
the top-level2333 seed. Our wrapper replaces only these unseeded retries with
the explicit deterministic seed sequence2334,2335,... and records every retry
in `generation.json`. This is a disclosed reproducibility repair; exact author
Zipf instances cannot be recovered from the release. The wrapper does not
silently substitute another synthetic distribution.

The 40M size parameter creates 40M rows **in each** of `EMP`, `EMP_B`,
`EMPNULLABLES`, and `BONUS`, plus 400K `DEPT` rows and a filtered
`EMPNULLABLES_20` subset. The aggregate database contains over160M rows. R-Bot
does not release its own 40M script or clarify the aggregate versus per-table
wording. This interpretation is recorded explicitly in the configurations and
per-instance provenance rather than claiming an unavailable snapshot.

Development instances are `tpch_dev` (SF0.01), `dsb_dev` (SF1, the official
generator's minimum), and `calcite_dev` (size10K, uniform). They are for
integration and finite-data checks. Published-scale instances are `tpch10`,
`tpch50`, `dsb10`, `dsb50`, `calcite10`, and `calcite10zipf`.

```bash
.venv/bin/python scripts/setup_datasets.py --write-configs
.venv/bin/python scripts/setup_datasets.py --profile development
.venv/bin/python scripts/setup_datasets.py --profile published --generate-only
.venv/bin/python scripts/setup_datasets.py --profile published --load-only
```

For independent generator processes, `--only TARGET` is repeatable;
`--load-when-ready` loads completed targets sequentially while waiting for the
remaining generation reports. This permits bounded generation concurrency
without concurrent PostgreSQL bulk loads. Stop all generation and loading
before final timing runs. `--replace` explicitly recreates the selected database.
`--resume` preserves the committed tables of an incomplete task-owned database:
it validates generator provenance, the pinned schema against PostgreSQL's actual
catalog, each completed source file's SHA256, and its exact committed row count.
Same-transaction server COPY/source-count receipts are authoritative for these
task-owned tables and the dedicated read-only benchmark role. Legacy file-only
checkpoints and `--full-count-verify` use independent `SELECT count(*)` scans.
Explicit full-count audits also cover already-ready databases in a repeatable
read snapshot and save a separate `.runtime/datasets/NAME.audit.json` receipt.
A mismatch stops loading. Tables without a completed checkpoint log their
previous row count as unknown and restart with a transactional TRUNCATE/COPY;
scanning their discarded storage provides no useful validation. New COPY
checkpoints commit in the same transaction as the table data; atomic file
reports mirror those server records. Completed databases are kept without
reloading.
The waiting loader prioritizes incomplete database checkpoints on restart and
retries connection interruptions while PostgreSQL recovers.
COPY flushes libpq output in 128KiB blocks so a slow server applies bounded
client backpressure rather than accumulating hundreds of megabytes in Python.
For DSB source files of at least 100MB, the loader copies the empty table and
rebuilds its original primary key in one transaction, using 256MB of temporary
maintenance memory. It commits the data and checkpoint only after the exact
original constraint/index is restored. Interruptions roll back all three;
duplicate keys reject the whole table. Final catalog validation checks the
original columns, constraints, indexes, and table behavior before `ANALYZE`.
See `dataset_resume_validation.json` and `dataset_index_load_validation.json`
for interruption and corruption checks against PostgreSQL14.13.

Long-running local generators, the single loader, and attestation should run as
systemd user services so interactive session interruption cannot kill them.
The recovery run uses `mcts-gen-tpch`, `mcts-gen-dsb`,
`mcts-gen-calcite-uniform`, `mcts-gen-calcite-zipf`, `mcts-load-published`, and
`mcts-attest-datasets`. Inspect them with
`systemctl --user list-units 'mcts*' --all` and their `.runtime/` logs.
The attestation service restores and verifies `autovacuum=on` after all targets
complete loading and analysis.

Data files and detailed readiness reports live under ignored `.runtime/data/`
and `.runtime/datasets/`. A target is ready only after every source row is
copied, each committed table count agrees, required tables exist, and `ANALYZE`
completes. Reports include source file hashes and row counts, schema hash,
generator provenance, elapsed load time, and database size. A copy is stored in
the database's `_mcts_dataset_provenance` table. Read-only benchmark connections
use the `mcts_read` role. Configuration credentials are not embedded.
