# VeriProbe

Supplemental source code, query manifests, frozen candidate proposals, small-database counterexample witness suites, raw PostgreSQL campaign and validation logs, and programmatic paper-generation scripts for **Verify, Probe, Fall Back: A Safe Selection Layer for Rule-Based and LLM-Proposed Query Rewrites**.

---

## Overview

Rule engines (`Apache Calcite`) and LLM-based query rewriters (`R-Bot`, `QUITE`) frequently propose candidate SQL statements that either return different rows on edge cases (`NULL`s, duplicates, empty relations) or run slower than the original statement because static optimizer cost estimates (`EXPLAIN`) misrank candidates.

**VeriProbe** places a generator-agnostic safe selection layer between candidate rewrite sources and PostgreSQL for recurring analytical statements:
1. **Unified Candidate Pool (`mcts_rewrite/adapters.py`, `mcts_rewrite/baselines.py`):** Accepts candidate rewrites from Apache Calcite rule sequences, replayed published rewrites of `R-Bot` and `QUITE`, frozen LLM macro proposals (`artifacts/model_*.jsonl`), and plan-steering hints (`pg_hint_plan`).
2. **Two-Tier Equivalence Gate (`mcts_rewrite/verify.py`, `mcts_rewrite/verification_db.py`, `mcts_rewrite/structural_empty_v2.py`):** Admits a candidate only if it passes either bounded formal SMT equivalence (`SQLSolver`) or structural empty-result proof (`proved`), or differential execution across six small counterexample database instances (`base`, `empty`, `null`, `duplicate`, `skew`, `unmatched`) under explicit row-ordering and floating-point contracts (`bounded`).
3. **Priority-Ordered Bounded Probing & Safe Fallback (`mcts_rewrite/fast_search.py`, `scripts/run_icde_paper_campaign.py`):** Ranks admitted candidates solely from planner estimates via $\pi(p) = (d(p), \mathbf{1}[\kappa(p) \ge \kappa(q)], b(p), \kappa(p), i(p))$ onto the prefix $V_M(q)$ ($M = 16$), probes at most two admitted candidates on the target PostgreSQL database under probe cutoff $c_{\mathrm{p}}(q) = \min(\ell_0(q), \bar{c})$ ($\bar{c} = 45\text{ s}$, $T_{\max} = 300\text{ s}$), times the fastest probe $p^\ast$ fully (1 warmup + 3 timed runs), and returns the original statement $q$ unless $\ell(p^\ast) < \ell_0(q)$.

---

## Repository Structure

| Path | Description |
| --- | --- |
| `mcts_rewrite/` | Core Python package implementing candidate pool adapters (`adapters.py`, `baselines.py`, `calcite_worker.py`), two-tier formal + differential equivalence verification (`verify.py`, `verification_db.py`, `structural_empty_v2.py`, `output_contract.py`), plan inspection (`plans.py`, `critic.py`), priority-ordered bounded probing & fallback (`fast_search.py`, `search.py`), and PostgreSQL measurement harnesses (`db.py`, `reporting.py`). *(Note: the Python package retains its initial module name `mcts_rewrite`; as documented in `paper/main.tex`, the evaluated `VeriProbe` campaign procedure uses deterministic priority ordering rather than Monte Carlo tree search.)* |
| `benchmarks/` | Canonical benchmark manifest (`manifest.jsonl`, `manifest.schema.json`), 30-statement development split (`stage_a_30.jsonl`), generator lockfile (`generators.lock.json`), and dataset/index validation records across `TPC-H`, `Calcite`, and `DSB`. |
| `artifacts/` | Frozen structured LLM macro candidates (`model_tpch.jsonl`, `model_calcite.jsonl`, `model_dsb.jsonl`, `model_candidates_stage_a_consolidated.jsonl`) and SHA-256 provenance manifest (`model_manifest.json`). |
| `prompts/` | Structured SQL rewrite action catalog (`action_patterns.jsonl`). |
| `configs/` | Database, miniature counterexample fixture, upstream commit lock (`configs/upstreams/`), and campaign configuration files (`published-hybrid.json`, `stage-a-sf10.json`, `validation_v2.json`). |
| `scripts/` | Environment bootstrap (`bootstrap.sh`, `bootstrap_upstreams.sh`, `postgres.sh`, `install_hints.sh`), dataset/witness builders (`setup_datasets.py`, `build_query_witness_fixtures_v2.py`, `materialize_query_witness_fixtures_v2.py`), campaign runners (`run_icde_paper_campaign.py`, `run_full_icde_suite.py`), equal-cutoff interleaved validation runner (`run_validation_session.py`), and cohort reporters (`report_icde_campaign.py`, `report_cohorts.py`). |
| `results/` | Raw JSONL campaign logs for all five evaluated cohorts (`icde_paper_campaign_v4`, `icde_full_suite_tpch10`, `icde_full_suite_calcite10`, `icde_calcite10zipf_stage`, `icde_sf50_cohort`), the completed 88-pair equal-cutoff (`300 s`) interleaved validation session (`icde_validation_session`), session-drift audit logs (`stage_a_sf10_v2_clean_r2`, `stage_a_sf10_local_resume`), and query-specific small-database witness SQL plans (`stage_a_sf10_v2_witnesses/`, `stage_a_followup/`). |
| `paper/` | LaTeX manuscript (`main.tex`, `main.pdf`), TikZ schematics (`figures/*.tex`), matplotlib plot generator (`scripts/make_figures.py`, `figures/plots/*.pdf`), programmatic macro/table/claim-check compiler (`scripts/build_data.py`, `generated/`), and one-command build script (`build.sh`). |
| `tests/` | Offline unit and contract test suite (33 test modules) covering SQL parsing/read-only safety, output contracts, witness fixture generation, candidate deduplication, selection/fallback logic, and cohort reporting. |

---

## Quick Start: Reproducing All Paper Numbers, Tables, Plots, and Claim Checks

Every numerical macro (`1,272` macros in `paper/generated/numbers.tex`), LaTeX table (`paper/generated/tables.tex`), plot dataset (`paper/generated/*.dat`), audit summary (`paper/generated/audit.json`, `paper/generated/validation.json`), and comparative claim assertion (`paper/generated/claim_checks.txt`) is computed directly from the raw JSONL logs in `results/` without requiring a live PostgreSQL instance.

### 1. Python Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[test]"
.venv/bin/pip install matplotlib
```

### 2. Recompute All Paper Macros, Tables, Plots, and Claim Checks from Raw Logs

```bash
.venv/bin/python paper/scripts/build_data.py
.venv/bin/python paper/scripts/make_figures.py
```

If `tectonic` and `poppler-utils` (`pdfinfo`) are installed, run the full end-to-end paper build and verification pipeline:

```bash
bash paper/build.sh
```

`paper/build.sh` regenerates all macros and tables from `results/*/raw.jsonl`, redraws all vector PDF plots in `paper/figures/plots/`, compiles `paper/main.pdf`, verifies the 12-page content limit, and checks all 17 automated comparative-claim assertions in `paper/generated/claim_checks.txt`.

---

## Mapping Paper Sections, Tables, and Figures to Artifacts

### 1. Application Scenario, Running Example & Planner Misranking (Sections I–II, Figs. 1–3)
- **Running example record (`rbot:dsb:query081_0`):** `results/icde_paper_campaign_v4/raw.jsonl`
- **Structured decorrelation macro (`decorrelate_return_average_by_group`):** `artifacts/model_dsb.jsonl`
- **Extracted macros (`\ExamplePool`, `\ExampleAccepted`, `\ExampleRuleCost`, `\ExampleMacroCost`, `\ExampleOurs`):** `paper/scripts/build_data.py`

### 2. Candidate Pool, Equivalence Gate & Budgeted Selection with Fallback (Sections III–VI, Figs. 4–8)
- **Rule, replay, LLM macro & planner-hint adapters:** `mcts_rewrite/adapters.py`, `mcts_rewrite/baselines.py`, `mcts_rewrite/calcite_worker.py`
- **Two-tier equivalence gate (`SQLSolver` + structural empty proof + 6 small counterexample databases):** `mcts_rewrite/verify.py`, `mcts_rewrite/structural_empty_v2.py`, `mcts_rewrite/verification_db.py`, `mcts_rewrite/output_contract.py`
- **Query-specific counterexample witness SQL suites:** `results/stage_a_sf10_v2_witnesses/plans/` and `results/stage_a_sf10_v2_witnesses/postgresql_validated_bindings.json`
- **Priority-ordered selection ($M = 16$, at most 2 probes, $c_{\mathrm{p}}(q) = \min(\ell_0(q), \bar{c})$, safe fallback):** `mcts_rewrite/fast_search.py`, `scripts/run_icde_paper_campaign.py`

### 3. Five Evaluation Cohorts — 161 Distinct Statements, 211 Pairs (Section VII, Tables I–II, Figs. 9–14, 18)
- **`Dev` (`30` statements across `TPC-H SF 10`, `DSB SF 10`, `Calcite uniform`):** `results/icde_paper_campaign_v4/raw.jsonl`
- **`TPC-H` held-out (`97` statements, `TPC-H SF 10`):** `results/icde_full_suite_tpch10/raw.jsonl`
- **`Calcite` held-out (`34` statements, `Calcite uniform`):** `results/icde_full_suite_calcite10/raw.jsonl`
- **`Zipf` (`30` statements, skewed `Calcite`):** `results/icde_calcite10zipf_stage/raw.jsonl`
- **`SF 50` (`20` statements, `TPC-H` and `DSB` at `SF 50`):** `results/icde_sf50_cohort/raw.jsonl`
- **Cross-session latency drift check (`rbot:tpch:query12_0`):** `results/stage_a_sf10_v2_clean_r2/raw.jsonl`

### 4. Component Ablations, Candidate Sources, Prefix Budget $M$ & Break-Even Analysis (Section VIII, Table III, Figs. 15–17)
- **Six recorded ablation variants (`full_mcts`, `class_a_symbolic_only`, `class_b_macro_only`, `without_plan_bottleneck`, `without_critic`, `without_canary_safety_gate`) and prefix sweep $M \in \{1, 2, 4, 8, 16\}$:** embedded per statement in `results/*/raw.jsonl` (`variants` field) and aggregated by `paper/scripts/build_data.py` into `paper/generated/tables.tex`, `sources.dat`, `budget.dat`, and `breakeven.dat`.

### 5. Equal-Cutoff (`300 s`) Interleaved Validation Session (Section VIII, Table IV, Fig. 19)
- **All 88 non-identical `(statement, database)` pairs re-timed back-to-back in one PostgreSQL session (`3` rotating interleaved rounds, uniform `300 s` cutoff):** `results/icde_validation_session/raw.jsonl`
- **Validation session runner:** `scripts/run_validation_session.py`
- **Machine-readable validation summary:** `paper/generated/validation.json`

---

## Re-Running Upstream Bootstrap, Offline Tests, and PostgreSQL Campaigns

```bash
# 1. Clone pinned upstream repositories (LLM4Rewrite and QUITE) for full test suite parity
bash scripts/bootstrap_upstreams.sh
.venv/bin/pytest

# 2. Build local PostgreSQL 14.13, pg_hint_plan, and Calcite/SQLSolver workers
bash scripts/bootstrap.sh
bash scripts/install_hints.sh

# 3. Populate benchmark databases and small counterexample witness fixtures
.venv/bin/python scripts/setup_datasets.py --config configs/published-hybrid.json
.venv/bin/python scripts/setup_miniatures.py --config configs/miniatures-join-aware.json

# 4. Run campaign cohorts or the equal-cutoff interleaved validation session
.venv/bin/python scripts/run_full_icde_suite.py --config configs/published-hybrid.json
.venv/bin/python scripts/run_validation_session.py
```

Note: adjust socket/DSN paths in `configs/*.json` (default `/workspace/VeriProbe/.runtime/socket`) to match your local checkout path before launching live PostgreSQL sessions.

---

## License

Released under the [MIT License](LICENSE).
