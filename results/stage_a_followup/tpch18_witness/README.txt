Targeted follow-up witness: rbot:tpch:query18_0

This directory is an isolated proposal and finite validation record. It changes
no active Stage A code, configuration, prebuilt fixture, model request, service,
or benchmark dataset. It makes no benchmark timing claim. Only tpch_dev was
connected for validation; the new schema was removed by transaction rollback.

Observed rejection
------------------
The frozen checkpoint results/stage_a_sf10/queries/04_2531e71bc6082d32.json has
two QUITE candidates at selection.trace indexes 2 and 3, rejected because all
miniature outputs were empty. Both compare as 0 original / 0 candidate rows on
tpch_dev and each of the five existing miniature cases. The selected QUITE arm
fell back to the original. provenance.json records these facts and exact SQL
hashes; original.sql and quite_trace_2.sql/quite_trace_3.sql are checkpoint copies.

Both proposals preaggregate SUM(l_quantity) per l_orderkey with HAVING >313,
join those totals to orders/customer, and apply SUM again over the outer groups.
The latter SUM matters because source and frozen live catalogs have no unique
or primary-key constraints on customer, orders or lineitem: duplicate matching
parent rows must preserve the original join multiplicity.

Why existing fixtures miss query18_0
-----------------------------------
The active join-aware bundle dedicates TPCH lanes/probes only to query17_0 and
query20_0. The global literal sampler does not make enough coordinated lines
per order to satisfy query18_0's aggregate threshold. Its HAVING literal 313 is
attached to SUM(l_quantity), not a direct column comparison that seeds a
qualifying quantity/multiplicity recipe.

A pure in-memory reconstruction using the current generator, source schema and
manifest found these maxima for SUM(l_quantity) by l_orderkey:
  empty: no lineitems
  null: 111
  duplicate: 124
  skew: 204
  unmatched: 111
No case crosses the strict threshold of 313, so no order can qualify even before
the outer joins. Existing miniature schemas were neither queried nor modified.
coverage_diagnosis.json stores the reconstructed totals and source hashes.

Concrete data recipe
--------------------
schema.sql contains exact source CREATE TABLE statements for the three queried
relations. All original NOT NULL fields are preserved, and data.sql explicitly
fills them. No illegal NULL fixture values or invented key constraints are used.

A compact nine-row aggregation core is one customer, one order and seven lines
with quantities 50,50,50,50,50,50,14, totaling 314. The quantities stay at most 50,
and no individual line satisfies >313. The executed nine_row_core scenario
isolates exactly these rows from data.sql and returns one output row with 314.

The expanded 57-row base contains two customers, six orders and 49 lineitems.
Seven lines per order total 312, 313, 313.01, 314 or 315 as appropriate. Order 106
has qualifying lines but no matching customer; order key 999 has qualifying
lines but no order. The expected ordered result is:
  order 102: price 900, date 2001-02-02, quantity 315
  order 101: price 900, date 2001-03-03, quantity 314
  order 105: price 800, date 2001-01-01, quantity 313.01

validate.py applies 17 independent small scenarios to this same base using a
savepoint and rollback per case. They cover:
  - the nine-row nonempty core; below/equal/just-above the strict threshold;
  - exact duplicate lineitems, including a duplicate that crosses the threshold;
  - duplicate customer and order rows separately and together, preserving the
    outer sum's multiplicative effects (e.g. order101 becomes 628 or 1256);
  - missing customer/orders joins and unmatched lineitem order keys;
  - empty output when no aggregate qualifies;
  - price descending and date ascending ordering;
  - 101 additional qualifying orders, each with seven lines, returning exactly
    the top 100 and excluding the 101st. The largest scenario has 865 total rows.

The ordering checks use distinct complete ordering keys, including equal prices
with different dates. The original ORDER BY does not specify a unique order for
rows tied on both price and date; this witness makes no assertion about those
unspecified ties or their LIMIT selection.

Validation procedure and result
-------------------------------
The script creates one fresh followup_tpch18_* schema in a tpch_dev transaction,
sets search_path to that schema and pg_catalog only, loads the recipe, checks
original results against explicit expectations, and compares both saved
candidates' ordered rows. All 17 original expectations passed and both saved
candidates matched in every case. validation.json records returned original
rows, checksums and comparison results. This is nonvacuous finite evidence,
not a universal equivalence proof or a retroactive change to Stage A decisions.

Validation began 2026-10-27T05:42:55.132954+00:00 and ended
2026-10-27T05:42:55.295650+00:00. The window is recorded to expose overlap with
active benchmark work. The new schema was absent after rollback. Every statement
had statement_timeout=2s and lock_timeout=1s. No EXPLAIN ANALYZE, target-dataset
queries, benchmark execution or model generation occurred.

The reproducible command is:
  .venv/bin/python results/stage_a_followup/tpch18_witness/validate.py

Narrow correction after Stage A
-------------------------------
In a new, separately versioned fixture bundle, add an explicit query18_0 data
lane that coordinates customer/orders keys and at least one order whose several
lineitems sum above 313. Keep adjacent below/equal/above threshold cases, exact
duplicates and parent multiplicity, and probe the exact query's nonempty output.
Merely seeding literal 313 as a line quantity does not test aggregation well.
Retain the original five-case policy and full source-table coverage when building
a production bundle: this three-table diagnostic is deliberately not represented
as a full coverage-complete fixture. Reuse only the frozen proposals for any
labeled follow-up verification; do not introduce another generation arm or amend
the active Stage A configuration/results.
