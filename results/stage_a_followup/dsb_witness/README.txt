Targeted follow-up witness: rbot:dsb:query010_0

This directory is an isolated proposal and finite validation record. It does
not alter the active Stage A run, source code, configuration, prebuilt fixtures,
model requests, services, or benchmark datasets. It makes no benchmark timing
claim. Only dsb_dev was connected during validation.

What was observed
-----------------
The frozen checkpoint results/stage_a_sf10/queries/02_7b6df850c6c2d99f.json
contains three QUITE proposals at selection.trace indexes 2, 3 and 4 rejected
with "all miniature query outputs were empty; finite evidence is vacuous".
Each proposal compared as 0 original / 0 candidate rows on dsb_dev and each of
the empty/null/duplicate/skew/unmatched miniature cases. The selected QUITE SQL
therefore fell back to the original. coverage_diagnosis.json saves the relevant
small checkpoint facts and active fixture inference evidence.

Why the current miniature bundle misses this query
-------------------------------------------------
1. scripts/build_join_aware_fixtures.py PROBES includes only query032_0 for DSB.
   The active DSB cases contain profiles global_literals and query032_0. There
   is no query010_0 profile or coverage probe, so its customer restrictions,
   qualifying dates/items and all three price ratios are not coordinated.
2. More decisively, cross-query equality inference unions ca_county with
   store.s_county from query054_0/query054_1 and QUITE entries 79-81. make_rows
   assigns join-group values before profile/literal choices. Text join values
   become numeric strings such as '1', '4', etc., rather than any of the five
   allowed counties. This makes query010_0's county predicate false throughout
   every nonempty active miniature. This diagnosis uses saved inference and a
   pure in-memory reconstruction of the generator; active fixture data was not
   modified or queried.
3. Other independent blockers are visible in the reconstruction: e.g. initial
   date rows pair (year,month) as (1999,4), (2002,9), (2001,12); initial customer
   rows with birth months 8,1,7 share demographic key 1, whose education is
   '2 yr Degree'; sales/list-price pools do not enforce ratios .77 through .87.
   The county blocker alone establishes emptiness; these examples explain why
   fixing only county text still does not establish adequate coverage.

Concrete SQL/data recipe
------------------------
schema.sql copies the exact source DDL for only the eight query relations.
All source primary keys and NOT NULL fields are preserved. Source DDL contains
no foreign keys among these tables. Use a fresh schema with no public fallback.

minimal_core.sql gives the seven-row nonempty core: one customer, address,
demographic, date, item, store sale and web sale. The core includes nullable
group output attributes and exact lower/upper ratio bounds. A commented
three-row extension adds a duplicate store payload, duplicate web payload and
an overlapping catalog branch, while count(*) remains 1. The minimal core is
a directly inspectable recipe; the executed validation uses the expanded base.

data.sql is the executed compact 23-row base. Four customers provide:
  C1: store + web, with duplicate matching payloads under distinct primary keys.
  C2: store + catalog only.
  C3: store + web + catalog, exercising OR overlap without double counting.
  C4: store + web, qualifying into a NULL-valued output group.
Customers 1-3 share one demographic group, so its six repeated count columns
must each equal 3. Customer 4's group has each count equal to 1. Both birth-month
choices, both marital/education choices, all three item categories, and the
inclusive month/manager/ratio endpoints appear in the base or bound checks.
web ship and catalog bill keys deliberately differ from the queried bill/ship
keys, so switching customer-key roles does not silently pass.

validate.py creates one fresh followup_dsb_q010_* schema inside a single
dsb_dev transaction, loads schema.sql and data.sql, and runs 35 small scenarios.
Each scenario starts from the same base using a savepoint and rollback. The
cases isolate the two OR branches, overlap/duplicate behavior, required store
and secondary sales, every outer filter, date/category/manager/ratio bounds,
NULL group output, NULL correlation/price fields, and unmatched joins. Every
statement has statement_timeout=2s and lock_timeout=1s. There are no EXPLAIN
ANALYZE calls, target-dataset queries, model calls, or service changes.

Run only with the authorized tiny validation scope:
  .venv/bin/python results/stage_a_followup/dsb_witness/validate.py

Validation result
-----------------
validation.json records exact returned rows for the original and whether each
frozen QUITE candidate matched those ordered rows. All original expectations
passed; all three candidates matched in every scenario. Base outputs are two
nonempty groups (counts 3 and 1). This is finite evidence, not an equivalence
proof, and it does not retroactively change the active run's rejections.

The final validation began 2026-10-27T05:14:58.811818+00:00 and ended
2026-10-27T05:14:58.971738+00:00. An earlier 31-case run began
2026-10-27T05:12:54.002569+00:00 and ended 2026-10-27T05:12:54.193126+00:00;
the final run added four unmatched web/catalog date/item join checks. Both
windows are saved in validation.json to expose possible overlap with active
benchmark work. Each newly created schema was absent after its rollback.

Narrow correction to use after Stage A
-------------------------------------
Build a fresh, separately versioned fixture bundle and apply this query010_0
data lane there. Merely adding query010_0 to PROBES is insufficient because
join-group assignment still overrides county literals. The query-specific lane
must explicitly coordinate its allowed county with its customer/demo joins,
2002 months 4-7, qualifying items, and .77-.87 ratios for store plus web/catalog.
Retain a nonempty lane for both OR branches, duplicate/overlap cases, and the
NULL output group, and probe exact query010_0 output before candidate checks.
Do not replace the existing fixture bundle or mark this eight-table diagnostic
as a full schema coverage-complete prebuilt fixture: the current gate also
requires five cases and population of every declared source table. This
directory deliberately does not change those policies or build generic fixture
infrastructure. Any follow-up verification must reuse the saved candidates,
remain separately labeled, and never become an additional generation arm.
