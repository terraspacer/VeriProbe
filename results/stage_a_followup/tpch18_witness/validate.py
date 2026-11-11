#!/usr/bin/env python3
"""Only TPCH query18_0 witness checks; owned schema is always rolled back."""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import uuid

import psycopg
from psycopg import sql

HERE = Path(__file__).resolve().parent
DSN = 'host=/workspace/VeriProbe/.runtime/socket port=55432 dbname=tpch_dev user=steve'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def clean(v):
    if isinstance(v, (Decimal, datetime)):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [clean(x) for x in v]
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    return v


BASE = [(102, '315.00'), (101, '314.00'), (105, '313.01')]
SCENARIOS = [
    ('base_ordering_and_thresholds', '', BASE),
    ('nine_row_core', 'DELETE FROM customer WHERE c_custkey<>1; DELETE FROM orders WHERE o_orderkey<>101; DELETE FROM lineitem WHERE l_orderkey<>101', [(101, '314.00')]),
    ('strict_threshold_equal_313', 'UPDATE lineitem SET l_quantity=13 WHERE l_orderkey=101 AND l_linenumber=7', [(102, '315.00'), (105, '313.01')]),
    ('threshold_below_313', 'UPDATE lineitem SET l_quantity=12 WHERE l_orderkey=101 AND l_linenumber=7', [(102, '315.00'), (105, '313.01')]),
    ('threshold_just_above_313', 'UPDATE lineitem SET l_quantity=13.01 WHERE l_orderkey=101 AND l_linenumber=7', [(102, '315.00'), (101, '313.01'), (105, '313.01')]),
    ('exact_duplicate_line_counts', 'INSERT INTO lineitem SELECT * FROM lineitem WHERE l_orderkey=101 AND l_linenumber=1', [(102, '315.00'), (101, '364.00'), (105, '313.01')]),
    ('duplicate_line_crosses_threshold', 'INSERT INTO lineitem SELECT * FROM lineitem WHERE l_orderkey=103 AND l_linenumber=1', [(103, '362.00'), (102, '315.00'), (101, '314.00'), (105, '313.01')]),
    ('duplicate_order_multiplies_sum', 'INSERT INTO orders SELECT * FROM orders WHERE o_orderkey=101', [(102, '315.00'), (101, '628.00'), (105, '313.01')]),
    ('duplicate_customer_multiplies_sum', 'INSERT INTO customer SELECT * FROM customer WHERE c_custkey=1', [(102, '630.00'), (101, '628.00'), (105, '313.01')]),
    ('both_parent_multiplicities', 'INSERT INTO customer SELECT * FROM customer WHERE c_custkey=1; INSERT INTO orders SELECT * FROM orders WHERE o_orderkey=101', [(102, '630.00'), (101, '1256.00'), (105, '313.01')]),
    ('customer_join_required', 'DELETE FROM customer WHERE c_custkey=1', [(105, '313.01')]),
    ('order_join_required', 'DELETE FROM orders WHERE o_orderkey=101', [(102, '315.00'), (105, '313.01')]),
    ('unmatched_line_key', 'UPDATE lineitem SET l_orderkey=888 WHERE l_orderkey=101', [(102, '315.00'), (105, '313.01')]),
    ('no_qualifying_order', 'UPDATE lineitem SET l_quantity=1', []),
    ('price_primary_sort', 'UPDATE orders SET o_totalprice=950 WHERE o_orderkey=101', [(101, '314.00'), (102, '315.00'), (105, '313.01')]),
    ('date_secondary_sort', "UPDATE orders SET o_orderdate=DATE '2001-01-01' WHERE o_orderkey=101", [(101, '314.00'), (102, '315.00'), (105, '313.01')]),
]

# 101 tiny qualifying orders, seven lines each, distinguish the LIMIT boundary.
# Distinct prices make the result deterministic under the original ORDER BY.
LIMIT_DELTA = """
INSERT INTO orders
SELECT 1000+n,1,'O',10000+n,DATE '2002-01-01','1-URGENT','clerk',0,'limit witness'
FROM generate_series(1,101) n;
INSERT INTO lineitem
SELECT 1000+n,1,1,j,CASE WHEN j<7 THEN 50 ELSE 14 END,100,0,0,'N','O',
       DATE '2002-01-01',DATE '2002-01-02',DATE '2002-01-03',
       'DELIVER IN PERSON','AIR','limit witness'
FROM generate_series(1,101) n CROSS JOIN generate_series(1,7) j;
"""


def main():
    report = {'started_at_utc': stamp(), 'database': 'tpch_dev',
              'namespace': 'followup_tpch18_' + uuid.uuid4().hex[:16],
              'statement_timeout_ms': 2000, 'lock_timeout_ms': 1000,
              'purpose': 'Finite verification witness only; no benchmark timing.', 'cases': []}
    provenance = json.loads((HERE / 'provenance.json').read_text())
    original = (HERE / 'original.sql').read_text()
    candidates = [(c['file'], (HERE / c['file']).read_text()) for c in provenance['candidates']]
    conn = psycopg.connect(DSN, autocommit=False)
    try:
        conn.execute("SET LOCAL statement_timeout='2s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(report['namespace'])))
        conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(report['namespace'])))
        conn.execute((HERE / 'schema.sql').read_text())
        conn.execute((HERE / 'data.sql').read_text())
        report['base_rows_by_table'] = {t: conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(t))).fetchone()[0] for t in ('customer','orders','lineitem')}
        cases = SCENARIOS + [('limit_100_of_101_qualifying_orders', LIMIT_DELTA,
                              [(k, '314.00') for k in range(1101, 1001, -1)])]
        for name, delta, expected in cases:
            conn.execute('SAVEPOINT witness_case')
            if delta:
                conn.execute(delta)
            rows = conn.execute(original).fetchall()
            actual = [(r[2], str(r[5])) for r in rows]
            assert actual == expected, (name, actual, expected)
            case = {'name': name, 'original_rows': clean(rows), 'row_count': len(rows),
                    'original_sha256': hashlib.sha256(repr(rows).encode()).hexdigest(), 'candidates': []}
            for filename, query in candidates:
                candidate_rows = conn.execute(query).fetchall()
                case['candidates'].append({'file': filename, 'same_ordered_rows': candidate_rows == rows,
                                           'row_count': len(candidate_rows)})
            report['cases'].append(case)
            conn.execute('ROLLBACK TO SAVEPOINT witness_case')
            conn.execute('RELEASE SAVEPOINT witness_case')
        report['all_original_expectations_passed'] = True
        report['all_candidates_match'] = all(c['same_ordered_rows'] for case in report['cases'] for c in case['candidates'])
    finally:
        conn.rollback()
        report['schema_absent_after_rollback'] = not conn.execute('SELECT EXISTS(SELECT FROM pg_namespace WHERE nspname=%s)', (report['namespace'],)).fetchone()[0]
        conn.rollback()
        conn.close()
        report['ended_at_utc'] = stamp()
        previous = HERE / 'validation.json'
        if previous.exists():
            prior = json.loads(previous.read_text())
            report['prior_validation_windows_utc'] = prior.get('prior_validation_windows_utc', []) + [
                {'started_at_utc': prior['started_at_utc'], 'ended_at_utc': prior['ended_at_utc']}]
        previous.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('started_at_utc','ended_at_utc','schema_absent_after_rollback','all_original_expectations_passed','all_candidates_match')}))
    print('Cases:', len(report['cases']), '; base rows:', sum(report['base_rows_by_table'].values()))


if __name__ == '__main__':
    main()
