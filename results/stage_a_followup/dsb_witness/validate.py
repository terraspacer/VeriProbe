#!/usr/bin/env python3
"""Validate only query010_0's tiny witness, then roll back the owned schema."""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import uuid

import psycopg
from psycopg import sql

HERE = Path(__file__).resolve().parent
DSN = 'host=/workspace/VeriProbe/.runtime/socket port=55432 dbname=dsb_dev user=steve'
TABLES = ('customer', 'customer_address', 'customer_demographics', 'date_dim',
          'item', 'store_sales', 'web_sales', 'catalog_sales')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, str):
        return value.rstrip()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


# Each delta applies to the same 23-row base and is rolled back before the next.
# C1/C2/C3 share the normal group; C4 is the nullable output group.
SCENARIOS = [
    ('base', '', (3, 1)),
    ('deduplicate_sales_payloads', 'DELETE FROM store_sales WHERE ss_ticket_number=102; DELETE FROM web_sales WHERE ws_order_number=102', (3, 1)),
    ('web_path_only', 'DELETE FROM catalog_sales', (2, 1)),
    ('catalog_path_only', 'DELETE FROM web_sales', (2, 0)),
    ('store_required', 'DELETE FROM store_sales', (0, 0)),
    ('secondary_required', 'DELETE FROM web_sales; DELETE FROM catalog_sales', (0, 0)),
    ('county_rejected', "UPDATE customer_address SET ca_county='outside'", (0, 0)),
    ('birth_month_rejected', 'UPDATE customer SET c_birth_month=6', (0, 0)),
    ('gender_rejected', "UPDATE customer_demographics SET cd_gender='M'", (0, 0)),
    ('marital_rejected', "UPDATE customer_demographics SET cd_marital_status='D'", (0, 0)),
    ('education_rejected', "UPDATE customer_demographics SET cd_education_status='College'", (0, 0)),
    ('year_rejected', 'UPDATE date_dim SET d_year=2001', (0, 0)),
    ('month_low_rejected', 'UPDATE date_dim SET d_moy=3', (0, 0)),
    ('month_high_rejected', 'UPDATE date_dim SET d_moy=8', (0, 0)),
    ('category_rejected', "UPDATE item SET i_category='Home'", (0, 0)),
    ('manager_low_rejected', 'UPDATE item SET i_manager_id=90', (0, 0)),
    ('manager_high_rejected', 'UPDATE item SET i_manager_id=101', (0, 0)),
    ('ratio_low_rejected', 'UPDATE store_sales SET ss_sales_price=76', (0, 0)),
    ('ratio_high_rejected', 'UPDATE store_sales SET ss_sales_price=88', (0, 0)),
    ('null_store_numerator', 'UPDATE store_sales SET ss_sales_price=NULL', (0, 0)),
    ('null_store_denominator', 'UPDATE store_sales SET ss_list_price=NULL', (0, 0)),
    ('null_outer_address', 'UPDATE customer SET c_current_addr_sk=NULL', (0, 0)),
    ('null_outer_demographic', 'UPDATE customer SET c_current_cdemo_sk=NULL', (0, 0)),
    ('null_store_customer', 'UPDATE store_sales SET ss_customer_sk=NULL', (0, 0)),
    ('unmatched_store_date', 'UPDATE store_sales SET ss_sold_date_sk=999', (0, 0)),
    ('unmatched_store_item', 'UPDATE store_sales SET ss_item_sk=999', (0, 0)),
    ('null_web_bill_customer', 'UPDATE web_sales SET ws_bill_customer_sk=NULL', (2, 0)),
    ('null_catalog_ship_customer', 'UPDATE catalog_sales SET cs_ship_customer_sk=NULL', (2, 1)),
    ('unmatched_web_date', 'UPDATE web_sales SET ws_sold_date_sk=999', (2, 0)),
    ('unmatched_web_item', 'UPDATE web_sales SET ws_item_sk=999', (2, 0)),
    ('unmatched_catalog_date', 'UPDATE catalog_sales SET cs_sold_date_sk=999', (2, 1)),
    ('unmatched_catalog_item', 'UPDATE catalog_sales SET cs_item_sk=999', (2, 1)),
    ('null_secondary_ratios', 'UPDATE web_sales SET ws_sales_price=NULL; UPDATE catalog_sales SET cs_list_price=NULL', (0, 0)),
    ('all_lower_bounds', 'UPDATE date_dim SET d_moy=4; UPDATE item SET i_manager_id=91; UPDATE store_sales SET ss_sales_price=77; UPDATE web_sales SET ws_sales_price=77; UPDATE catalog_sales SET cs_sales_price=77', (3, 1)),
    ('all_upper_bounds', 'UPDATE date_dim SET d_moy=7; UPDATE item SET i_manager_id=100; UPDATE store_sales SET ss_sales_price=87; UPDATE web_sales SET ws_sales_price=87; UPDATE catalog_sales SET cs_sales_price=87', (3, 1)),
]


def main():
    started = stamp()
    namespace = 'followup_dsb_q010_' + uuid.uuid4().hex[:16]
    provenance = json.loads((HERE / 'provenance.json').read_text())
    original = (HERE / 'original.sql').read_text()
    candidates = [(c['file'], (HERE / c['file']).read_text()) for c in provenance['candidates']]
    report = {'started_at_utc': started, 'database': 'dsb_dev', 'namespace': namespace,
              'statement_timeout_ms': 2000, 'purpose': 'Finite witness checks only; no benchmark timing.',
              'all_queries_from_frozen_checkpoint': True, 'cases': []}
    conn = psycopg.connect(DSN, autocommit=False)
    try:
        conn.execute("SET LOCAL statement_timeout='2s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace)))
        conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(namespace)))
        conn.execute((HERE / 'schema.sql').read_text())
        conn.execute((HERE / 'data.sql').read_text())
        report['base_rows_by_table'] = {t: conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(t))).fetchone()[0] for t in TABLES}
        for name, delta, expected in SCENARIOS:
            conn.execute('SAVEPOINT witness_case')
            if delta:
                conn.execute(delta)
            rows = conn.execute(original).fetchall()
            counts = {r[1].strip(): r[3] for r in rows}
            assert (counts.get('S', 0), counts.get('M', 0)) == expected, (name, rows, expected)
            assert all(all(r[i] == r[3] for i in (5, 7, 9, 11, 13)) for r in rows), (name, rows)
            case = {'name': name, 'original_rows': clean(rows), 'row_count': len(rows),
                    'original_sha256': hashlib.sha256(repr(rows).encode()).hexdigest(),
                    'candidates': []}
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
        # The CREATE SCHEMA was transactional; this checks that rollback removed it.
        report['schema_absent_after_rollback'] = not conn.execute('SELECT EXISTS(SELECT FROM pg_namespace WHERE nspname=%s)', (namespace,)).fetchone()[0]
        conn.rollback()
        conn.close()
        report['ended_at_utc'] = stamp()
        previous = HERE / 'validation.json'
        if previous.exists():
            prior = json.loads(previous.read_text())
            report['prior_validation_windows_utc'] = prior.get('prior_validation_windows_utc', []) + [
                {'started_at_utc': prior['started_at_utc'], 'ended_at_utc': prior['ended_at_utc']}]
        previous.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('started_at_utc', 'ended_at_utc', 'schema_absent_after_rollback', 'all_original_expectations_passed', 'all_candidates_match')}))
    print('Cases:', len(report['cases']), '; base rows:', sum(report['base_rows_by_table'].values()))


if __name__ == '__main__':
    main()
