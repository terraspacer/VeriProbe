"""Compact, deterministic source-valid rows for Stage A query witness recipes.

This module contains planned examples and expected outputs, never measured DB
results. Every row is rendered against the pinned authoritative DDL by the
builder; omitted nullable columns remain NULL and required columns get inert
values. Query-specific values are explicit here.
"""
from __future__ import annotations

SCHEMAS = {
    'dsb': '805a1b08edf45fee0326efca6056ca881d92a00969e102fe731cf0781d24c0ea',
    'tpch': '4fc33b97f09b573d748179f6a0a4d049f8d5057c7919ce2c163bb80e1af23eef',
    'calcite': '429ae34be8e654b0d06ea1e4f82eccb03edc0ef4c91af3ce38a0f0901b198cb2',
}
QUERY_HASHES = {
    'rbot:dsb:query081_0': 'e00c051b9f43302a954097c000833ad213197ecfce59e0eebcdd3df8ed440c72',
    'rbot:dsb:query038_0': 'c01836be5d904954daa0c5bd1b097dc909726bc0d71f46bddcf973639ab1d9fb',
    'rbot:dsb:query069_0': '93ea2cb38d811b1cdbcd9ee970034fa96083d8730182f94b0822cf5174966b81',
    'rbot:dsb:query014_0': 'be74a4c6b69aee7f60db60a470a864f3017726fe2e2fdcf0a98986368840c8d2',
    'rbot:dsb:query065_0': '0b9c71552f1d194830f78f62764f43f679848ffe8ba74bdd9e1a2ab80e7c6306',
    'rbot:dsb:query087_0': '2f890280f5f2bbb46cad8f2cfe5dbf9085573b51d5d46fd866b58a53f22b0720',
    'rbot:dsb:query100_0': 'a8eeae98d6c13d94c2ba9d84a3060fd0d1458fa03fc831b0331c99239ca4c1c5',
    'rbot:calcite:testDecorrelateTwoExists': 'f11cd65d38eea553b3dfe944bb6044778d8f74bfe54a1e82b31be40519632f23',
    'rbot:calcite:testIsNullPushDown': '4080eeb38b1ae9f3baf7d5a878e11e5a3a2f1c7f33d2a26f5e70542ace9624c5',
    'rbot:tpch:query12_0': '6b286f310f2e4e742e5cd805831d23116107a39a720fdcfd0c32769510d8dd67',
    'rbot:tpch:query19_0': '6ea1a47a5aba5dc62ecdd9380910bf0d2eceefdf7ee098e04fe04b2ce1c6cbc5',
}
PROOF_QUERY = 'rbot:calcite:testIsNullPushDown'


def generated_recipes():
    return {query_id: {'benchmark': query_id.split(':')[1],
                       'directory': '_'.join(query_id.split(':')[1:]),
                       'schema_sha256': SCHEMAS[query_id.split(':')[1]],
                       'query_sha256': digest, 'generated': True}
            for query_id, digest in QUERY_HASHES.items()}


def case_rows(query_id, case):
    """Return partial row dicts, hand-derived expected rows and semantic features."""
    rows = {}

    def add(table, **values):
        rows.setdefault(table, []).append(values)
        return values

    def date(key=1, **values):
        return add('date_dim', d_date_sk=key, d_date_id=f'd{key}', **values)

    def customer(key, **values):
        return add('customer', c_customer_sk=key, c_customer_id=f'C{key}', **values)

    def sale(channel, number, customer_key=1, item=1, date_key=1, **values):
        prefix = {'store': 'ss', 'catalog': 'cs', 'web': 'ws'}[channel]
        result = {f'{prefix}_item_sk': item, f'{prefix}_sold_date_sk': date_key,
                  f'{prefix}_{"ticket_number" if channel == "store" else "order_number"}': number,
                  f'{prefix}_{"customer_sk" if channel == "store" else "bill_customer_sk"}': customer_key}
        result.update(values)
        return add(channel + '_sales', **result)

    def channel_sale(channel, number, key, **values):
        prefix = {'store': 'ss', 'catalog': 'cs', 'web': 'ws'}[channel]
        item, date_key = values.pop('item', 1), values.pop('date_key', 1)
        return sale(channel, number, key, item=item, date_key=date_key,
                    **{f'{prefix}_{k}': v for k, v in values.items()})

    def emp(key, **values):
        data = dict(empno=key, ename=f'E{key}', job='CLERK', mgr=None,
                    hiredate='2000-01-01', sal=10, comm=0, deptno=10, slacker=False)
        data.update(values)
        return add('emp', **data)

    def line(number, **values):
        data = dict(l_orderkey=number, l_partkey=1, l_linenumber=number,
                    l_quantity=1, l_extendedprice=100, l_discount='0.10',
                    l_shipdate='1995-01-01', l_commitdate='1995-01-02',
                    l_receiptdate='1995-01-03', l_shipmode='RAIL')
        data.update(values)
        return add('lineitem', **data)

    name = query_id.split(':')[-1]
    if case == 'empty':
        return {}, ([[0]] if name in {'query038_0', 'query087_0'} else
                    [[None]] if name == 'query19_0' else []), ['all_source_tables_empty']
    features = ['full_source_schema_population']
    if name == 'query081_0':
        date(d_year=2002)
        add('customer_address', ca_address_sk=1, ca_address_id='MI', ca_state='MI')
        add('customer_address', ca_address_sk=2, ca_address_id='NY', ca_state='NY')
        for key, amount in ((1, 300), (2, 100), (3, 5000), (4, 5000)):
            customer(key, c_current_addr_sk=1 if key < 3 else 2, c_first_name=f'First{key}')
            add('catalog_returns', cr_item_sk=key, cr_order_number=key,
                cr_returned_date_sk=1, cr_returning_customer_sk=key,
                cr_returning_addr_sk=1 if key < 3 else 2, cr_return_amt_inc_tax=amount)
        total = 300
        if case == 'null':
            customer(5, c_current_addr_sk=1)
            add('catalog_returns', cr_item_sk=5, cr_order_number=5,
                cr_returned_date_sk=1, cr_returning_customer_sk=5,
                cr_returning_addr_sk=1, cr_return_amt_inc_tax=None)
            features += ['null_group_excluded_from_avg', 'nullable_address_output']
        elif case == 'duplicate':
            add('catalog_returns', **dict(rows['catalog_returns'][0], cr_order_number=101))
            total = 600
            features += ['duplicate_return_payload_changes_sum']
        elif case == 'skew':
            rows['catalog_returns'][0]['cr_return_amt_inc_tax'] = total = 900
            features += ['state_correlated_average', 'distinct_nonempty_output']
        else:
            add('catalog_returns', cr_item_sk=5, cr_order_number=5, cr_returned_date_sk=999,
                cr_returning_customer_sk=1, cr_returning_addr_sk=1, cr_return_amt_inc_tax=9999)
            features += ['orphan_return_date_excluded']
        return rows, [['C1', None, 'First1', None] + [None] * 6 + ['MI'] + [None] * 4 + [f'{total:.2f}']], features
    if name in {'query038_0', 'query087_0'}:
        intersection = name == 'query038_0'
        date(d_date='2001-01-01', d_month_seq=1221 if intersection else 1206)
        for key in range(1, 3 if intersection else 4):
            customer(key, c_last_name=None if case == 'null' else f'Last{key}',
                     c_first_name=f'First{key}', c_birth_month=2, c_birth_year=1975)
        def channel(channel, key, number=None):
            return channel_sale(channel, number or key, key, list_price=280, wholesale_cost=80)
        for key in range(1, 3 if intersection else 4):
            channel('store', key)
        for key in ((1, 2) if intersection else (2,)):
            channel('catalog', key)
        for key in ((1, 2) if intersection else (3,)):
            channel('web', key)
        count = 2 if intersection else 1
        if case == 'null':
            features += ['set_operations_compare_null_payloads_as_equal']
        elif case == 'duplicate':
            for ch in ('store', 'catalog', 'web'):
                channel(ch, 1 if intersection or ch == 'store' else 2 if ch == 'catalog' else 3, 101)
            features += ['duplicate_payloads_preserve_set_semantics']
        elif case == 'skew':
            key = 3 if intersection else 4
            customer(key, c_last_name=f'Last{key}', c_first_name=f'First{key}', c_birth_month=2, c_birth_year=1975)
            for ch in (('store', 'catalog', 'web') if intersection else ('store',)):
                channel(ch, key)
            count += 1
            features += ['distinct_nonempty_output']
        else:
            target = rows['web_sales' if intersection else 'catalog_sales'][-1]
            target['ws_sold_date_sk' if intersection else 'cs_sold_date_sk'] = 999
            count += -1 if intersection else 1
            features += ['unmatched_date_changes_set_membership']
        return rows, [[count]], features
    if name == 'query069_0':
        date(d_year=1999, d_moy=7)
        add('customer_address', ca_address_sk=1, ca_address_id='KS', ca_state='KS')
        add('customer_demographics', cd_demo_sk=1, cd_gender='F', cd_marital_status='S',
            cd_education_status='4 yr Degree', cd_purchase_estimate=100, cd_credit_rating='Good')
        add('customer_demographics', cd_demo_sk=2, cd_gender='M', cd_marital_status='W',
            cd_education_status='Unknown')
        for key in range(1, 5):
            customer(key, c_current_addr_sk=1, c_current_cdemo_sk=2 if key == 2 else 1)
            channel_sale('store', key, key, list_price=180)
        channel_sale('web', 1, 3, list_price=180, ship_customer_sk=2)
        channel_sale('catalog', 1, 2, list_price=180, ship_customer_sk=4)
        expected = [['F', 'S', '4 yr Degree', 1, 100, 1, 'Good', 1],
                    ['M', 'W', 'Unknown', 1, None, 1, None, 1]]
        if case == 'null':
            customer(5, c_current_addr_sk=1, c_current_cdemo_sk=1)
            channel_sale('store', 5, 5, list_price=None)
            features += ['null_store_price_excluded', 'null_output_group']
        elif case == 'duplicate':
            channel_sale('store', 101, 1, list_price=180)
            features += ['duplicate_exists_payloads_do_not_multiply', 'separate_web_and_catalog_exclusions']
        elif case == 'skew':
            rows['customer'][1]['c_current_cdemo_sk'] = 1
            expected = [['F', 'S', '4 yr Degree', 2, 100, 2, 'Good', 2]]
            features += ['two_customers_one_group', 'distinct_nonempty_output']
        else:
            rows['store_sales'][1]['ss_sold_date_sk'] = 999
            expected = expected[:1]
            features += ['unmatched_store_date', 'bill_ship_role_decoys']
        return rows, expected, features
    if name == 'query014_0':
        date(1, d_year=1998, d_moy=12, d_dom=13, d_week_seq=10)
        date(2, d_year=1999, d_moy=12, d_dom=13, d_week_seq=20)
        add('item', i_item_sk=1, i_item_id='I1', i_brand_id=1, i_class_id=2,
            i_category_id=3, i_category='Books', i_manager_id=95)
        channel_sale('store', 1, 1, quantity=10, list_price=10, wholesale_cost=80)
        channel_sale('store', 2, 1, date_key=2, quantity=20, list_price=10, wholesale_cost=80)
        channel_sale('catalog', 1, 1, quantity=1, list_price=10, wholesale_cost=80)
        channel_sale('web', 1, 1, quantity=1, list_price=10, wholesale_cost=80)
        ly, count = 100, 1
        if case == 'null':
            channel_sale('catalog', 2, 1, quantity=None, list_price=10, wholesale_cost=80)
            features += ['null_product_ignored_by_average']
        elif case == 'duplicate':
            channel_sale('store', 101, 1, quantity=10, list_price=10, wholesale_cost=80)
            ly, count = 200, 2
            features += ['duplicate_payload_changes_sum_and_count']
        elif case == 'skew':
            rows['store_sales'][0]['ss_quantity'] = 30
            ly = 300
            features += ['coordinated_year_week_and_channel_intersection', 'distinct_nonempty_output']
        else:
            channel_sale('web', 101, 1, item=999, date_key=999, quantity=999, list_price=10, wholesale_cost=80)
            features += ['unmatched_item_and_date_excluded']
        return rows, [['store', 1, 2, 3, '200.00', 1, 'store', 1, 2, 3, f'{ly:.2f}', count]], features
    if name == 'query065_0':
        date(d_month_seq=1207)
        add('store', s_store_sk=1, s_store_id='S1', s_store_name='Store1', s_state='MI')
        for key in (1, 2):
            add('item', i_item_sk=key, i_item_id=f'I{key}', i_manager_id=55,
                i_item_desc=None if key == 1 and case == 'null' else f'Item{key}',
                i_current_price=10, i_wholesale_cost=5, i_brand='Brand')
        channel_sale('store', 1, 1, store_sk=1, sales_price=1, list_price=2)
        channel_sale('store', 2, 1, item=2, store_sk=1, sales_price=99, list_price=198)
        revenue = 1
        if case == 'null':
            channel_sale('store', 3, 1, store_sk=1, sales_price=None, list_price=2)
            features += ['null_ratio_excluded', 'nullable_item_description_output']
        elif case == 'duplicate':
            channel_sale('store', 101, 1, store_sk=1, sales_price=1, list_price=2)
            revenue = 2
            features += ['duplicate_sales_payload_changes_revenue']
        elif case == 'skew':
            rows['store_sales'][0].update(ss_sales_price=3, ss_list_price=6)
            rows['store_sales'][1].update(ss_sales_price=199, ss_list_price=398)
            revenue = 3
            features += ['skewed_item_revenue_average', 'distinct_nonempty_output']
        else:
            channel_sale('store', 101, 1, item=999, date_key=999, store_sk=999, sales_price=999, list_price=1998)
            features += ['unmatched_date_store_item_excluded']
        return rows, [['Store1', None if case == 'null' else 'Item1', f'{revenue:.2f}', '10.00', '5.00', 'Brand']], features
    if name == 'query100_0':
        date(d_year=2000)
        add('item', i_item_sk=1, i_item_id='I1', i_category='Books')
        add('item', i_item_sk=2, i_item_id='I2', i_category=None if case == 'null' else 'Home', i_manager_id=50)
        add('customer_address', ca_address_sk=1, ca_address_id='A1')
        add('customer_demographics', cd_demo_sk=1, cd_marital_status='W', cd_education_status='4 yr Degree')
        customer(1, c_current_addr_sk=1, c_current_cdemo_sk=1)
        tickets = 3 if case == 'duplicate' else 4 if case == 'skew' else 2
        for ticket in range(1, tickets + 1):
            channel_sale('store', ticket, 1, item=1, list_price=80)
            channel_sale('store', ticket, None if case == 'null' else 1, item=2, list_price=94)
        count = tickets
        if case == 'unmatched':
            rows['store_sales'][-1]['ss_ticket_number'] = 999
            count -= 1
            features += ['unmatched_ticket_excludes_pair']
        elif case == 'null':
            features += ['second_sale_customer_is_not_a_join_key', 'nullable_unused_item_category']
        else:
            features += ['duplicate_pair_payloads_with_distinct_primary_keys', 'distinct_nonempty_output']
        return rows, [[1, 2, count]], features
    if name in {'testDecorrelateTwoExists', 'testIsNullPushDown'}:
        emp(1)
        emp(2, job='BLOCKED', sal=34, deptno=20)
        emp(3, job='BLOCKED', sal=50)
        emp(4, job='GOOD', sal=50, deptno=99)
        if case == 'duplicate':
            emp(5, ename='E1')
            features += ['duplicate_job_department_payloads_with_distinct_primary_keys']
        elif case == 'skew':
            emp(5, ename='E5')
            emp(6, ename='E6')
            features += ['skewed_department_partition', 'distinct_nonempty_output']
        elif case == 'unmatched':
            rows['emp'][-1]['deptno'] = 999
            features += ['department_absent_from_dept_still_self_matches_exists']
        else:
            features += ['nullable_manager', 'query_predicate_columns_are_not_null']
        if name == 'testIsNullPushDown':
            return rows, [], features + ['tautological_empty', 'proof_required', 'count_window_never_null']
        expected = [[r[k] for k in ('empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno', 'slacker')]
                    for r in rows['emp'] if r['job'] != 'BLOCKED']
        return rows, expected, features + ['exists_self_match', 'not_exists_blocks_all_same_job_rows']
    if name == 'query12_0':
        for key, priority in ((1, '1-URGENT'), (2, '2-HIGH'), (3, '3-MEDIUM'), (4, '5-LOW')):
            add('orders', o_orderkey=key, o_orderpriority=priority)
        line(1, l_shipdate='1994-12-30', l_commitdate='1994-12-31', l_receiptdate='1995-01-01')
        line(2)
        line(3)
        line(4, l_shipmode='SHIP')
        line(5, l_orderkey=1, l_receiptdate='1996-01-01')
        line(6, l_orderkey=1, l_commitdate='1995-01-03')
        high, low = 2, 1
        if case == 'duplicate':
            add('lineitem', **rows['lineitem'][0])
            high = 3
            features += ['exact_duplicate_line_multiplies_aggregate']
        elif case == 'skew':
            rows['orders'][2]['o_orderpriority'] = '1-URGENT'
            high, low = 3, 0
            features += ['priority_skew', 'distinct_nonempty_output']
        elif case == 'unmatched':
            rows['lineitem'][2]['l_orderkey'] = 999
            low = 0
            features += ['unmatched_order_excluded']
        else:
            features += ['nullable_source_dimensions_only', 'query_column_nulls_prohibited_by_source_ddl']
        return rows, [['RAIL', high, low]], features + ['inclusive_lower_exclusive_upper_receipt_bounds', 'strict_date_ordering', 'ordered_limit_selects_rail']
    if name == 'query19_0':
        for key, brand, container, size in ((1, 'Brand#55', 'SM CASE', 1), (2, 'Brand#21', 'MED BOX', 10), (3, 'Brand#44', 'LG PKG', 15)):
            add('part', p_partkey=key, p_brand=brand, p_container=container, p_size=size)
            line(key, l_partkey=key, l_quantity={1: 5, 2: 29, 3: 39}[key],
                 l_discount={1: '0.10', 2: '0.20', 3: '0.30'}[key],
                 l_shipmode='AIR' if key != 2 else 'AIR REG', l_shipinstruct='DELIVER IN PERSON')
        revenue = 240
        if case == 'duplicate':
            add('lineitem', **rows['lineitem'][0])
            revenue = 330
            features += ['exact_duplicate_line_multiplies_revenue']
        elif case == 'skew':
            rows['lineitem'][2]['l_extendedprice'] = 200
            revenue = 310
            features += ['skewed_branch_revenue', 'distinct_nonempty_output']
        elif case == 'unmatched':
            rows['lineitem'][2]['l_partkey'] = 999
            revenue = 170
            features += ['unmatched_part_excluded']
        else:
            features += ['nullable_source_dimensions_only', 'query_column_nulls_prohibited_by_source_ddl']
        return rows, [[f'{revenue:.4f}']], features + ['all_three_or_branches', 'inclusive_quantity_and_size_boundaries']
    raise ValueError(f'No generated case recipe for {query_id}')
