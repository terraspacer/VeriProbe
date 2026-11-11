-- 57-row base. All source NOT NULL columns are explicitly populated.
-- q101=314, q102=315, q103=312, q104=313, q105=313.01.
-- q106 has no customer; q999 has no order. Both have quantity 350.
-- To isolate the nine-row qualifying core, retain only customer 1, order 101,
-- and the seven lineitems for order 101.
INSERT INTO customer
  (c_custkey,c_name,c_address,c_nationkey,c_phone,c_acctbal,c_mktsegment,c_comment)
VALUES (1,'Alpha','A',1,'1',0,'BUILDING','witness'),
       (2,'Beta','B',1,'2',0,'BUILDING','witness');

INSERT INTO orders
  (o_orderkey,o_custkey,o_orderstatus,o_totalprice,o_orderdate,
   o_orderpriority,o_clerk,o_shippriority,o_comment)
VALUES (101,1,'O',900,DATE '2001-03-03','1-URGENT','clerk',0,'witness'),
       (102,1,'O',900,DATE '2001-02-02','1-URGENT','clerk',0,'witness'),
       (103,2,'O',1000,DATE '2001-04-04','1-URGENT','clerk',0,'witness'),
       (104,2,'O',1100,DATE '2001-05-05','1-URGENT','clerk',0,'witness'),
       (105,2,'O',800,DATE '2001-01-01','1-URGENT','clerk',0,'witness'),
       (106,999,'O',1200,DATE '2001-06-06','1-URGENT','clerk',0,'unmatched customer');

INSERT INTO lineitem
  (l_orderkey,l_partkey,l_suppkey,l_linenumber,l_quantity,l_extendedprice,
   l_discount,l_tax,l_returnflag,l_linestatus,l_shipdate,l_commitdate,
   l_receiptdate,l_shipinstruct,l_shipmode,l_comment)
SELECT q.orderkey,1,1,n,
       CASE WHEN n<7 THEN 50 ELSE q.last_quantity END,
       100,0,0,'N','O',DATE '2001-06-01',DATE '2001-06-02',
       DATE '2001-06-03','DELIVER IN PERSON','AIR','witness'
FROM (VALUES (101,14.00),(102,15.00),(103,12.00),(104,13.00),
             (105,13.01),(106,50.00),(999,50.00)) AS q(orderkey,last_quantity)
CROSS JOIN generate_series(1,7) AS n;
