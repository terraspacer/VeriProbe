WITH qualifying_totals AS MATERIALIZED (
    SELECT l_orderkey, SUM(l_quantity) AS total_quantity
    FROM lineitem
    GROUP BY l_orderkey
    HAVING SUM(l_quantity) > 313
)
SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice, SUM(q.total_quantity) AS sum
FROM qualifying_totals AS q
JOIN orders AS o ON o.o_orderkey = q.l_orderkey
JOIN customer AS c ON c.c_custkey = o.o_custkey
GROUP BY c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice
ORDER BY o.o_totalprice DESC, o.o_orderdate
LIMIT 100;
