-- Query010_0-specific base witness, 23 rows; all omitted fields are nullable.
-- Execute only after schema.sql in a fresh isolated schema.
-- Customer 1: web only, duplicate sales payloads. Customer 2: catalog only.
-- Customer 3: both secondary channels. Customer 4: NULL output group.
INSERT INTO customer_address (ca_address_sk, ca_address_id, ca_county)
VALUES (1, 'addr1', 'Northampton County');

INSERT INTO customer_demographics
  (cd_demo_sk, cd_gender, cd_marital_status, cd_education_status,
   cd_purchase_estimate, cd_credit_rating, cd_dep_count,
   cd_dep_employed_count, cd_dep_college_count)
VALUES (1, 'F', 'S', '4 yr Degree', 1200, 'Good', 2, 1, 0),
       (2, 'F', 'M', 'Unknown', NULL, NULL, NULL, NULL, NULL);

INSERT INTO customer
  (c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_addr_sk, c_birth_month)
VALUES (1, 'customer1', 1, 1, 7), (2, 'customer2', 1, 1, 11),
       (3, 'customer3', 1, 1, 7), (4, 'customer4', 2, 1, 11);

INSERT INTO date_dim (d_date_sk, d_date_id, d_year, d_moy)
VALUES (1, 'date1', 2002, 4), (2, 'date2', 2002, 7);

INSERT INTO item (i_item_sk, i_item_id, i_category, i_manager_id)
VALUES (1, 'item1', 'Books', 91), (2, 'item2', 'Electronics', 100),
       (3, 'item3', 'Sports', 95);

-- Distinct ticket/order keys preserve the source primary keys while repeating
-- the exact payload read by EXISTS. EXISTS must not multiply outer customers.
INSERT INTO store_sales
  (ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_ticket_number,
   ss_sales_price, ss_list_price)
VALUES (1, 1, 1, 101, 77, 100), (1, 1, 1, 102, 77, 100),
       (2, 2, 2, 201, 87, 100), (1, 3, 3, 301, 82, 100),
       (2, 1, 4, 401, 82, 100);

INSERT INTO web_sales
  (ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_ship_customer_sk,
   ws_order_number, ws_sales_price, ws_list_price)
VALUES (1, 1, 1, 9001, 101, 82, 100), (1, 1, 1, 9001, 102, 82, 100),
       (2, 2, 3, 9003, 301, 82, 100), (1, 3, 4, 9004, 401, 82, 100);

INSERT INTO catalog_sales
  (cs_sold_date_sk, cs_item_sk, cs_ship_customer_sk, cs_bill_customer_sk,
   cs_order_number, cs_sales_price, cs_list_price)
VALUES (2, 2, 2, 9002, 201, 82, 100), (1, 3, 3, 9003, 301, 82, 100);
