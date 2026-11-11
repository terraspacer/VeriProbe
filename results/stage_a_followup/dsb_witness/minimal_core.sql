-- Seven rows are enough for one nonempty original output group.
-- Apply after schema.sql in a fresh isolated schema, instead of data.sql.
-- The address, demographic, customer, date, item, store sale and web sale are
-- each necessary on this web branch. Date/item rows are shared by both sales.
INSERT INTO customer_address (ca_address_sk, ca_address_id, ca_county)
VALUES (1, 'addr1', 'Northampton County');
INSERT INTO customer_demographics
  (cd_demo_sk, cd_gender, cd_marital_status, cd_education_status,
   cd_purchase_estimate, cd_credit_rating, cd_dep_count,
   cd_dep_employed_count, cd_dep_college_count)
VALUES (1, 'F', 'S', '4 yr Degree', NULL, NULL, NULL, NULL, NULL);
INSERT INTO customer
  (c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_addr_sk, c_birth_month)
VALUES (1, 'customer1', 1, 1, 7);
INSERT INTO date_dim (d_date_sk, d_date_id, d_year, d_moy)
VALUES (1, 'date1', 2002, 4);
INSERT INTO item (i_item_sk, i_item_id, i_category, i_manager_id)
VALUES (1, 'item1', 'Books', 91);
INSERT INTO store_sales
  (ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_ticket_number,
   ss_sales_price, ss_list_price)
VALUES (1, 1, 1, 101, 77, 100);
INSERT INTO web_sales
  (ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_order_number,
   ws_sales_price, ws_list_price)
VALUES (1, 1, 1, 101, 87, 100);

-- Optional three-row extension covers duplicate matching payloads and the
-- overlapping catalog branch. The single outer customer's count stays 1.
-- INSERT INTO store_sales
--   (ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_ticket_number,
--    ss_sales_price, ss_list_price) VALUES (1, 1, 1, 102, 77, 100);
-- INSERT INTO web_sales
--   (ws_sold_date_sk, ws_item_sk, ws_bill_customer_sk, ws_order_number,
--    ws_sales_price, ws_list_price) VALUES (1, 1, 1, 102, 87, 100);
-- INSERT INTO catalog_sales
--   (cs_sold_date_sk, cs_item_sk, cs_ship_customer_sk, cs_order_number,
--    cs_sales_price, cs_list_price) VALUES (1, 1, 1, 101, 82, 100);
