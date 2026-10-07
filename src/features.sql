-- Client-level features from X5 RetailHero purchase history (45.8M receipt lines).
-- Executed by src/build_features.py with DuckDB, which streams the gzipped CSV instead of loading it into memory.
-- All features are computed from purchases made BEFORE the SMS campaign, so there is no leakage from the target.

CREATE OR REPLACE TABLE items AS
SELECT *
FROM read_csv('data/purchases.csv.gz', header = true,
              columns = {
                  'client_id': 'VARCHAR', 'transaction_id': 'VARCHAR', 'transaction_datetime': 'TIMESTAMP',
                  'regular_points_received': 'DOUBLE', 'express_points_received': 'DOUBLE',
                  'regular_points_spent': 'DOUBLE', 'express_points_spent': 'DOUBLE',
                  'purchase_sum': 'DOUBLE', 'store_id': 'VARCHAR', 'product_id': 'VARCHAR',
                  'product_quantity': 'DOUBLE', 'trn_sum_from_iss': 'DOUBLE', 'trn_sum_from_red': 'DOUBLE'});

-- Transaction-level fields are repeated on every receipt line: collapse to one row per receipt.
CREATE OR REPLACE TABLE transactions AS
SELECT client_id,
       transaction_id,
       ANY_VALUE(transaction_datetime)    AS ts,
       ANY_VALUE(purchase_sum)            AS purchase_sum,
       ANY_VALUE(regular_points_received) AS reg_received,
       ANY_VALUE(express_points_received) AS exp_received,
       ANY_VALUE(regular_points_spent)    AS reg_spent,      -- stored as negative numbers
       ANY_VALUE(express_points_spent)    AS exp_spent,
       ANY_VALUE(store_id)                AS store_id,
       COUNT(*)                           AS n_lines,
       SUM(product_quantity)              AS quantity
FROM items
GROUP BY client_id, transaction_id;

CREATE OR REPLACE TABLE ref AS SELECT MAX(ts) AS ref_ts FROM transactions;

CREATE OR REPLACE TABLE trn_features AS
WITH ordered AS (
    SELECT t.*,
           DATE_DIFF('day', LAG(ts) OVER (PARTITION BY client_id ORDER BY ts), ts) AS days_since_prev
    FROM transactions t
)
SELECT client_id,
       COUNT(*)                                                         AS n_trn,
       SUM(purchase_sum)                                                AS total_spend,
       AVG(purchase_sum)                                                AS avg_check,
       STDDEV_SAMP(purchase_sum)                                        AS std_check,
       MAX(purchase_sum)                                                AS max_check,
       DATE_DIFF('day', MAX(ts), (SELECT ref_ts FROM ref))              AS recency_days,
       DATE_DIFF('day', MIN(ts), (SELECT ref_ts FROM ref))              AS first_trn_days_ago,
       COUNT(DISTINCT CAST(ts AS DATE))                                 AS active_days,
       AVG(days_since_prev)                                             AS avg_days_between,
       COUNT(DISTINCT store_id)                                         AS n_stores,
       AVG(n_lines)                                                     AS avg_lines_per_trn,
       AVG(quantity)                                                    AS avg_quantity_per_trn,
       SUM(reg_received)                                                AS reg_points_received,
       SUM(exp_received)                                                AS exp_points_received,
       -SUM(reg_spent)                                                  AS reg_points_spent,
       -SUM(exp_spent)                                                  AS exp_points_spent,
       AVG(CASE WHEN reg_spent < 0 OR exp_spent < 0 THEN 1 ELSE 0 END)  AS share_trn_points_spent,
       AVG(CASE WHEN exp_received > 0 THEN 1 ELSE 0 END)                AS share_trn_express_points,
       AVG(CASE WHEN DAYOFWEEK(ts) IN (0, 6) THEN 1 ELSE 0 END)         AS share_trn_weekend,
       AVG(HOUR(ts))                                                    AS avg_hour,
       COUNT(*) FILTER (WHERE ts > (SELECT ref_ts FROM ref) - INTERVAL 30 DAY)                   AS n_trn_30d,
       COALESCE(SUM(purchase_sum) FILTER (WHERE ts > (SELECT ref_ts FROM ref) - INTERVAL 30 DAY), 0) AS spend_30d,
       COUNT(*) FILTER (WHERE ts > (SELECT ref_ts FROM ref) - INTERVAL 7 DAY)                    AS n_trn_7d
FROM ordered
GROUP BY client_id;

CREATE OR REPLACE TABLE item_features AS
SELECT client_id,
       COUNT(DISTINCT product_id) AS n_unique_products,
       SUM(product_quantity)      AS total_quantity
FROM items
GROUP BY client_id;

CREATE OR REPLACE TABLE features AS
SELECT c.client_id,
       CASE WHEN c.age BETWEEN 14 AND 100 THEN c.age END                         AS age,   -- raw data has ages like -7000 and 1900
       c.gender,
       DATE_DIFF('day', c.first_issue_date, (SELECT ref_ts FROM ref))            AS card_age_days,
       DATE_DIFF('day', c.first_issue_date, c.first_redeem_date)                 AS redeem_delay_days,
       CASE WHEN c.first_redeem_date IS NULL THEN 0 ELSE 1 END                   AS has_redeemed,
       t.* EXCLUDE (client_id),
       i.n_unique_products,
       i.total_quantity,
       t.spend_30d / NULLIF(t.total_spend, 0)                                    AS share_spend_30d
FROM read_csv('data/clients.csv.gz', header = true,
              columns = {'client_id': 'VARCHAR', 'first_issue_date': 'TIMESTAMP',
                         'first_redeem_date': 'TIMESTAMP', 'age': 'INTEGER', 'gender': 'VARCHAR'}) c
LEFT JOIN trn_features  t USING (client_id)
LEFT JOIN item_features i USING (client_id);
