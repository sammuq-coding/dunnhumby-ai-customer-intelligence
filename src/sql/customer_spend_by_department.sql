-- Requires feature_transactions (DAY-filtered at the configured cutoff) and
-- product_features temp views. A null DEPARTMENT remains visible as its own group.
SELECT t.household_key, p.DEPARTMENT, SUM(t.SALES_VALUE) AS total_sales
FROM feature_transactions t
JOIN product_features p ON t.PRODUCT_ID = p.PRODUCT_ID
GROUP BY t.household_key, p.DEPARTMENT
ORDER BY t.household_key, total_sales DESC;
