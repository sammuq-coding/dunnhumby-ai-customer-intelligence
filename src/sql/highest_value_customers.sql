-- Highest observed customer value in the configured pre-cutoff feature window.
SELECT household_key, total_sales, transaction_count, average_basket_value,
       total_quantity, recency_days
FROM customer_features
ORDER BY total_sales DESC
LIMIT 100;
