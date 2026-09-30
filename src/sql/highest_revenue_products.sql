-- Revenue-ranked catalog products; unsold products have total_sales = 0.
SELECT PRODUCT_ID, DEPARTMENT, BRAND, COMMODITY_DESC, total_units_sold,
       total_sales, number_of_households_purchasing,
       number_of_baskets_containing_product, repeat_purchase_rate
FROM product_features
ORDER BY total_sales DESC
LIMIT 100;
