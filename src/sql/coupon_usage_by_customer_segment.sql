-- classification_1 is an observed coded field; its business meaning is not
-- asserted here. Requires household_demographics temp view.
SELECT d.classification_1, COUNT(*) AS households,
       AVG(c.coupon_usage) AS average_coupon_discounted_baskets,
       SUM(c.coupon_redemption_count) AS coupon_redemption_rows
FROM customer_features c
JOIN household_demographics d ON c.household_key = d.household_key
GROUP BY d.classification_1
ORDER BY average_coupon_discounted_baskets DESC;
