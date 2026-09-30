-- Descriptive results among observed campaign assignments, not a randomized lift.
SELECT CAMPAIGN, campaign_type, START_DAY, END_DAY, exposed_households,
       coupon_redemption_count, redeeming_households, descriptive_redemption_rate,
       spend_before_campaign, spend_during_campaign, spend_after_campaign,
       campaign_window_complete, post_campaign_observation_complete
FROM campaign_features
ORDER BY descriptive_redemption_rate DESC NULLS LAST;
