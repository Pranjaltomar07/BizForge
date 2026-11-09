DROP TABLE IF EXISTS mart_churn_by_contract;
CREATE TABLE mart_churn_by_contract AS
SELECT Contract AS segment, COUNT(*) AS customers, AVG(churned) AS churn_rate, SUM(monthly_charges) AS monthly_revenue
FROM stg_customers GROUP BY 1;
DROP TABLE IF EXISTS mart_churn_by_tenure;
CREATE TABLE mart_churn_by_tenure AS
SELECT CASE WHEN tenure_months < 6 THEN '0-5' WHEN tenure_months < 12 THEN '6-11' WHEN tenure_months < 24 THEN '12-23'
            WHEN tenure_months < 48 THEN '24-47' ELSE '48+' END AS segment,
       COUNT(*) AS customers, AVG(churned) AS churn_rate FROM stg_customers GROUP BY 1;
