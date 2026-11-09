-- Staging: typed, cleaned customers. TotalCharges is blank for the 11 customers with tenure 0
-- (never billed), so blank -> 0 rather than dropping them.
DROP TABLE IF EXISTS stg_customers;
CREATE TABLE stg_customers AS
SELECT customerID AS customer_id,
       SeniorCitizen AS senior,
       CASE WHEN Partner='Yes' THEN 1 ELSE 0 END AS has_partner,
       CASE WHEN Dependents='Yes' THEN 1 ELSE 0 END AS has_dependents,
       tenure AS tenure_months,
       PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, StreamingTV, StreamingMovies, Contract,
       CASE WHEN PaperlessBilling='Yes' THEN 1 ELSE 0 END AS paperless,
       PaymentMethod,
       CAST(MonthlyCharges AS REAL) AS monthly_charges,
       COALESCE(CAST(NULLIF(TRIM(TotalCharges), '') AS REAL), 0.0) AS total_charges,
       CASE WHEN Churn='Yes' THEN 1 ELSE 0 END AS churned
FROM raw_customers;
