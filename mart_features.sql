-- Model-ready features. gender is deliberately excluded (no business reason to price/target by it).
DROP TABLE IF EXISTS mart_features;
CREATE TABLE mart_features AS
SELECT customer_id, churned, senior, has_partner, has_dependents, tenure_months, paperless,
       monthly_charges, total_charges, Contract, InternetService, PaymentMethod,
       (PhoneService='Yes') + (OnlineSecurity='Yes') + (OnlineBackup='Yes') + (DeviceProtection='Yes')
         + (TechSupport='Yes') + (StreamingTV='Yes') + (StreamingMovies='Yes') AS n_services,
       CASE WHEN OnlineSecurity='Yes' OR TechSupport='Yes' THEN 1 ELSE 0 END AS has_support_addon,
       CASE WHEN tenure_months = 0 THEN 0.0 ELSE total_charges / tenure_months END AS avg_monthly_paid,
       monthly_charges - CASE WHEN tenure_months = 0 THEN monthly_charges ELSE total_charges / tenure_months END AS price_drift
FROM stg_customers;
