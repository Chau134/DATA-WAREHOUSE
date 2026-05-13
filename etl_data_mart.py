import oracledb
import pandas as pd

# Kết nối DB
conn = oracledb.connect(
    user="nch_sleep_dw",
    password="database",
    dsn="localhost:1521/FREE"
)

# Query dữ liệu
query = """
SELECT
    mean AS "Mean",
    variance AS "Variance",
    skewness AS "Skewness",
    kurtosis AS "Kurtosis",
    power_delta AS "Power_Delta",
    power_theta AS "Power_Theta",
    power_alpha AS "Power_Alpha",
    power_beta AS "Power_Beta",
    theta_alpha_ratio AS "Theta_Alpha_Ratio",
    spectral_entropy AS "Spectral_Entropy",
    rolling_var AS "Rolling_Var",
    label AS "Label"
FROM fact_eeg_features
"""

# Đọc vào pandas
df = pd.read_sql(query, conn)

# Xuất CSV
df.to_csv("all_subjects_features.csv", index=False)

print("Export thành công file all_subjects_features.csv")

conn.close()