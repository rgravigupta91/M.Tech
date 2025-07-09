import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, np.nan, 10],
    'C': [11, 12, 13, 14, np.nan]
}
df = pd.DataFrame(data)

# Removing rows with missing values
df_dropna = df.dropna()
print("DataFrame after removing rows with missing values:")
print(df_dropna)

# Imputing missing values with the mean
df_fillna_mean = df.fillna(df.mean())
print("\nDataFrame after imputing missing values with the mean:")
print(df_fillna_mean)


