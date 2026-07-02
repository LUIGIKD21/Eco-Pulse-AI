import pandas as pd

df = pd.read_csv("C:\Eco Pulse\PowerLoad_Dataset.csv")
print("="*50)
print("   DATASET CORRELATION ANALYSIS")
print("="*50)
# Calculate correlation matrix against our target variable
correlations = df.select_dtypes(include=['number']).corr()['Power_Load_kW'].sort_values(ascending=False)
print(correlations)
print("="*50)