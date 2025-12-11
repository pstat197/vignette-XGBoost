# Import dependencies
import pandas as pd

# Get data from local CSV
df = pd.read_csv("data/Iris.csv", index_col=0)

# EDA
print(df.head())
print(len(df))
# 150 observations and 5 covariates; 4 predictors (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) and 1 response (Species)