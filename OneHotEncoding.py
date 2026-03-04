import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("job_placement_rcolumns.csv")

categorical_cols = ['gender', 'degree', 'stream', 'college_name', 'placement_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first = False)

boolean_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[boolean_cols] = df_encoded[boolean_cols].astype(int)

