import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("job_placement_rcolumns.csv")

categorical_cols = ['gender', 'stream', 'college_name', 'placement_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first = False)

boolean_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[boolean_cols] = df_encoded[boolean_cols].astype(int)

rename_col = {}

for col in df_encoded.columns:
    if col.startswith('gender_'):
        value = col.replace("gender_","")
        rename_col[col] = f'is_{value.lower()}'
    elif col.startswith("degree"):
        value = col.replace("degree_", "")
        rename_col[col] = f'is_degree{value.lower().replace(" ", "").replace(" ", "_")}'
    elif col.startswith("stream"):
        value = col.replace("stream_","")
        rename_col[col] = f'_is_{value.lower().replace(" ", "_")}'
    elif col.startswith("college_name_"):
        college = col.replace("college_name_","")
        rename_col[col] = f'is_college_{college.lower().replace(" ", "_").replace("--","_")}'
    elif col.startswith("placement_status_"):
        status = col.replace('placement_status_',"")
        if status == "Placed":
            rename_col[col] = "is_placed"
        else:
            rename_col[col] = "is_not_placed"


df_encoded = df_encoded.rename(columns=rename_col)
output_path = "/Users/palani/Documents/GitHub/KNN_Project_2/job_placement_encoded_official.csv"
df_encoded.to_csv(output_path, index=False)
print("Encoded data is saved in:", output_path)