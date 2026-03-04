import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

data_path = "job_placement_encoded_official.csv"
df = pd.read_csv(data_path)

if "salary" in df.columns:
    y = df["salary"]
    print("Hi")

college_col = [ c for c in df.columns if c.startwith("is_college_")]
stream_col = [c for c in df.columsn if c.startswith("_is_")]

gpa_col = "gpa"


feature_col = gpa_col + stream_col + college_col


X = df[feature_col]