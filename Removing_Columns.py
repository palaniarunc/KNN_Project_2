import pandas as pd
import numpy as np
df = pd.read_csv('job_placement.csv')
col_drop = ['id','name','degree']
df = df.drop(columns=col_drop)


output_path = "/Users/palani/Documents/GitHub/KNN_Project_2/job_placement_rcolumns.csv"
df.to_csv(output_path, index=False)
print("Encoded data saved to,", output_path)


