import pandas as pd
import numpy as np
df = pd.read_csv('job_placement.csv')
col_drop = ['id','name','degree']
df = df.drop(columns=col_drop)


output_path = "/"