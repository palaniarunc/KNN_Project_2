import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("job_placement.csv")
encodeddf = pd.read_csv("job_placement_encoded_official.csv")


sns.countplot(data=df, y="stream", hue="placement_status")
plt.title("Job Placement Status by Major")
plt.show()

sns.boxplot(data = df, x="placement_status", y= "gpa")
plt.title("Job Placement by GPA")
plt.show()