import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("job_placement.csv")
placeddf = df[df["placement_status"]=="Placed"]


categorical_vars = ["gender", "stream", "college_name"]
num_vars = ["gpa", "age", "years_of_experience"]

for var in categorical_vars:
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, y=var, hue="placement_status")
    plt.title(f"Placement Status by {var}")

for var in num_vars:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x = "placement_status", y=var)
    plt.title(f"Distribution of {var} by Placement Status")

plt.figure(figsize=(10,6))
sns.boxplot(data=placeddf, x = "placement_status", y="salary")
plt.title("Salary of Placed Students")

plt.show()