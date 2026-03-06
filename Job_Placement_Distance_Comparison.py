import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

scaler = StandardScaler()
df = pd.read_csv("job_placement_encoded_official.csv")
df = df.dropna()
dftest = df.drop(["is_placed", "is_not_placed", "salary"], axis=1)
dfplacements = df["is_placed"]

scaled_df = scaler.fit_transform(dftest[["age", "gpa", "years_of_experience"]])

distance_formulas = [1,1.5,2] # minkowski distance p values: 1 = manhattan, 2 = euclidean, 1.5 inbetween
results = {}

X_train, X_test, y_train, y_test = train_test_split(scaled_df,dfplacements,test_size=0.25, random_state=42)


for formula in distance_formulas:
    knn = KNeighborsClassifier(n_neighbors=5, weights = "distance", p=formula)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_predict)
    results[formula]=accuracy
    print(f"Accuracy with p = {formula} : {accuracy}")

best_distance_metric = max(results)
k_vals = range(1,31)
k_scores = []

for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k, p = best_distance_metric)
    knn.fit(X_train, y_train)
    k_scores.append(knn.score(X_test, y_test))

bestk = max(k_scores)
print(bestk)


plt.figure(figsize = (10,6))
plt.subplot(1,2,1)
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Comparison of Distance Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(k_vals, k_scores, marker='o', color='red')
plt.title(f'Optimal K-Value for p = {best_distance_metric}')
plt.xlabel('K Neighbors')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

print(f"The optimal distance p value for this dataset is {best_distance_metric.upper()}.")