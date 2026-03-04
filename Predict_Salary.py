import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np


data_path = "job_placement_encoded_official.csv"
df = pd.read_csv(data_path)

df = df[df["salary"] > 0]

if "salary" in df.columns:
    y = df["salary"]
    print("Hi")

college_col = [ c for c in df.columns if c.startswith("is_college_")]
stream_col = [c for c in df.columns if c.startswith("_is_")]

gpa_col = "gpa"


feature_col = [gpa_col] + stream_col + college_col


X = df[feature_col]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)


#Finding the best k from 1 -30
k_val = range(1,31)
r2_scores = []
mae_scores = []

best_k = None
best_r2 = float("-inf")
best_pipe = None


#Scaling is below and finding the best k value
for k in k_val:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            metric="minkowski",
            p=2
    
      ))  
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    r2_scores.append(r2)
    mae_scores.append(mae)

    if r2 > best_r2:
        best_r2 = r2
        best_k = k
        best_pipe = pipe

print("Best model parameters are below:")
print("Best K:", best_k)
print("Best R2 Score:", best_r2)



best_pred = best_pipe.predict(X_test)
best_mae = mean_absolute_error(y_test, best_pred)
print("Best Test MAE:", best_mae)


#Graphing Below


#K vs R^2 
plt.figure()
plt.plot(list(k_val), r2_scores)
plt.xlabel("K value")
plt.ylabel("R^2 Score")
plt.title("K Value vs R^2 Score")
plt.scatter(best_k, best_r2, color = "red", label="Best K")
plt.annotate(f"Best k: {best_k}\nR^2: {best_r2:.4f}", (best_k, best_r2), textcoords="offset points", xytext=(0,-40), ha="center")



#Actual vs predicted Salary
#Graph needs title

sorted_x = np.argsort(y_test)
sorted_y_test = y_test.values[sorted_x]
sorted_y_pred = y_pred[sorted_x]
plt.figure(figsize=(10,6))
plt.xlabel("Test Case Number")
plt.ylabel("Salary in $")
plt.title("Comparsion of Actual and Predicted Salaries")

plt.plot(sorted_y_test, label = "Actual Salary", color = "blue")
#might need to put index
plt.plot(sorted_y_pred, label = "Predicted Salary", color = "orange")
plt.legend()



plt.show()
