import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\HARISH\student_performance.csv")

print("\n--- Student Data Preview ---")
print(df.head())

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
df["Result"] = df["FinalMarks"].apply(lambda x: "Pass" if x >= 50 else "Fail")

# -------------------------------
# Step 3: Data Visualization
# -------------------------------
sns.scatterplot(x="StudyHours", y="FinalMarks", hue="Result", data=df)
plt.title("Study Hours vs Final Marks")
plt.show()

numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# Step 4: Model Training
# -------------------------------
X = df[["StudyHours", "Attendance", "PreviousMarks"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# Step 5: GUI for Prediction
# -------------------------------
root = tk.Tk()
root.title("🎓 Student Performance Predictor")

tk.Label(root, text="Enter Study Hours:").grid(row=0, column=0, padx=10, pady=5)
study_entry = tk.Entry(root)
study_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Enter Attendance (%):").grid(row=1, column=0, padx=10, pady=5)
attend_entry = tk.Entry(root)
attend_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Enter Previous Marks:").grid(row=2, column=0, padx=10, pady=5)
marks_entry = tk.Entry(root)
marks_entry.grid(row=2, column=1, padx=10, pady=5)

def predict_result():
    try:
        study = float(study_entry.get())
        attend = float(attend_entry.get())
        marks = float(marks_entry.get())
        new_data = np.array([[study, attend, marks]])
        prediction = model.predict(new_data)
        messagebox.showinfo("Prediction Result", f"Predicted Result: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values!")

tk.Button(root, text="Predict Result", command=predict_result, bg="#4CAF50", fg="white").grid(row=3, column=1, pady=10)
tk.Label(root, text=f"Model Accuracy: {round(accuracy * 100, 2)}%", font=("Arial", 10, "bold")).grid(row=4, column=1, pady=5)

root.mainloop()
