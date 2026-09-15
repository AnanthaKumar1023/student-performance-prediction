# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# Marks Prediction + Pass/Fail Prediction
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("student_performance.csv")

print("\n==============================================")
print("     STUDENT PERFORMANCE PREDICTION SYSTEM")
print("==============================================")

print("\nDataset loaded successfully!")


# ============================================================
# 3. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n----------------------------------------------")
print("FIRST 5 ROWS")
print("----------------------------------------------")

print(df.head())


# ============================================================
# 4. DATASET SHAPE
# ============================================================

print("\n----------------------------------------------")
print("DATASET SHAPE")
print("----------------------------------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n----------------------------------------------")
print("DATASET INFORMATION")
print("----------------------------------------------")

print(df.info())


# ============================================================
# 6. STATISTICAL INFORMATION
# ============================================================

print("\n----------------------------------------------")
print("STATISTICAL INFORMATION")
print("----------------------------------------------")

print(df.describe())


# ============================================================
# 7. DATA CLEANING
# ============================================================

print("\n----------------------------------------------")
print("MISSING VALUES")
print("----------------------------------------------")

print(df.isnull().sum())


print("\n----------------------------------------------")
print("DUPLICATE ROWS")
print("----------------------------------------------")

print(df.duplicated().sum())


print("\n----------------------------------------------")
print("DATA TYPES")
print("----------------------------------------------")

print(df.dtypes)


# ============================================================
# 8. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "sleep_hours",
    "previous_score",
    "marks",
    "pass_fail"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) > 0:

    print("\nERROR!")
    print("The following columns are missing:")
    print(missing_columns)

    print("\nRequired columns are:")
    print(required_columns)

    raise SystemExit


# ============================================================
# 9. REMOVE MISSING VALUES
# ============================================================

df = df.dropna().copy()

print("\n----------------------------------------------")
print("AFTER DATA CLEANING")
print("----------------------------------------------")

print("Rows remaining:", len(df))


# ============================================================
# 10. FEATURES
# ============================================================

features = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "sleep_hours",
    "previous_score"
]


# ============================================================
# 11. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n----------------------------------------------")
print("DISPLAYING EDA GRAPHS")
print("----------------------------------------------")

plt.figure(figsize=(14, 10))


# ------------------------------------------------------------
# Graph 1: Study Hours vs Marks
# ------------------------------------------------------------

plt.subplot(2, 2, 1)

plt.scatter(
    df["study_hours"],
    df["marks"]
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.grid(True)


# ------------------------------------------------------------
# Graph 2: Attendance vs Marks
# ------------------------------------------------------------

plt.subplot(2, 2, 2)

plt.scatter(
    df["attendance"],
    df["marks"]
)

plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.grid(True)


# ------------------------------------------------------------
# Graph 3: Assignments vs Marks
# ------------------------------------------------------------

plt.subplot(2, 2, 3)

plt.scatter(
    df["assignments_completed"],
    df["marks"]
)

plt.xlabel("Assignments Completed")
plt.ylabel("Marks")
plt.title("Assignments Completed vs Marks")
plt.grid(True)


# ------------------------------------------------------------
# Graph 4: Correlation Heatmap
# ------------------------------------------------------------

plt.subplot(2, 2, 4)

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

print("\nClose the EDA graph window to continue.")

plt.show()


# ============================================================
# 12. MARKS PREDICTION
#    LINEAR REGRESSION
# ============================================================

print("\n==============================================")
print("             MARKS PREDICTION")
print("==============================================")


# Input features
X = df[features]

# Target = Marks
y = df["marks"]


print("\nInput Features:")
print(features)

print("\nTarget:")
print("marks")


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n----------------------------------------------")
print("TRAIN / TEST SPLIT")
print("----------------------------------------------")

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 14. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\n----------------------------------------------")
print("FEATURE SCALING COMPLETED")
print("----------------------------------------------")


# ============================================================
# 15. CREATE LINEAR REGRESSION MODEL
# ============================================================

linear_model = LinearRegression()


# ============================================================
# 16. TRAIN LINEAR REGRESSION MODEL
# ============================================================

linear_model.fit(
    X_train_scaled,
    y_train
)

print("\nLinear Regression model trained successfully!")


# ============================================================
# 17. PREDICT MARKS
# ============================================================

y_pred = linear_model.predict(
    X_test_scaled
)


print("\n----------------------------------------------")
print("ACTUAL VS PREDICTED MARKS")
print("----------------------------------------------")

for i in range(len(y_pred)):

    print(
        "Actual:",
        y_test.iloc[i],
        "  Predicted:",
        round(y_pred[i], 2)
    )


# ============================================================
# 18. MARKS MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n----------------------------------------------")
print("MARKS MODEL EVALUATION")
print("----------------------------------------------")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))


# ============================================================
# 19. ACTUAL VS PREDICTED MARKS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")

plt.title("Actual vs Predicted Marks")

plt.grid(True)

print("\nClose the Actual vs Predicted graph to continue.")

plt.show()


# ============================================================
# 20. PASS / FAIL PREDICTION
#    LOGISTIC REGRESSION
# ============================================================

print("\n==============================================")
print("          PASS / FAIL PREDICTION")
print("==============================================")


# Input features
X_class = df[features]

# Pass/Fail target
y_class_original = df["pass_fail"]


# ============================================================
# 21. ENCODE PASS / FAIL
# ============================================================

label_encoder = LabelEncoder()

y_class = label_encoder.fit_transform(
    y_class_original.astype(str)
)


print("\nPass/Fail classes:")

for number, name in enumerate(label_encoder.classes_):

    print(number, "=", name)


# Check number of classes

if len(label_encoder.classes_) < 2:

    print("\nERROR: Dataset contains only one Pass/Fail class.")

    raise SystemExit


# ============================================================
# 22. TRAIN / TEST SPLIT FOR CLASSIFICATION
# ============================================================

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)


print("\n----------------------------------------------")
print("CLASSIFICATION TRAIN / TEST SPLIT")
print("----------------------------------------------")

print("Training samples :", len(X_train_c))
print("Testing samples  :", len(X_test_c))


# ============================================================
# 23. FEATURE SCALING FOR CLASSIFICATION
# ============================================================

scaler_class = StandardScaler()

X_train_c_scaled = scaler_class.fit_transform(
    X_train_c
)

X_test_c_scaled = scaler_class.transform(
    X_test_c
)


# ============================================================
# 24. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

classifier = LogisticRegression(
    max_iter=1000
)


# ============================================================
# 25. TRAIN LOGISTIC REGRESSION
# ============================================================

classifier.fit(
    X_train_c_scaled,
    y_train_c
)


print("\nLogistic Regression model trained successfully!")


# ============================================================
# 26. PASS / FAIL PREDICTION
# ============================================================

y_pred_class = classifier.predict(
    X_test_c_scaled
)


print("\n----------------------------------------------")
print("ACTUAL VS PREDICTED PASS / FAIL")
print("----------------------------------------------")


for i in range(len(y_pred_class)):

    actual = label_encoder.inverse_transform(
        [y_test_c[i]]
    )[0]

    predicted = label_encoder.inverse_transform(
        [y_pred_class[i]]
    )[0]

    print(
        "Actual:",
        actual,
        "  Predicted:",
        predicted
    )


# ============================================================
# 27. CLASSIFICATION ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test_c,
    y_pred_class
)


print("\n----------------------------------------------")
print("CLASSIFICATION ACCURACY")
print("----------------------------------------------")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 28. CLASSIFICATION REPORT
# ============================================================

print("\n----------------------------------------------")
print("CLASSIFICATION REPORT")
print("----------------------------------------------")

print(
    classification_report(
        y_test_c,
        y_pred_class,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 29. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_c,
    y_pred_class
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Pass / Fail Confusion Matrix")

print("\nClose the Confusion Matrix to finish.")

plt.show()


# ============================================================
# 30. FINAL PROJECT RESULT
# ============================================================

print("\n==============================================")
print("       PROJECT COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nRESULT SUMMARY")

print("----------------------------------------------")
print("Marks Prediction")
print("----------------------------------------------")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))

print("\n----------------------------------------------")
print("Pass / Fail Prediction")
print("----------------------------------------------")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\n==============================================")
print("              END OF PROJECT")
print("==============================================")