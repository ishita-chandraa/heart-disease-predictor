import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import joblib


# Load dataset
data = pd.read_csv("heart.csv")

# Display first five rows
print(data.head())

# Information about dataset
print(data.info())

# Shape of dataset
print("Shape:", data.shape)
print("\nCondition Counts:")
print(data["condition"].value_counts())
print("\nCondition Percentages:")
print(data["condition"].value_counts(normalize=True) * 100)

# Features and target
X = data.drop("condition", axis=1)
y = data["condition"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
model = LogisticRegression(
    C=10,
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Probability of class 1 (Heart Disease)
y_prob = model.predict_proba(X_test)[:, 1]

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc)


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

y_prob = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
}

joblib.dump(metrics, "metrics.pkl")

print("\nModel Saved Successfully!")