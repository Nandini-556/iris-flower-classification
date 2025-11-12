# simple_debug_iris_pipeline.py (save this as iris_pipeline.py, overwrite existing)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import sys

print("STARTING IRIS PIPELINE")  # visible start marker

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
print("Loaded Iris dataset. Samples:", len(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split done. Train size:", len(X_train), "Test size:", len(X_test))

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling done.")

# Train
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model training complete.")

# Predict & Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("ACCURACY:", acc)

# Save artifacts
joblib.dump(model, 'iris_model.joblib')
joblib.dump(scaler, 'iris_scaler.joblib')
print("Saved: iris_model.joblib, iris_scaler.joblib")

print("PIPELINE FINISHED")
 