import sys
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os

# Fix Windows encoding for print statements
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'diabetes.csv')

diabetes_dataset = pd.read_csv(csv_path)

print("=" * 50)
print("       DIABETES PREDICTION - SVM MODEL")
print("=" * 50)

print(f"\nDataset Shape: {diabetes_dataset.shape}")
print(f"\nFirst 5 rows:\n{diabetes_dataset.head()}")
print(f"\nStatistical Summary:\n{diabetes_dataset.describe()}")
print(f"\nOutcome Distribution:\n{diabetes_dataset['Outcome'].value_counts()}")
print(f"\nMean by Outcome:\n{diabetes_dataset.groupby('Outcome').mean(numeric_only=True)}")

# ─────────────────────────────────────────────
# 2. Separate Features and Labels
# ─────────────────────────────────────────────
X = diabetes_dataset.drop('Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape:   {Y.shape}")

# ─────────────────────────────────────────────
# 3. Data Standardization
# ─────────────────────────────────────────────
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

print("\nData standardized successfully.")

# ─────────────────────────────────────────────
# 4. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(f"\nDataset split:")
print(f"   Total samples : {X.shape[0]}")
print(f"   Training      : {X_train.shape[0]}")
print(f"   Testing       : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 5. Train SVM Classifier
# ─────────────────────────────────────────────
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
print("\nSVM model trained successfully.")

# ─────────────────────────────────────────────
# 6. Model Evaluation
# ─────────────────────────────────────────────
X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print(f"\nModel Accuracy:")
print(f"   Training Data : {training_accuracy * 100:.2f}%")
print(f"   Test Data     : {test_accuracy * 100:.2f}%")

# ─────────────────────────────────────────────
# 7. Save Model and Scaler
# ─────────────────────────────────────────────
model_path = os.path.join(script_dir, 'diabetes_model.sav')
scaler_path = os.path.join(script_dir, 'scaler.sav')

pickle.dump(classifier, open(model_path, 'wb'))
pickle.dump(scaler, open(scaler_path, 'wb'))

print(f"\nModel saved  -> {model_path}")
print(f"Scaler saved -> {scaler_path}")

# ─────────────────────────────────────────────
# 8. Quick Prediction Test
# ─────────────────────────────────────────────
print("\n" + "-" * 50)
print("  SAMPLE PREDICTION TEST")
print("-" * 50)

# Sample: Diabetic case (Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print(f"\n  Input: {input_data}")
    print("  Result: The person is NOT Diabetic")
else:
    print(f"\n  Input: {input_data}")
    print("  Result: The person IS Diabetic")

print("\n" + "=" * 50)
print("  Training complete! Run the app with Streamlit.")
print("  Command: streamlit run app.py")
print("=" * 50 + "\n")
