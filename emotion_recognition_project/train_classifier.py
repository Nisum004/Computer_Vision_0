import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# -------------------------------
# 1️⃣ Load the dataset
# -------------------------------
data_path = './data.txt'

if not os.path.exists(data_path):
    raise FileNotFoundError("data.txt not found! Please run emotion_detection.py first.")

# Each row = [1404 landmark features + 1 label]
data = np.loadtxt(data_path)

# Split features (X) and labels (y)
X = data[:, :-1]   # all columns except last
y = data[:, -1]    # last column

# -------------------------------
# 2️⃣ Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3️⃣ Normalize features
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4️⃣ Train SVM classifier
# -------------------------------
print("Training SVM classifier...")
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# -------------------------------
# 5️⃣ Evaluate
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 6️⃣ Save model and scaler
# -------------------------------
save_dir = "../saved"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "emotion_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

print(f"\n💾 Model saved to {os.path.abspath(save_dir)}")
