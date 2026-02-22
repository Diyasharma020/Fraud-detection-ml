import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (prevents blocking)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE


# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("data/creditcard.csv")

print("Shape of Dataset:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

print("\nClass Distribution:\n")
print(df["Class"].value_counts())

fraud_percent = (df["Class"].sum() / len(df)) * 100
print(f"\nFraud Percentage: {fraud_percent:.4f}%")

# Save class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.tight_layout()
plt.savefig("model/class_distribution.png")
plt.close()


# -----------------------------
# 2️⃣ Train-Test Split
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# -----------------------------
# 3️⃣ Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 4️⃣ Handle Imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_resampled).value_counts())


# -----------------------------
# 5️⃣ Logistic Regression
# -----------------------------
print("\n=========== LOGISTIC REGRESSION ===========")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_resampled, y_train_resampled)

y_pred_log = log_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_log))


# -----------------------------
# 6️⃣ Random Forest (Final Model)
# -----------------------------
print("\n=========== RANDOM FOREST ===========")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_resampled, y_train_resampled)

rf_pred = rf_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, rf_pred))


# -----------------------------
# 7️⃣ Save Final Model
# -----------------------------
joblib.dump(rf_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved successfully!")
print("Class distribution plot saved in model/class_distribution.png")