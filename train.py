# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib

# ============================
# 1. LOAD DATASET
# ============================
# Ganti nama file jika perlu
df = pd.read_csv("tb_dummy_500.csv", parse_dates=["symptom_onset_date", "diagnosis_date", "first_visit_date"])

print("Dataset loaded. Total rows:", len(df))

# ============================
# 2. FEATURE ENGINEERING
# ============================
# Hitung delay_days
df["delay_days"] = (df["diagnosis_date"] - df["symptom_onset_date"]).dt.days

# Label: long_delay = 1 jika > 30 hari
df["long_delay"] = (df["delay_days"] > 30).astype(int)

# ============================
# 3. FEATURE SELECTION
# ============================
features = [
    "age", "sex", "education_level", "socioeconomic_proxy",
    "cough_duration_days", "hemoptysis", "weight_loss",
    "fever_night_sweats", "smoking_status", "contact_with_TB_case",
    "comorbidity_diabetes", "comorbidity_HIV", "xray_findings",
    "distance_to_healthcare_km"
]

X = df[features]
y = df["long_delay"]

# ============================
# 4. PREPROCESSING
# ============================
numeric_features = ["age", "cough_duration_days", "distance_to_healthcare_km"]
categorical_features = ["sex", "education_level", "socioeconomic_proxy",
                        "smoking_status", "xray_findings"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough"
)

# ============================
# 5. SPLIT DATASET
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ============================
# 6. BUILD MODEL
# ============================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)

# ============================
# 7. EVALUATION
# ============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL PERFORMANCE ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))

# AUC bisa error jika test datanya cuma 1 kelas → catch error
try:
    print("AUC      :", roc_auc_score(y_test, y_proba))
except:
    print("AUC      : Could not be computed (only one class in test set).")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================
# 8. SAVE MODEL
# ============================
joblib.dump(model, "model_pipeline.joblib")
print("\nModel saved as model_pipeline.joblib")
