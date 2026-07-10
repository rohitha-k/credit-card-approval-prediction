'''import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("dataset/final_credit_dataset.csv")

print(df.head())
print(df.shape)

# ============================================
# REMOVE UNNECESSARY COLUMN
# ============================================

if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

# ============================================
# HANDLE MISSING VALUES
# ============================================

df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# ============================================
# FEATURE ENGINEERING
# ============================================

df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)

df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(
    lambda x: 0 if x > 0 else abs(x) / 365
).astype(int)

df.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)

if "FLAG_MOBIL" in df.columns:
    if df["FLAG_MOBIL"].nunique() == 1:
        df.drop("FLAG_MOBIL", axis=1, inplace=True)

# ============================================
# LABEL ENCODING
# ============================================

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

joblib.dump(encoders, "encoders.pkl")

print("Encoders Saved")

# ============================================
# FEATURES & TARGET
# ============================================

X = df.drop("Approved", axis=1)
y = df["Approved"]

joblib.dump(list(X.columns), "feature_names.pkl")

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================
# TRAIN MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ============================================
# EVALUATION
# ============================================

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(10,8),
    title="Feature Importance"
)

plt.tight_layout()
plt.show()'''
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/final_credit_dataset.csv")

print(df.head())
print("\nShape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# REMOVE ID COLUMN
# ==========================================================

if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\nPerforming Feature Engineering...")

df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)

df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(
    lambda x: 0 if x > 0 else abs(x) / 365
).astype(int)

df.drop(
    ["DAYS_BIRTH", "DAYS_EMPLOYED"],
    axis=1,
    inplace=True
)

# Remove constant column

if "FLAG_MOBIL" in df.columns:
    if df["FLAG_MOBIL"].nunique() == 1:
        df.drop("FLAG_MOBIL", axis=1, inplace=True)

print("Feature Engineering Completed.")

# ==========================================================
# LABEL ENCODING
# ==========================================================

categorical_columns = [

    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"

]

encoders = {}

print("\nEncoding Categorical Columns...")

for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = encoder

joblib.dump(encoders, "encoders.pkl")

print("Encoders Saved Successfully.")

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Approved", axis=1)

y = df["Approved"]

joblib.dump(
    list(X.columns),
    "feature_names.pkl"
)

print("\nFeature Names Saved.")

print("\nFeatures Used")

for feature in X.columns:
    print(feature)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================================
# APPLY SMOTE
# ==========================================================

print("\nBefore SMOTE")

print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE")

print(y_train.value_counts())

print("\nDataset Balanced Successfully.")
# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )

}

# ==========================================================
# TRAIN MODELS
# ==========================================================

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

print("\n")
print("="*70)
print("MODEL TRAINING STARTED")
print("="*70)

for name, model in models.items():

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    # Train Model
    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy
        }
    )

    print("Accuracy :", round(accuracy,4))

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            prediction
        )
    )

    # Save Best Model

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(results_df)

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(
    best_model,
    "model.pkl"
)

print("\n")
print("="*70)
print("BEST MODEL SAVED")
print("="*70)

print("Model :", best_model_name)

print("Accuracy :", round(best_accuracy,4))

# ==========================================================
# ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xlabel("Models")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("static/model_comparison.png")
plt.close()
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(ascending=False)

    print("\n")
    print("=" * 70)
    print("TOP 10 IMPORTANT FEATURES")
    print("=" * 70)

    print(importance.head(10))

    plt.figure(figsize=(10,6))

    importance.head(10).plot(
        kind="bar",
        color="steelblue"
    )

    plt.title("Top 10 Feature Importance")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("static/feature_importance.png")
    plt.close()

# ==========================================================
# ROC CURVE
# ==========================================================

if hasattr(best_model, "predict_proba"):

    probabilities = best_model.predict_proba(X_test)[:,1]

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nROC AUC Score :", round(auc_score,4))

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(figsize=(7,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("static/roc_curve.png")
    plt.close()

# ==========================================================
# SAVE MODEL COMPARISON
# ==========================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved as model_comparison.csv")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Dataset Shape        : {df.shape}")
print(f"Training Samples     : {len(X_train)}")
print(f"Testing Samples      : {len(X_test)}")
print(f"Best Model           : {best_model_name}")
print(f"Best Accuracy        : {round(best_accuracy*100,2)}%")

if hasattr(best_model, "predict_proba"):
    print(f"ROC-AUC Score        : {round(auc_score,4)}")

print("\nGenerated Files")

print("[OK] model.pkl")
print("[OK] encoders.pkl")
print("[OK] feature_names.pkl")
print("[OK] model_comparison.csv")

print("\nThank You!")