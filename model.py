import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("customer_churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

print("Dataset shape:", df.shape)
print(df.head())

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

print("\nCategorical features:")
print(categorical_features.tolist())

print("\nNumeric features:")
print(numeric_features.tolist())

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

print("\nModel pipeline created.")

model.fit(X_train, y_train)

print("\nModel trained successfully.")

y_pred = model.predict(X_test)

print("\nPredictions generated.")
print(y_pred[:10])

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

print("\nRandom Forest pipeline created.")

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest predictions generated.")
print(rf_pred[:10])

rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.2%}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    (y_test == "Yes").astype(int),
    y_prob
)

print(f"\nLogistic Regression ROC-AUC: {roc_auc:.2f}")

import joblib

joblib.dump(model, "churn_model.pkl")

print("\nModel saved as churn_model.pkl")