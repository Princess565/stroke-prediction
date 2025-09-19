# ================================
# Stroke Prediction Model Training
# ================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
import joblib

warnings.filterwarnings(action='ignore')

# ---------------------
# 1. Load dataset
# ---------------------
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
print(df.info())

# ---------------------
# 2. Preprocessing
# ---------------------
def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df = df.copy()
    df = df.drop('id', axis=1)

    # Binary encoding
    df['ever_married'] = df['ever_married'].replace({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].replace({'Rural': 0, 'Urban': 1})

    # One-hot encoding
    for column in ['gender', 'work_type', 'smoking_status']:
        df = onehot_encode(df, column=column)

    # Split into X and y
    y = df['stroke']
    X = df.drop('stroke', axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=1
    )

    # Impute missing values
    imputer = KNNImputer()
    imputer.fit(X_train)
    X_train = pd.DataFrame(imputer.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)

    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_inputs(df)

# ---------------------
# 3. Define models
# ---------------------
models = {
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "LinearSVC": LinearSVC(),
    "SVC_RBF": SVC(),
    "MLP": MLPClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# ---------------------
# 4. Handle imbalance (oversample)
# ---------------------
oversampled_data = pd.concat([X_train, y_train], axis=1).copy()
num_samples = y_train.value_counts()[0] - y_train.value_counts()[1]
new_samples = oversampled_data.query("stroke == 1").sample(num_samples, replace=True, random_state=1)
oversampled_data = pd.concat([oversampled_data, new_samples], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

y_train_oversampled = oversampled_data['stroke']
X_train_oversampled = oversampled_data.drop('stroke', axis=1)

# ---------------------
# 5. Train and evaluate
# ---------------------
print("\nModel Performance\n-----------------")
for name, model in models.items():
    model.fit(X_train_oversampled, y_train_oversampled)   # fit on oversampled
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred)
    print(f"{name:20s} Accuracy: {acc:.3f}%\tF1-Score: {f1:.5f}")

# ---------------------
# 6. Save the best model + scaler + features
# ---------------------
# (we choose CatBoost here)
best_model = models["CatBoost"]  # already fitted above

joblib.dump(best_model, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

print("\n✅ Model, scaler and feature columns saved successfully as:")
print("   stroke_model.pkl, scaler.pkl, feature_columns.pkl")
