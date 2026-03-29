import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

print("Loading dataset...")

# Load dataset
data = pd.read_csv("data/UNSW_NB15_training-set.csv")

# Fill missing attack categories
data['attack_cat'] = data['attack_cat'].fillna('Normal')

# Selected features + label
cols = [
    'dur','spkts','dpkts','sbytes','dbytes',
    'rate','sttl','dttl','sload','dload',
    'attack_cat'
]

data = data[cols].dropna()

X = data.drop('attack_cat', axis=1)
y_text = data['attack_cat']

# Encode attack labels
le = LabelEncoder()
y = le.fit_transform(y_text)

print("Attack Classes:", list(le.classes_))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base models
rf = RandomForestClassifier(
    n_estimators=60,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    n_jobs=-1
)

ann = MLPClassifier(
    hidden_layer_sizes=(64,),
    max_iter=250,
    random_state=42
)

# Meta model
meta_xgb = XGBClassifier(
    n_estimators=60,
    max_depth=3,
    learvning_rate=0.1,
    objective="multi:softmax",
    num_class=len(le.classes_),
    eal_metric="mlogloss",
    n_jobs=-1
)

# Stacking model
stack = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('ann', ann)
    ],
    final_estimator=meta_xgb,
    cv=3,
    n_jobs=-1
)

print("Training stacking model...")
stack.fit(X_train, y_train)

# Evaluation
pred = stack.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, pred))

print("Saving models for Django...")

# ✅ CORRECT SAVE PATH (matches detection code)
joblib.dump(stack, "../ml_training/ml_models/stacking_model.pkl")
joblib.dump(scaler, "../ml_training/ml_models/scaler.pkl")
joblib.dump(le, "../ml_training/ml_models/label_encoder.pkl")

print("Training completed successfully.")
